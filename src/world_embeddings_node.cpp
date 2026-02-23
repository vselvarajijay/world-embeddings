#include "world_embeddings/association.hpp"
#include "world_embeddings/detector.hpp"
#include "world_embeddings/encoder.hpp"
#include "world_embeddings/entity_store.hpp"
#include "world_embeddings/snapshot.hpp"
#include "world_embeddings/types.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <memory>
#include <string>

namespace world_embeddings {

class WorldEmbeddingsNode : public rclcpp::Node {
 public:
  WorldEmbeddingsNode() : Node("world_embeddings_node") {
    declare_parameter<std::string>("image_topic", "/camera/image_raw");
    declare_parameter<std::string>("snapshot_path", "/tmp/world_embeddings_snapshot.json");
    declare_parameter<double>("snapshot_interval_sec", 30.0);
    declare_parameter<std::string>("yolo_onnx_path", "");
    declare_parameter<std::string>("clip_onnx_path", "");
    declare_parameter<double>("association_match_threshold", 0.7);
    declare_parameter<double>("association_max_pose_distance", 5.0);
    declare_parameter<double>("association_weight_embedding", 0.8);
    declare_parameter<double>("association_weight_pose", 0.2);
    declare_parameter<double>("ema_alpha", 0.9);

    std::string image_topic = get_parameter("image_topic").as_string();
    snapshot_path_ = get_parameter("snapshot_path").as_string();
    snapshot_interval_sec_ = get_parameter("snapshot_interval_sec").as_double();
    std::string yolo_path = get_parameter("yolo_onnx_path").as_string();
    std::string clip_path = get_parameter("clip_onnx_path").as_string();

    AssociationParams ap;
    ap.match_threshold = static_cast<float>(get_parameter("association_match_threshold").as_double());
    ap.max_pose_distance = static_cast<float>(get_parameter("association_max_pose_distance").as_double());
    ap.weight_embedding = static_cast<float>(get_parameter("association_weight_embedding").as_double());
    ap.weight_pose = static_cast<float>(get_parameter("association_weight_pose").as_double());
    association_ = std::make_unique<AssociationEngine>(ap);

    EntityStoreParams esp;
    esp.ema_alpha = static_cast<float>(get_parameter("ema_alpha").as_double());
    store_ = std::make_unique<EntityStore>(esp);

    if (!yolo_path.empty()) {
      YoloDetectorParams yp;
      yp.onnx_path = yolo_path;
      detector_ = make_yolo_detector(yp);
    }
    if (!clip_path.empty()) {
      ClipEncoderParams cp;
      cp.onnx_path = clip_path;
      encoder_ = make_clip_encoder(cp);
    }

    sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic, 10, std::bind(&WorldEmbeddingsNode::on_image, this, std::placeholders::_1));
    snapshot_timer_ = create_wall_timer(
        std::chrono::duration<double>(snapshot_interval_sec_),
        std::bind(&WorldEmbeddingsNode::on_snapshot_timer, this));
  }

  ~WorldEmbeddingsNode() {
    write_snapshot(snapshot_path_, store_->get_all_entities());
  }

 private:
  void on_image(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!detector_ || !encoder_) return;
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }
    cv::Mat img = cv_ptr->image;
    auto detections = detector_->detect(img);
    uint64_t tick = static_cast<uint64_t>(msg->header.stamp.sec) * 1000000000ULL +
                    static_cast<uint64_t>(msg->header.stamp.nanosec);
    if (tick == 0) tick = ++tick_counter_;

    for (const auto& d : detections) {
      cv::Mat crop = img(d.box);
      if (crop.empty()) continue;
      Embedding obs_emb = encoder_->encode(crop);
      if (obs_emb.size() == 0) continue;

      Pose3D obs_pose;
      obs_pose.x = d.box.x + d.box.width / 2.f;
      obs_pose.y = d.box.y + d.box.height / 2.f;
      obs_pose.z = 0.f;

      auto candidates = store_->get_candidates();
      auto match = association_->find_match(obs_emb, obs_pose, candidates);
      if (match) {
        store_->update_entity(*match, obs_emb, obs_pose, tick, 0.9f);
      } else {
        store_->add_entity(obs_emb, obs_pose, tick, 1.0f, d.label);
      }
    }
  }

  void on_snapshot_timer() {
    if (write_snapshot(snapshot_path_, store_->get_all_entities()))
      RCLCPP_INFO(get_logger(), "Snapshot written to %s (%zu entities)",
                  snapshot_path_.c_str(), store_->get_all_entities().size());
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr snapshot_timer_;
  std::string snapshot_path_;
  double snapshot_interval_sec_;
  uint64_t tick_counter_ = 0;

  std::unique_ptr<AssociationEngine> association_;
  std::unique_ptr<EntityStore> store_;
  std::unique_ptr<IDetector> detector_;
  std::unique_ptr<IEncoder> encoder_;
};

}  // namespace world_embeddings

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<world_embeddings::WorldEmbeddingsNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
