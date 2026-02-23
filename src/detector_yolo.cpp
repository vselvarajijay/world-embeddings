#include "world_embeddings/detector.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace world_embeddings {

namespace {
// COCO class names (80 classes) - abbreviated for MVP; can be extended
const char* COCO_CLASSES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
constexpr int COCO_NUM_CLASSES = 80;
}  // namespace

class YoloDetector : public IDetector {
 public:
  explicit YoloDetector(YoloDetectorParams params)
      : params_(std::move(params)), env_(ORT_LOGGING_LEVEL_WARNING, "world_embeddings") {
    Ort::SessionOptions opts;
    session_ = std::make_unique<Ort::Session>(env_, params_.onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = std::string(session_->GetInputNameAllocated(0, alloc).get());
    output_name_ = std::string(session_->GetOutputNameAllocated(0, alloc).get());
  }

  std::vector<Detection> detect(const cv::Mat& image) override {
    if (image.empty()) return {};
    const int orig_h = image.rows;
    const int orig_w = image.cols;
    float scale = std::min(static_cast<float>(params_.input_size) / orig_w,
                           static_cast<float>(params_.input_size) / orig_h);
    int new_w = static_cast<int>(std::round(orig_w * scale));
    int new_h = static_cast<int>(std::round(orig_h * scale));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));
    cv::Mat padded(params_.input_size, params_.input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    std::vector<float> input_data(1 * 3 * params_.input_size * params_.input_size);
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < params_.input_size; ++y)
        for (int x = 0; x < params_.input_size; ++x)
          input_data[c * params_.input_size * params_.input_size + y * params_.input_size + x] =
              padded.at<cv::Vec3b>(y, x)[c] / 255.0f;

    std::array<int64_t, 4> shape = {1, 3, params_.input_size, params_.input_size};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(),
                                                        shape.data(), shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto out = session_->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    float* out_data = out[0].GetTensorMutableData<float>();
    auto info = out[0].GetTensorTypeAndShapeInfo();
    auto sh = info.GetShape();
    int num_channels = static_cast<int>(sh[1]);
    int num_proposals = static_cast<int>(sh[2]);
    if (num_channels < 5) return {};

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < num_proposals; ++i) {
      float cx = out_data[i];
      float cy = out_data[num_proposals + i];
      float w = out_data[2 * num_proposals + i];
      float h = out_data[3 * num_proposals + i];
      int best_class = 0;
      float best_score = out_data[4 * num_proposals + i];
      for (int c = 1; c < num_channels - 4; ++c) {
        float s = out_data[(4 + c) * num_proposals + i];
        if (s > best_score) {
          best_score = s;
          best_class = c;
        }
      }
      if (best_score < params_.confidence_threshold) continue;
      float scale_back = 1.f / scale;
      int left = static_cast<int>((cx - 0.5f * w) * scale_back);
      int top = static_cast<int>((cy - 0.5f * h) * scale_back);
      int width = static_cast<int>(w * scale_back);
      int height = static_cast<int>(h * scale_back);
      left = std::max(0, std::min(left, orig_w - 1));
      top = std::max(0, std::min(top, orig_h - 1));
      width = std::max(1, std::min(width, orig_w - left));
      height = std::max(1, std::min(height, orig_h - top));
      class_ids.push_back(best_class);
      confidences.push_back(best_score);
      boxes.push_back(cv::Rect(left, top, width, height));
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, params_.confidence_threshold, params_.nms_threshold, nms_indices);

    std::vector<Detection> result;
    result.reserve(nms_indices.size());
    for (int idx : nms_indices) {
      Detection d;
      d.box = boxes[idx];
      d.confidence = confidences[idx];
      d.label = (class_ids[idx] >= 0 && class_ids[idx] < COCO_NUM_CLASSES)
          ? COCO_CLASSES[class_ids[idx]]
          : ("class_" + std::to_string(class_ids[idx]));
      result.push_back(std::move(d));
    }
    return result;
  }

 private:
  YoloDetectorParams params_;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
};

std::unique_ptr<IDetector> make_yolo_detector(const YoloDetectorParams& params) {
  if (params.onnx_path.empty()) return nullptr;
  return std::make_unique<YoloDetector>(params);
}

}  // namespace world_embeddings
