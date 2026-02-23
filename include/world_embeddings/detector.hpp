#pragma once

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

namespace world_embeddings {

struct Detection {
  cv::Rect box;
  float confidence = 0.f;
  std::string label;
};

class IDetector {
 public:
  virtual ~IDetector() = default;
  virtual std::vector<Detection> detect(const cv::Mat& image) = 0;
};

struct YoloDetectorParams {
  std::string onnx_path;
  float confidence_threshold = 0.5f;
  float nms_threshold = 0.45f;
  int input_size = 640;
};

std::unique_ptr<IDetector> make_yolo_detector(const YoloDetectorParams& params);

}  // namespace world_embeddings
