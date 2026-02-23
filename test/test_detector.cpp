#include "world_embeddings/detector.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <opencv2/imgproc.hpp>

namespace world_embeddings {
namespace {

TEST(Detector, MakeYoloDetectorEmptyPathReturnsNullptr) {
  YoloDetectorParams p;
  p.onnx_path = "";
  auto det = make_yolo_detector(p);
  EXPECT_EQ(det.get(), nullptr);
}

TEST(Detector, DetectWithRealModelIfAvailable) {
  const char* path = std::getenv("WORLD_EMBEDDINGS_YOLO_ONNX_PATH");
  if (!path || path[0] == '\0') {
    GTEST_SKIP() << "Set WORLD_EMBEDDINGS_YOLO_ONNX_PATH to run detector test";
  }
  YoloDetectorParams p;
  p.onnx_path = path;
  auto det = make_yolo_detector(p);
  ASSERT_NE(det.get(), nullptr);
  cv::Mat img(480, 640, CV_8UC3);
  img.setTo(cv::Scalar(200, 200, 200));
  auto results = det->detect(img);
  for (const auto& d : results) {
    EXPECT_GE(d.box.x, 0);
    EXPECT_GE(d.box.y, 0);
    EXPECT_GE(d.confidence, 0.f);
    EXPECT_LE(d.confidence, 1.f);
  }
}

}  // namespace
}  // namespace world_embeddings
