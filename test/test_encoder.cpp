#include "world_embeddings/encoder.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <opencv2/imgproc.hpp>

namespace world_embeddings {
namespace {

TEST(Encoder, MakeClipEncoderEmptyPathReturnsNullptr) {
  ClipEncoderParams p;
  p.onnx_path = "";
  auto enc = make_clip_encoder(p);
  EXPECT_EQ(enc.get(), nullptr);
}

TEST(Encoder, EncodeWithRealModelIfAvailable) {
  const char* path = std::getenv("WORLD_EMBEDDINGS_CLIP_ONNX_PATH");
  if (!path || path[0] == '\0') {
    GTEST_SKIP() << "Set WORLD_EMBEDDINGS_CLIP_ONNX_PATH to run encoder test";
  }
  ClipEncoderParams p;
  p.onnx_path = path;
  auto enc = make_clip_encoder(p);
  ASSERT_NE(enc.get(), nullptr);
  cv::Mat crop(224, 224, CV_8UC3);
  crop.setTo(cv::Scalar(128, 128, 128));
  Embedding e1 = enc->encode(crop);
  Embedding e2 = enc->encode(crop);
  EXPECT_EQ(e1.size(), e2.size());
  EXPECT_GT(e1.size(), 0);
  for (Eigen::Index i = 0; i < e1.size(); ++i)
    EXPECT_FLOAT_EQ(e1(i), e2(i)) << "Deterministic repeatability";
  EXPECT_EQ(enc->embedding_dim(), static_cast<size_t>(e1.size()));
}

}  // namespace
}  // namespace world_embeddings
