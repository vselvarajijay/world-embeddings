#pragma once

#include "world_embeddings/types.hpp"
#include <opencv2/core/mat.hpp>
#include <memory>

namespace world_embeddings {

class IEncoder {
 public:
  virtual ~IEncoder() = default;
  virtual Embedding encode(const cv::Mat& image) = 0;
  virtual size_t embedding_dim() const = 0;
};

struct ClipEncoderParams {
  std::string onnx_path;
  int input_size = 224;
};

std::unique_ptr<IEncoder> make_clip_encoder(const ClipEncoderParams& params);

}  // namespace world_embeddings
