#include "world_embeddings/encoder.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace world_embeddings {

namespace {
// CLIP ViT-B/32 image normalization (ImageNet stats used by CLIP)
constexpr float CLIP_MEAN[] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float CLIP_STD[] = {0.26862954f, 0.26130258f, 0.27577711f};
}  // namespace

class ClipEncoder : public IEncoder {
 public:
  explicit ClipEncoder(ClipEncoderParams params)
      : params_(std::move(params)), env_(ORT_LOGGING_LEVEL_WARNING, "world_embeddings") {
    Ort::SessionOptions opts;
    session_ = std::make_unique<Ort::Session>(env_, params_.onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = std::string(session_->GetInputNameAllocated(0, alloc).get());
    output_name_ = std::string(session_->GetOutputNameAllocated(0, alloc).get());
  }

  Embedding encode(const cv::Mat& image) override {
    if (image.empty()) return Embedding(0);
    cv::Mat rgb;
    if (image.channels() == 3)
      cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    else
      rgb = image;
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(params_.input_size, params_.input_size));
    std::vector<float> input_data(1 * 3 * params_.input_size * params_.input_size);
    const int H = params_.input_size;
    const int W = params_.input_size;
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          input_data[c * H * W + y * W + x] =
              (resized.at<cv::Vec3b>(y, x)[c] / 255.0f - CLIP_MEAN[c]) / CLIP_STD[c];

    std::array<int64_t, 4> shape = {1, 3, params_.input_size, params_.input_size};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(),
                                                        shape.data(), shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto out = session_->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    float* out_data = out[0].GetTensorMutableData<float>();
    auto info = out[0].GetTensorTypeAndShapeInfo();
    size_t dim = info.GetElementCount();
    Embedding e(static_cast<Eigen::Index>(dim));
    for (size_t i = 0; i < dim; ++i) e(static_cast<Eigen::Index>(i)) = out_data[i];
    return e;
  }

  size_t embedding_dim() const override {
    return static_cast<size_t>(session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back());
  }

 private:
  ClipEncoderParams params_;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
};

std::unique_ptr<IEncoder> make_clip_encoder(const ClipEncoderParams& params) {
  if (params.onnx_path.empty()) return nullptr;
  return std::make_unique<ClipEncoder>(params);
}

}  // namespace world_embeddings
