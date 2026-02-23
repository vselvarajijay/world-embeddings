#include "world_embeddings/association.hpp"
#include <cmath>
#include <optional>

namespace world_embeddings {

AssociationEngine::AssociationEngine(AssociationParams params) : params_(std::move(params)) {}

static float cosine_similarity(const Embedding& a, const Embedding& b) {
  if (a.size() != b.size() || a.size() == 0) return 0.f;
  float dot = a.dot(b);
  float na = a.norm();
  float nb = b.norm();
  if (na < 1e-9f || nb < 1e-9f) return 0.f;
  return dot / (na * nb);
}

static float pose_distance(const Pose3D& a, const Pose3D& b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::optional<EntityId> AssociationEngine::find_match(
    const Embedding& observation_embedding,
    const Pose3D& observation_pose,
    const std::vector<CandidateEntity>& candidates) const {
  std::optional<EntityId> best_id;
  float best_score = params_.match_threshold;

  for (const auto& c : candidates) {
    float dist = pose_distance(c.pose, observation_pose);
    if (dist > params_.max_pose_distance) continue;

    float cos_sim = cosine_similarity(c.embedding, observation_embedding);
    float pose_score = (params_.max_pose_distance > 0)
        ? (1.f - std::min(dist / params_.max_pose_distance, 1.f))
        : 1.f;
    float score = params_.weight_embedding * cos_sim + params_.weight_pose * pose_score;

    if (score >= params_.match_threshold && score > best_score) {
      best_score = score;
      best_id = c.id;
    }
  }
  return best_id;
}

}  // namespace world_embeddings
