#pragma once

#include "world_embeddings/types.hpp"
#include <vector>

namespace world_embeddings {

struct AssociationParams {
  float match_threshold = 0.7f;
  float max_pose_distance = 5.0f;
  float weight_embedding = 0.8f;
  float weight_pose = 0.2f;
};

class AssociationEngine {
 public:
  explicit AssociationEngine(AssociationParams params = {});

  std::optional<EntityId> find_match(
      const Embedding& observation_embedding,
      const Pose3D& observation_pose,
      const std::vector<CandidateEntity>& candidates) const;

 private:
  AssociationParams params_;
};

}  // namespace world_embeddings
