#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace world_embeddings {

using EntityId = uint64_t;

struct Pose3D {
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
};

using Embedding = Eigen::VectorXf;

struct EntityUpdate {
  uint64_t tick = 0;
  Embedding observation_embedding;
  Pose3D observed_pose;
  float match_score = 0.f;
};

struct Entity {
  EntityId id = 0;
  Embedding embedding;
  Pose3D pose;
  uint64_t last_seen_tick = 0;
  float confidence = 0.f;
  std::optional<std::string> semantic_label;
  std::vector<EntityUpdate> history;
};

struct CandidateEntity {
  EntityId id;
  Embedding embedding;
  Pose3D pose;
};

}  // namespace world_embeddings
