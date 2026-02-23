#pragma once

#include "world_embeddings/types.hpp"
#include <map>
#include <vector>

namespace world_embeddings {

struct EntityStoreParams {
  float ema_alpha = 0.9f;
  size_t max_history_per_entity = 50;
};

class EntityStore {
 public:
  explicit EntityStore(EntityStoreParams params = {});

  EntityId add_entity(const Embedding& observation_embedding,
                      const Pose3D& pose,
                      uint64_t tick,
                      float confidence = 1.0f,
                      std::optional<std::string> semantic_label = std::nullopt);

  void update_entity(EntityId id,
                    const Embedding& observation_embedding,
                    const Pose3D& pose,
                    uint64_t tick,
                    float match_score);

  std::optional<Entity> get_entity(EntityId id) const;
  std::vector<Entity> get_all_entities() const;

  std::vector<CandidateEntity> get_candidates() const;

 private:
  EntityId next_id_ = 1;
  std::map<EntityId, Entity> entities_;
  EntityStoreParams params_;
};

}  // namespace world_embeddings
