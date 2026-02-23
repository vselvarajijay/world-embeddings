#include "world_embeddings/entity_store.hpp"
#include <algorithm>

namespace world_embeddings {

EntityStore::EntityStore(EntityStoreParams params) : params_(std::move(params)) {}

EntityId EntityStore::add_entity(const Embedding& observation_embedding,
                                const Pose3D& pose,
                                uint64_t tick,
                                float confidence,
                                std::optional<std::string> semantic_label) {
  EntityId id = next_id_++;
  Entity e;
  e.id = id;
  e.embedding = observation_embedding;
  e.pose = pose;
  e.last_seen_tick = tick;
  e.confidence = confidence;
  e.semantic_label = std::move(semantic_label);
  entities_[id] = std::move(e);
  return id;
}

void EntityStore::update_entity(EntityId id,
                               const Embedding& observation_embedding,
                               const Pose3D& pose,
                               uint64_t tick,
                               float match_score) {
  auto it = entities_.find(id);
  if (it == entities_.end()) return;
  Entity& e = it->second;

  float alpha = params_.ema_alpha;
  e.embedding = alpha * e.embedding + (1.f - alpha) * observation_embedding;
  e.pose = pose;
  e.last_seen_tick = tick;

  EntityUpdate u;
  u.tick = tick;
  u.observation_embedding = observation_embedding;
  u.observed_pose = pose;
  u.match_score = match_score;
  e.history.push_back(std::move(u));
  if (e.history.size() > params_.max_history_per_entity)
    e.history.erase(e.history.begin(),
                    e.history.begin() + static_cast<std::ptrdiff_t>(e.history.size() - params_.max_history_per_entity));
}

std::optional<Entity> EntityStore::get_entity(EntityId id) const {
  auto it = entities_.find(id);
  if (it == entities_.end()) return std::nullopt;
  return it->second;
}

std::vector<Entity> EntityStore::get_all_entities() const {
  std::vector<Entity> out;
  out.reserve(entities_.size());
  for (const auto& [_, e] : entities_) out.push_back(e);
  return out;
}

std::vector<CandidateEntity> EntityStore::get_candidates() const {
  std::vector<CandidateEntity> out;
  out.reserve(entities_.size());
  for (const auto& [_, e] : entities_) {
    CandidateEntity c;
    c.id = e.id;
    c.embedding = e.embedding;
    c.pose = e.pose;
    out.push_back(std::move(c));
  }
  return out;
}

}  // namespace world_embeddings
