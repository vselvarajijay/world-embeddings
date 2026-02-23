#include "world_embeddings/snapshot.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

namespace world_embeddings {

namespace {
nlohmann::json entity_to_json(const Entity& e) {
  nlohmann::json j;
  j["id"] = e.id;
  j["embedding"] = std::vector<float>(e.embedding.data(), e.embedding.data() + e.embedding.size());
  j["pose"] = {{"x", e.pose.x}, {"y", e.pose.y}, {"z", e.pose.z}};
  j["last_seen_tick"] = e.last_seen_tick;
  j["confidence"] = e.confidence;
  if (e.semantic_label) j["semantic_label"] = *e.semantic_label;
  j["history"] = nlohmann::json::array();
  for (const auto& u : e.history) {
    nlohmann::json h;
    h["tick"] = u.tick;
    h["observation_embedding"] = std::vector<float>(u.observation_embedding.data(),
                                                    u.observation_embedding.data() + u.observation_embedding.size());
    h["observed_pose"] = {{"x", u.observed_pose.x}, {"y", u.observed_pose.y}, {"z", u.observed_pose.z}};
    h["match_score"] = u.match_score;
    j["history"].push_back(std::move(h));
  }
  return j;
}
}  // namespace

std::string to_json(const std::vector<Entity>& entities) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto& e : entities) arr.push_back(entity_to_json(e));
  nlohmann::json root;
  root["entities"] = std::move(arr);
  return root.dump(2);
}

bool write_snapshot(const std::string& path, const std::vector<Entity>& entities) {
  std::ofstream f(path);
  if (!f) return false;
  f << to_json(entities);
  return f.good();
}

}  // namespace world_embeddings
