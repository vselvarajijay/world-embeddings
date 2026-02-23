#include "world_embeddings/snapshot.hpp"
#include "world_embeddings/types.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <nlohmann/json.hpp>

namespace world_embeddings {
namespace {

TEST(Snapshot, EmptyEntitiesProducesValidJson) {
  std::vector<Entity> entities;
  std::string json = to_json(entities);
  auto j = nlohmann::json::parse(json);
  EXPECT_TRUE(j.contains("entities"));
  EXPECT_TRUE(j["entities"].is_array());
  EXPECT_EQ(j["entities"].size(), 0u);
}

TEST(Snapshot, OneEntityRoundtrip) {
  Entity e;
  e.id = 1;
  e.embedding = Embedding(3);
  e.embedding << 0.1f, 0.2f, 0.3f;
  e.pose = Pose3D{1.f, 2.f, 3.f};
  e.last_seen_tick = 100;
  e.confidence = 0.9f;
  e.semantic_label = "chair";
  std::vector<Entity> entities = {e};

  std::string json = to_json(entities);
  auto j = nlohmann::json::parse(json);
  ASSERT_TRUE(j.contains("entities"));
  ASSERT_EQ(j["entities"].size(), 1u);
  auto& ent = j["entities"][0];
  EXPECT_EQ(ent["id"], 1u);
  EXPECT_EQ(ent["pose"]["x"], 1.0);
  EXPECT_EQ(ent["pose"]["y"], 2.0);
  EXPECT_EQ(ent["pose"]["z"], 3.0);
  EXPECT_EQ(ent["last_seen_tick"], 100u);
  EXPECT_EQ(ent["confidence"], 0.9);
  EXPECT_EQ(ent["semantic_label"], "chair");
  ASSERT_TRUE(ent["embedding"].is_array());
  EXPECT_EQ(ent["embedding"].size(), 3u);
}

TEST(Snapshot, WriteSnapshotToFile) {
  Entity e;
  e.id = 2;
  e.embedding = Embedding(2);
  e.embedding << 0.f, 1.f;
  e.pose = Pose3D{0.f, 0.f, 0.f};
  std::vector<Entity> entities = {e};
  std::string path = "/tmp/world_embeddings_test_snapshot.json";
  bool ok = write_snapshot(path, entities);
  EXPECT_TRUE(ok);
  std::ifstream f(path);
  ASSERT_TRUE(f.good());
  std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  f.close();
  auto j = nlohmann::json::parse(content);
  EXPECT_TRUE(j.contains("entities"));
  EXPECT_EQ(j["entities"].size(), 1u);
}

}  // namespace
}  // namespace world_embeddings
