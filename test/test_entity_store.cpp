#include "world_embeddings/entity_store.hpp"
#include "world_embeddings/types.hpp"
#include <gtest/gtest.h>

namespace world_embeddings {
namespace {

TEST(EntityStore, AddEntityReturnsIdAndGetAll) {
  EntityStore store;
  Embedding e(4);
  e << 1.f, 0.f, 0.f, 0.f;
  Pose3D pose{1.f, 2.f, 0.f};
  EntityId id = store.add_entity(e, pose, 100, 1.0f, "chair");
  EXPECT_GT(id, 0u);
  auto all = store.get_all_entities();
  ASSERT_EQ(all.size(), 1u);
  EXPECT_EQ(all[0].id, id);
  EXPECT_EQ(all[0].embedding.size(), 4);
  EXPECT_FLOAT_EQ(all[0].pose.x, 1.f);
  EXPECT_FLOAT_EQ(all[0].pose.y, 2.f);
  EXPECT_EQ(all[0].last_seen_tick, 100u);
  ASSERT_TRUE(all[0].semantic_label.has_value());
  EXPECT_EQ(*all[0].semantic_label, "chair");
}

TEST(EntityStore, UpdateEntityEMA) {
  EntityStoreParams p;
  p.ema_alpha = 0.5f;
  EntityStore store(p);
  Embedding e0(2);
  e0 << 1.f, 0.f;
  EntityId id = store.add_entity(e0, Pose3D{0.f, 0.f, 0.f}, 1, 1.0f, std::nullopt);
  Embedding e1(2);
  e1 << 0.f, 1.f;
  store.update_entity(id, e1, Pose3D{1.f, 1.f, 0.f}, 2, 0.9f);
  auto ent = store.get_entity(id);
  ASSERT_TRUE(ent.has_value());
  EXPECT_FLOAT_EQ(ent->embedding(0), 0.5f);
  EXPECT_FLOAT_EQ(ent->embedding(1), 0.5f);
  EXPECT_EQ(ent->history.size(), 1u);
  EXPECT_EQ(ent->last_seen_tick, 2u);
}

TEST(EntityStore, GetEntityMissingReturnsNullopt) {
  EntityStore store;
  auto ent = store.get_entity(999);
  EXPECT_FALSE(ent.has_value());
}

TEST(EntityStore, GetCandidatesMatchesGetAll) {
  EntityStore store;
  Embedding e(2);
  e << 1.f, 0.f;
  store.add_entity(e, Pose3D{0.f, 0.f, 0.f}, 1, 1.0f, std::nullopt);
  auto candidates = store.get_candidates();
  auto all = store.get_all_entities();
  ASSERT_EQ(candidates.size(), all.size());
  EXPECT_EQ(candidates[0].id, all[0].id);
}

}  // namespace
}  // namespace world_embeddings
