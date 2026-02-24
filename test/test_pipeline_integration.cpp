#include "world_embeddings/association.hpp"
#include "world_embeddings/entity_store.hpp"
#include "world_embeddings/snapshot.hpp"
#include "world_embeddings/types.hpp"
#include <gtest/gtest.h>

namespace world_embeddings {
namespace {

TEST(PipelineIntegration, AssociateUpdateSnapshot) {
  AssociationParams ap;
  ap.match_threshold = 0.6f;
  ap.max_pose_distance = 5.f;
  ap.weight_embedding = 1.f;
  ap.weight_pose = 0.f;
  AssociationEngine association(ap);
  EntityStore store;

  Embedding obs1(4);
  obs1 << 1.f, 0.f, 0.f, 0.f;
  obs1.normalize();
  Pose3D pose1{0.f, 0.f, 0.f};
  EntityId id1 = store.add_entity(obs1, pose1, 1, 1.0f, "chair");
  EXPECT_GT(id1, 0u);

  Embedding obs2(4);
  obs2 << 0.99f, 0.01f, 0.f, 0.f;
  obs2.normalize();
  Pose3D pose2{0.1f, 0.f, 0.f};
  auto match = association.find_match(obs2, pose2, store.get_candidates());
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(*match, id1);
  store.update_entity(*match, obs2, pose2, 2, 0.95f);

  auto entities = store.get_all_entities();
  ASSERT_EQ(entities.size(), 1u);
  EXPECT_EQ(entities[0].history.size(), 1u);

  std::string json = to_json(entities);
  EXPECT_FALSE(json.empty());
  EXPECT_NE(json.find("\"entities\""), std::string::npos);
  EXPECT_NE(json.find("\"embedding\""), std::string::npos);
}

TEST(PipelineIntegration, NewObservationCreatesNewEntity) {
  AssociationParams ap;
  ap.match_threshold = 0.9f;
  ap.max_pose_distance = 1.f;
  AssociationEngine association(ap);
  EntityStore store;

  Embedding obs1(3);
  obs1 << 1.f, 0.f, 0.f;
  obs1.normalize();
  store.add_entity(obs1, Pose3D{0.f, 0.f, 0.f}, 1, 1.0f, "a");

  Embedding obs2(3);
  obs2 << 0.f, 1.f, 0.f;
  obs2.normalize();
  auto match = association.find_match(obs2, Pose3D{0.f, 0.f, 0.f}, store.get_candidates());
  EXPECT_FALSE(match.has_value());
  store.add_entity(obs2, Pose3D{0.f, 0.f, 0.f}, 2, 1.0f, "b");
  auto all = store.get_all_entities();
  EXPECT_EQ(all.size(), 2u);
}

}  // namespace
}  // namespace world_embeddings
