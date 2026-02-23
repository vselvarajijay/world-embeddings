#include "world_embeddings/association.hpp"
#include "world_embeddings/types.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace world_embeddings {
namespace {

TEST(AssociationEngine, NoCandidatesReturnsNullopt) {
  AssociationParams p;
  p.match_threshold = 0.7f;
  AssociationEngine engine(p);
  Embedding obs(3);
  obs << 1.f, 0.f, 0.f;
  Pose3D obs_pose{0.f, 0.f, 0.f};
  auto match = engine.find_match(obs, obs_pose, {});
  EXPECT_FALSE(match.has_value());
}

TEST(AssociationEngine, MatchByEmbeddingAndPose) {
  AssociationParams p;
  p.match_threshold = 0.5f;
  p.max_pose_distance = 10.f;
  p.weight_embedding = 1.f;
  p.weight_pose = 0.f;
  AssociationEngine engine(p);

  Embedding obs(3);
  obs << 1.f, 0.f, 0.f;
  obs.normalize();
  Pose3D obs_pose{1.f, 1.f, 0.f};

  CandidateEntity c;
  c.id = 42;
  c.embedding = obs;
  c.pose = obs_pose;
  std::vector<CandidateEntity> candidates = {c};

  auto match = engine.find_match(obs, obs_pose, candidates);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(*match, 42u);
}

TEST(AssociationEngine, PoseGateRejectsFarEntity) {
  AssociationParams p;
  p.match_threshold = 0.5f;
  p.max_pose_distance = 0.5f;
  p.weight_embedding = 1.f;
  p.weight_pose = 0.f;
  AssociationEngine engine(p);

  Embedding obs(3);
  obs << 1.f, 0.f, 0.f;
  obs.normalize();
  Pose3D obs_pose{0.f, 0.f, 0.f};

  CandidateEntity c;
  c.id = 42;
  c.embedding = obs;
  c.pose = Pose3D{100.f, 100.f, 0.f};
  std::vector<CandidateEntity> candidates = {c};

  auto match = engine.find_match(obs, obs_pose, candidates);
  EXPECT_FALSE(match.has_value());
}

TEST(AssociationEngine, BelowThresholdReturnsNullopt) {
  AssociationParams p;
  p.match_threshold = 0.99f;
  p.max_pose_distance = 10.f;
  AssociationEngine engine(p);

  Embedding obs(3);
  obs << 1.f, 0.f, 0.f;
  obs.normalize();
  Embedding other(3);
  other << 0.f, 1.f, 0.f;
  other.normalize();
  Pose3D pose{0.f, 0.f, 0.f};

  CandidateEntity c;
  c.id = 1;
  c.embedding = other;
  c.pose = pose;
  auto match = engine.find_match(obs, pose, {c});
  EXPECT_FALSE(match.has_value());
}

}  // namespace
}  // namespace world_embeddings
