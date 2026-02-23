#pragma once

#include "world_embeddings/types.hpp"
#include <string>
#include <vector>

namespace world_embeddings {

std::string to_json(const std::vector<Entity>& entities);
bool write_snapshot(const std::string& path, const std::vector<Entity>& entities);

}  // namespace world_embeddings
