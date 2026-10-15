#pragma once
namespace peel {
// Shared algorithm controls; arithmetic remains in the exact/fast policies.
struct PeelOptions {
  bool prune_rank_subtrees = true;
  bool merge_residuals = false;
};
} // namespace peel
