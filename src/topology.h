#pragma once
#include "geometry.h"
namespace peel {
std::vector<K::Segment_2> boundaries(const Domains &d);
std::vector<K::Segment_2> node(const std::vector<K::Segment_2> &segments, Stats *stats);
std::vector<std::array<P, 3>> triangle_cells(const std::vector<K::Segment_2> &planar_edges);
inline P centroid(const std::array<P, 3> &t) {
  return P((t[0].x() + t[1].x() + t[2].x()) / 3, (t[0].y() + t[1].y() + t[2].y()) / 3);
}
} // namespace peel
