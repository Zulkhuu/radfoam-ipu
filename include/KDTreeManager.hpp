#pragma  once

#include <vector>
#include <memory>
#include <random>
#include <string>

#include <highfive/H5File.hpp>
#include <spdlog/fmt/fmt.h>

#include "io/hdf5_types.hpp"
#include "geometry/primitives.hpp"   // GenericPoint, LocalPoint
#include "ipu/rf_config.hpp" 
#include "util/nanoflann.hpp"

class KDTreeManager {
public:
  explicit KDTreeManager(const std::string& h5Path) { loadPoints(h5Path); buildKD(); }

  const std::vector<radfoam::geometry::GenericPoint>& getPoints() const { return pts_; }

  radfoam::geometry::GenericPoint
  getNearestNeighbor(const glm::vec3& p) const {
    float q[3] = {p.x, p.y, p.z};
    size_t idx = nearestIndex(q);
    return pts_[idx];
  }

  radfoam::geometry::GenericPoint getRandomPoint() const {
    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> d(0, pts_.size() - 1);
    return pts_[d(rng)];
  }

  void printRandomPoint() const {
    auto pt = getRandomPoint();
    fmt::print("Random point  ({:+.4f},{:+.4f},{:+.4f})  cluster={}  local={}\n",
               pt.x, pt.y, pt.z, pt.cluster_id, pt.local_id);
  }

private:
  // nanoflann adaptor ---------------------------------------------------------
  struct CloudAdaptor {
    const std::vector<radfoam::geometry::GenericPoint>& pts;
    size_t kdtree_get_point_count() const { return pts.size(); }
    float  kdtree_get_pt(size_t i, size_t dim) const {
      switch (dim) { case 0: return pts[i].x; case 1: return pts[i].y; default: return pts[i].z; }
    }
    template<class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
  };
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, CloudAdaptor>, CloudAdaptor, 3>;

  // helpers -------------------------------------------------------------------
  void loadPoints(const std::string& h5) {
    using namespace radfoam::geometry;
    HighFive::File f(h5, HighFive::File::ReadOnly);
    pts_.clear();
    for (size_t tid=0; tid<radfoam::config::kNumRayTracerTiles; ++tid) {
      auto g = f.getGroup(fmt::format("part{:04}", tid));
      std::vector<LocalPoint> locals; g.getDataSet("local_pts").read(locals);
      for (size_t i=0;i<locals.size();++i) {
        const auto& p = locals[i];
        pts_.push_back({p.x,p.y,p.z,static_cast<uint16_t>(tid),static_cast<uint16_t>(i)});
      }
    }
  }

  void buildKD() {
    adaptor_ = std::make_unique<CloudAdaptor>(CloudAdaptor{pts_});
    kdtree_  = std::make_unique<KDTree>(3,*adaptor_,
                   nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree_->buildIndex();
  }

  size_t nearestIndex(const float q[3]) const {
    size_t idx; float dist;
    nanoflann::KNNResultSet<float> rs(1); rs.init(&idx,&dist);
    kdtree_->findNeighbors(rs,q,nanoflann::SearchParameters());
    return idx;
  }

  // data members --------------------------------------------------------------
  std::vector<radfoam::geometry::GenericPoint> pts_;
  std::unique_ptr<CloudAdaptor> adaptor_;
  std::unique_ptr<KDTree>       kdtree_;
};
