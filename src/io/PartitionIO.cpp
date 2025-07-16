#include "io/PartitionIO.hpp"
#include <highfive/H5File.hpp>
#include <sstream>

using namespace HighFive;

PartitionData load_partition(const std::string& hdf5_path,
                             uint16_t           rank_id)
{
  // ───  open file & group ─────────────────────────────────────
  File file(hdf5_path, File::ReadOnly);
  std::ostringstream grp_name;
  grp_name << "part" << std::setfill('0') << std::setw(4) << rank_id;

  if (!file.exist(grp_name.str()))
    throw std::runtime_error("Partition group \"" + grp_name.str()
                             + "\" not found in " + hdf5_path);

  auto grp = file.getGroup(grp_name.str());

  // ───  read datasets straight into std::vector<T> ────────────
  PartitionData out;

  auto ds_local = grp.getDataSet("local_pts");
  ds_local.read(out.local_pts);              // raw bytes → struct vector

  auto ds_neigh = grp.getDataSet("neighbor_pts");
  ds_neigh.read(out.neighbor_pts);

  auto ds_csr = grp.getDataSet("csr_indices");
  const hsize_t csr_elems = ds_csr.getElementCount();
  out.csr_raw.resize(csr_elems * ds_csr.getDataType().getSize());
  ds_csr.read(out.csr_raw.data());

  // sanity: last adj_end == csr_elems
  if (!out.local_pts.empty() &&
      out.local_pts.back().adj_end != csr_elems)
    throw std::runtime_error("CSR length mismatch in partition "
                             + std::to_string(rank_id));

  return out;
}
