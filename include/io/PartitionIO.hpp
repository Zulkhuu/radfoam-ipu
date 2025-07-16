#pragma once
/**
 *  Lightweight reader for the KD-partitioned HDF5 file produced by
 *  partition_from_ply.py.
 *
 *  The binary layout of the compound types exactly mirrors the packed
 *  structs described in the design doc, so no re-packing is required after
 *  reading: you can copy the vectors straight onto an IPU exchange tensor.
 */

#include <cstdint>
#include <string>
#include <vector>

struct alignas(8) LocalPoint {
  float    x, y, z;              // 12 B
  uint8_t  r, g, b, _pad;        //  4 B (incl. padding for 4-byte align)
  float    density;              //  4 B
  uint32_t adj_end;              //  4 B
};
static_assert(sizeof(LocalPoint) == 24, "LocalPoint must be 24 bytes");

struct NeighborPoint {
  float    x, y, z;              // 12 B
  uint32_t gid;                  //  4 B
};
static_assert(sizeof(NeighborPoint) == 16, "NeighborPoint must be 16 bytes");

// CSR index type is chosen per partition (16 or 32 bit)
using CsrIndex16 = uint16_t;
using CsrIndex32 = uint32_t;

struct PartitionData {
  std::vector<LocalPoint>     local_pts;
  std::vector<NeighborPoint>  neighbor_pts;
  // csr_indices is either 16- or 32-bit depending on the file
  std::vector<uint8_t>        csr_raw;     // opaque blob; cast after reading
};

/**
 *  Load one partition (rank) from the single HDF5 container.
 *
 *  @param hdf5_path Path to cloud.h5
 *  @param rank_id   Partition index 0-1023
 *  @returns         Filled PartitionData
 *  @throws          std::runtime_error on errors
 */
PartitionData load_partition(const std::string& hdf5_path,
                             uint16_t           rank_id);
