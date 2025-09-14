    
import numpy as np
import pandas as pd
import h5py

local_point_dt = np.dtype([
    ("x",       "f4"), ("y", "f4"), ("z", "f4"),
    ("r",       "u1"), ("g", "u1"), ("b", "u1"), ("_pad", "u1"),  # _pad keeps 4-byte align
    ("density", "f4"),
    ("adj_end", "u4"),                                            # CSR exclusive offset
], align=True)                                                    # 24 B

neighbor_point_dt = np.dtype([
    ("x",         "f4"), ("y", "f4"), ("z", "f4"),
    ("cluster_id","u2"), ("local_id","u2"),                      
])

str_dt = h5py.string_dtype(encoding='utf-8')
    

def write_partition_hdf5(
    h5_filename: str,
    pts: np.ndarray,
    colours: np.ndarray,
    density: np.ndarray,
    kd_partitions_with_paths: list[tuple[np.ndarray, str]],
    adj_idx: np.ndarray,
    adj: np.ndarray,
    verbose_every: int = 128,
) -> dict:
    """
    Write partitioned point-cloud data + adjacency in a compact HDF5 layout.

    kd_partitions_with_paths: list of (indices, path)
      - indices: 1D array of global ids in the partition
      - path:    string path for that partition
    """

    paths = [path for _, path in kd_partitions_with_paths]
    partition_indices = [np.asarray(indices) for indices, _ in kd_partitions_with_paths]
    n_pts = pts.shape[0]
    n_clusters = len(partition_indices)

    gid_to_local_id = np.empty(n_pts, dtype=np.uint16)
    for inds in partition_indices:
        gid_to_local_id[inds] = np.arange(inds.size, dtype=np.uint16)

    gid_to_cluster_id = np.empty(n_pts, dtype=np.uint16)
    for p, inds in enumerate(partition_indices):
        gid_to_cluster_id[inds] = p

    local_sizes = []
    neighbor_sizes = []
    adjacency_lengths = []
    neighbor_cluster_counts = []
    global_to_local_id = np.full(n_pts, 0xFFFFFFFF, dtype=np.uint32)

    with h5py.File(h5_filename, "w", libver="earliest") as h5f:
        for cluster_id, gids_in_cluster in enumerate(partition_indices):
            gids_in_cluster = np.asarray(gids_in_cluster, dtype=np.uint32)
            num_pts_in_cluster = gids_in_cluster.size
            global_to_local_id[gids_in_cluster] = np.arange(num_pts_in_cluster, dtype=np.uint32)

            # local points
            local_pts = np.empty(num_pts_in_cluster, dtype=local_point_dt)
            local_pts["x"] = pts[gids_in_cluster, 0]
            local_pts["y"] = pts[gids_in_cluster, 1]
            local_pts["z"] = pts[gids_in_cluster, 2]
            local_pts["r"] = colours[gids_in_cluster, 0]
            local_pts["g"] = colours[gids_in_cluster, 1]
            local_pts["b"] = colours[gids_in_cluster, 2]
            local_pts["_pad"][:] = 0
            local_pts["density"] = density[gids_in_cluster]

            neighbor_map = {}
            neighbor_global_ids = []
            adjacency_list = []
            neighbor_cluster_ids = set()
            edges_written = 0

            # CSR build
            for local_idx, global_id in enumerate(gids_in_cluster):
                nbr_start = 0 if global_id == 0 else adj_idx[global_id - 1]
                nbr_end   = adj_idx[global_id]

                for neighbor_gid in adj[nbr_start:nbr_end]:
                    neighbor_cluster_id = gid_to_cluster_id[neighbor_gid]
                    if neighbor_cluster_id == cluster_id:
                        adjacency_list.append(global_to_local_id[neighbor_gid])
                    else:
                        neighbor_cluster_ids.add(neighbor_cluster_id)
                        nb_idx = neighbor_map.setdefault(
                            int(neighbor_gid),
                            num_pts_in_cluster + len(neighbor_global_ids)
                        )
                        if nb_idx == num_pts_in_cluster + len(neighbor_global_ids):
                            neighbor_global_ids.append(int(neighbor_gid))
                        adjacency_list.append(nb_idx)

                edges_written += (nbr_end - nbr_start)
                local_pts["adj_end"][local_idx] = edges_written

            # neighbor points
            neighbor_gids_np = np.asarray(neighbor_global_ids, dtype=np.uint32)
            neighbor_pts = np.empty(neighbor_gids_np.size, dtype=neighbor_point_dt)
            if neighbor_gids_np.size:
                neighbor_pts["x"] = pts[neighbor_gids_np, 0]
                neighbor_pts["y"] = pts[neighbor_gids_np, 1]
                neighbor_pts["z"] = pts[neighbor_gids_np, 2]
                neighbor_pts["cluster_id"] = gid_to_cluster_id[neighbor_gids_np]
                neighbor_pts["local_id"]   = gid_to_local_id[neighbor_gids_np]
            else:
                # keep fields initialized even if empty
                neighbor_pts["x"] = []
                neighbor_pts["y"] = []
                neighbor_pts["z"] = []
                neighbor_pts["cluster_id"] = []
                neighbor_pts["local_id"] = []

            # choose 16- or 32-bit CSR index type
            total_indexable = num_pts_in_cluster + neighbor_pts.size
            adjacency_dt = np.uint16 if total_indexable < 65_536 else np.uint32
            adjacency_list = np.asarray(adjacency_list, dtype=adjacency_dt)

            # write this partition
            grp = h5f.create_group(f"part{cluster_id:04d}")
            grp.create_dataset("local_pts", data=local_pts,
                               dtype=local_point_dt, compression="gzip", shuffle=True)
            grp.create_dataset("neighbor_pts", data=neighbor_pts,
                               dtype=neighbor_point_dt, compression="gzip", shuffle=True)
            grp.create_dataset("adjacency_list", data=adjacency_list,
                               dtype=adjacency_dt, compression="gzip", shuffle=True)
            grp.create_dataset("path", data=paths[cluster_id], dtype=str_dt)

            if verbose_every and (cluster_id % verbose_every == 0):
                print(f"part {cluster_id:4d}: {num_pts_in_cluster:7,} local | "
                      f"{neighbor_pts.size:6,} neighbours from {len(neighbor_cluster_ids):2,} clusters | "
                      f"adjacency list {edges_written:9,}")

            local_sizes.append(num_pts_in_cluster)
            neighbor_sizes.append(neighbor_pts.size)
            adjacency_lengths.append(edges_written)
            neighbor_cluster_counts.append(len(neighbor_cluster_ids))

    print(f"✓ {len(partition_indices)} partitions written to {h5_filename}")
    return pd.DataFrame({
        "cluster_id": np.arange(n_clusters, dtype=np.int32),
        "num_local_pts": np.asarray(local_sizes, dtype=np.int64),
        "num_neighbor_pts": np.asarray(neighbor_sizes, dtype=np.int64),
        "num_adjacency_lengths": np.asarray(adjacency_lengths, dtype=np.int64),
        "num_neighbor_clusters": np.asarray(neighbor_cluster_counts, dtype=np.int64),
        "num_partitions": len(partition_indices),
        "output": h5_filename,
    })
