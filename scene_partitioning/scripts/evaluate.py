from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_partition_metrics(cluster_id, gids_in_cluster, gid_to_cluster_id, pts, colours, density, adj, adj_idx):
    num_pts_in_cluster = int(gids_in_cluster.size)

    if num_pts_in_cluster == 0:
        return {
            "cluster_id": int(cluster_id),
            "num_local_pts": 0,
            "num_neighbor_pts": 0,
            "num_adjacency_lengths": 0,
            "num_neighbor_clusters": 0,
        }

    global_to_local_id = np.full(pts.shape[0], 0xFFFFFFFF, dtype=np.uint32)
    global_to_local_id[gids_in_cluster] = np.arange(num_pts_in_cluster, dtype=np.uint32)

    neighbor_map = {}
    neighbor_global_ids = []
    neighbor_cluster_ids = set()
    edges_written = 0

    for global_id in gids_in_cluster:
        nbr_start = 0 if global_id == 0 else adj_idx[global_id - 1]
        nbr_end   = adj_idx[global_id]

        for neighbor_gid in adj[nbr_start:nbr_end]:
            neighbor_cluster_id = gid_to_cluster_id[neighbor_gid]
            if neighbor_cluster_id != cluster_id:
                neighbor_cluster_ids.add(int(neighbor_cluster_id))
                # assign stable local index for external neighbors
                nb_idx = neighbor_map.setdefault(
                    int(neighbor_gid), num_pts_in_cluster + len(neighbor_global_ids)
                )
                if nb_idx == num_pts_in_cluster + len(neighbor_global_ids):
                    neighbor_global_ids.append(int(neighbor_gid))

        edges_written += int(nbr_end - nbr_start)

    return {
        "cluster_id": int(cluster_id),
        "num_local_pts": num_pts_in_cluster,
        "num_neighbor_pts": int(len(neighbor_global_ids)),
        "num_adjacency_lengths": int(edges_written),
        "num_neighbor_clusters": int(len(neighbor_cluster_ids)),
    }

def compute_all_partition_metrics(partition_indices, pts, colours, density, adj, adj_idx, verbose=False):
    metrics_list = []

    n_pts = len(pts)
    gid_to_cluster_id = np.empty(n_pts, dtype=np.uint32)
    for p, inds in enumerate(partition_indices):
        if(len(inds) > 0):
            gid_to_cluster_id[inds] = p

    for cluster_id, gids_in_cluster in enumerate(partition_indices):
        result = compute_partition_metrics(
            cluster_id=cluster_id,
            gids_in_cluster=gids_in_cluster,
            pts=pts,
            colours=colours,
            density=density,
            adj=adj,
            adj_idx=adj_idx,
            gid_to_cluster_id=gid_to_cluster_id
        )
        if verbose and (cluster_id % 128 == 0 or cluster_id == len(partition_indices) - 1):
            print(f"part {result['cluster_id']:4d}: "
                  f"{result['num_local_pts']:7,} local | "
                  f"{result['num_neighbor_pts']:6,} neighbours from "
                  f"{result['num_neighbor_clusters']:2,} clusters | "
                  f"adjacency list {result['num_adjacency_lengths']:9,}")
        metrics_list.append(result)

    return metrics_list

def plot_partition_metrics(metrics, save_filename=None):
    """
    Plot standard metrics for partitions.

    Parameters:
        metrics : list[dict] or pd.DataFrame
            Output from compute_all_partition_metrics().
        save_filename : str | pathlib.Path | None
    """
    if isinstance(metrics, list):
        df = pd.DataFrame(metrics)
    else:
        df = metrics.copy()

    # keep plots in cluster order
    if "cluster_id" in df.columns:
        df = df.sort_values("cluster_id", kind="stable").reset_index(drop=True)

    # Resolve save path and ensure folder exists (pathlib-friendly)
    save_path = None
    if save_filename is not None:
        save_path = Path(save_filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.flatten()

    axes[0].bar(df["cluster_id"], df["num_local_pts"])
    axes[0].set_title("Local Cells")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("#Local Cells")

    axes[1].bar(df["cluster_id"], df["num_neighbor_pts"])
    axes[1].set_title("Neighbor Cells")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("#Neighbors")

    axes[2].bar(df["cluster_id"], df["num_adjacency_lengths"])
    axes[2].set_title("Adjacency")
    axes[2].set_xlabel("Cluster ID")
    axes[2].set_ylabel("#Adjacency")

    axes[3].bar(df["cluster_id"], df["num_neighbor_clusters"])
    axes[3].set_title("Neighbor Clusters")
    axes[3].set_xlabel("Cluster ID")
    axes[3].set_ylabel("#Neighbor Clusters")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
