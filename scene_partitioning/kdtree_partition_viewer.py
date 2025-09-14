#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from collections import defaultdict
import numpy as np
import pyvista as pv
from plyfile import PlyData
import open3d as o3d

from scripts.kdtree import recursive_kd_partition_with_path, group_partitions_by_prefix
from scripts.helpers import rotation_to_align_plane  # expects an open3d PointCloud -> 3x3 rotation

def read_ply_vertices(ply_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points[N,3] float32, colors[N,3] uint8) from a .ply."""
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if ply_path.suffix.lower() != ".ply":
        raise ValueError(f"Expected a .ply file, got {ply_path.suffix}")
    ply = PlyData.read(ply_path)
    vtx = ply["vertex"].data
    pts = np.column_stack([vtx["x"], vtx["y"], vtx["z"]]).astype(np.float32)
    cols = np.column_stack([vtx["red"], vtx["green"], vtx["blue"]]).astype(np.uint8)
    return pts, cols

def to_o3d_point_cloud(points: np.ndarray, colors_uint8: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """Create an Open3D PointCloud; normalize uint8 colors to [0,1]."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors_uint8 is not None:
        c = (colors_uint8.astype(np.float32) / 255.0).clip(0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return pcd

def build_lod_colors(
    n_points: int,
    partitions_with_paths: Sequence[Tuple[np.ndarray, str]],
    depths: Iterable[int],
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Build a list of color arrays (uint8, Nx3), one per depth.
    Child prefixes inherit parent color for continuity; otherwise new color is sampled.
    """
    rng = np.random.default_rng(seed)
    color_map: Dict[str | None, np.ndarray] = {None: rng.integers(0, 256, size=3, dtype=np.uint8)}  # root color
    color_layers: List[np.ndarray] = []

    for depth in depths:
        grouped = group_partitions_by_prefix(partitions_with_paths, prefix_length=depth)
        layer = np.zeros((n_points, 3), dtype=np.uint8)

        # organize children by parent prefix
        children_by_parent: Dict[str | None, List[str]] = defaultdict(list)
        for prefix in grouped.keys():
            parent = prefix[:-1] if depth > 1 else None
            children_by_parent[parent].append(prefix)

        # assign colors
        for parent, kids in children_by_parent.items():
            rng.shuffle(kids)  # random child inherits parent color
            parent_color = color_map.get(parent)
            for i, kid in enumerate(kids):
                if i == 0 and parent_color is not None:
                    c = parent_color
                else:
                    c = rng.integers(0, 256, size=3, dtype=np.uint8)
                color_map[kid] = c
                layer[grouped[kid]] = c

        color_layers.append(layer)

    return color_layers

def show_point_cloud_with_partitions(
    points: np.ndarray,
    original_colors: np.ndarray,
    partition_color_layers: List[np.ndarray],
    window_title: str = "Partitions Viewer",
    point_size: int = 4,
) -> None:
    """Adds keyboard toggles: 'c' next palette, 'd' previous palette."""
    pc = pv.PolyData(points)
    pc["original_colors"] = original_colors
    pc["active_colors"] = original_colors.copy()
    for i, layer in enumerate(partition_color_layers, start=1):
        pc[f"partition_depth{i}"] = layer

    palettes: List[np.ndarray] = [original_colors] + partition_color_layers
    state = {"i": 0, "palettes": palettes, "pc": pc}

    plotter = pv.Plotter(window_size=(1024, 768), title=window_title)
    plotter.add_points(pc, scalars="active_colors", rgb=True, point_size=point_size)
    plotter.add_text("Press 'c' / 'd' to cycle color layers", position="upper_left", font_size=12)

    def _apply(idx: int) -> None:
        arr = state["palettes"][idx]
        state["pc"]["active_colors"] = arr
        state["pc"].active_scalars_name = "active_colors"
        plotter.render()

    def _next():
        state["i"] = (state["i"] + 1) % len(state["palettes"])
        _apply(state["i"])

    def _prev():
        state["i"] = (state["i"] - 1) % len(state["palettes"])
        _apply(state["i"])

    plotter.add_key_event("c", _next)
    plotter.add_key_event("d", _prev)
    plotter.show()

def main(ply_file: Path | str, max_leaves: int = 1024, depths: Iterable[int] = range(1, 11)) -> None:
    ply_path = Path(ply_file)
    scene_name = ply_path.stem

    pts_unrot, cols_orig = read_ply_vertices(ply_path)

    pcd = to_o3d_point_cloud(pts_unrot, cols_orig)
    R = rotation_to_align_plane(pcd, distance_threshold=0.01) 

    pts = (pts_unrot @ R.T).astype(np.float32)

    leaves = recursive_kd_partition_with_path(pts, max_partitions=max_leaves)

    color_layers = build_lod_colors(n_points=len(pts), partitions_with_paths=leaves, depths=depths, seed=42)

    show_point_cloud_with_partitions(
        points=pts,
        original_colors=cols_orig,
        partition_color_layers=color_layers,
        window_title=f"{scene_name} partitions (KD, up to {max_leaves} leaves)",
        point_size=4,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KD-tree partition viewer (.ply input)")
    parser.add_argument("ply_file", type=Path, help="Full path to a .ply file (e.g. ./data/garden.ply)")
    parser.add_argument("-k", "--max-leaves", type=int, default=1024, help="Maximum KD-tree leaves (default: 1024)")
    args = parser.parse_args()
    main(ply_file=args.ply_file, max_leaves=args.max_leaves)
