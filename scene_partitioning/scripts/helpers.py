import json
import copy
import struct
import h5py
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import open3d as o3d
from open3d.visualization import rendering

from tqdm import tqdm
from typing import List
from pathlib import Path
from plyfile import PlyData, PlyElement
from collections import defaultdict
from collections import deque

def generate_colors_from_cmap(n):
    cmap = plt.cm.get_cmap("hsv", n)
    return [cmap(i)[:3] for i in range(n)]  # remove alpha

def generate_random_colors(n, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, 3)

def visualize_partitions(partitioned_points, show_bounding_boxes=False):
    """
    Visualize point partitions with optional bounding boxes.

    Args:
        partitioned_points (List[np.ndarray]): list of (N_i, 3) point arrays.
        show_bounding_boxes (bool): whether to show bounding boxes around each partition.
    """
    all_points = np.vstack(partitioned_points)
    colors = generate_random_colors(len(partitioned_points))

    all_colors = np.vstack([
        np.tile(color, (len(part), 1)) for part, color in zip(partitioned_points, colors)
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    geometries = [pcd]

    if show_bounding_boxes:
        for part in partitioned_points:
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(part)
            )
            bbox.color = (0, 0, 0)  # black bounding box
            geometries.append(bbox)

    o3d.visualization.draw_geometries(geometries, window_name="Partition Visualization")
    return pcd

def rotation_to_align_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    enforce_upward: bool = True,
):
    """
    Estimate rotation that aligns the dominant plane in `pcd` so its normal matches z axis.

    Returns:
        R (3x3 np.ndarray): rotation matrix to apply with pcd.rotate(R, center=(0,0,0))
    """

    # Fit plane (ax + by + cz + d = 0)
    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, _ = plane_model
    n = np.array([a, b, c], dtype=float)

   # Desired normal (Z-axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    ground_normal = np.array([a, b, c])

    # Compute rotation axis and angle
    rotation_axis = np.cross(ground_normal, z_axis)
    rotation_angle = np.arccos(np.dot(ground_normal, z_axis) / 
                            (np.linalg.norm(ground_normal) * np.linalg.norm(z_axis)))

    # Normalize axis
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Create rotation matrix using Rodrigues' formula
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    R_flip = o3d.geometry.get_rotation_matrix_from_axis_angle([math.pi, 0, 0])
    R_total = R @ R_flip

    return R_total

