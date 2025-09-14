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

def recursive_kd_partition_with_path(points, indices=None, depth=0, max_partitions=1024, path=''):
    if indices is None:
        indices = np.arange(len(points))

    if len(indices) == 0:
        return []

    if 2 ** depth >= max_partitions:
        return [(indices, path)]

    # Choose axis with highest standard deviation (spread)
    # axis = np.argmax(np.std(points, axis=0))
    iqr = np.percentile(points, 75, axis=0) - np.percentile(points, 25, axis=0)
    axis = np.argmax(iqr)

    sorted_idx = points[:, axis].argsort()
    mid = len(points) // 2

    left_idx = sorted_idx[:mid]
    right_idx = sorted_idx[mid:]

    left_points = points[left_idx]
    right_points = points[right_idx]

    left_partitions = recursive_kd_partition_with_path(
        left_points, indices[left_idx], depth + 1, max_partitions, path + '0'
    )
    right_partitions = recursive_kd_partition_with_path(
        right_points, indices[right_idx], depth + 1, max_partitions, path + '1'
    )

    return left_partitions + right_partitions

def group_partitions_by_prefix(partitions_with_paths, prefix_length):
    """
    Group KD-tree leaf partitions by a common prefix of their path.

    Args:
        partitions_with_paths: List of (indices, path) tuples from KD-tree.
        prefix_length: Number of bits in the path to use for grouping.

    Returns:
        Dictionary mapping prefix -> combined indices (as numpy arrays).
    """
    grouped = defaultdict(list)

    for indices, path in partitions_with_paths:
        prefix = path[:prefix_length]
        grouped[prefix].append(indices)

    # Concatenate index arrays for each group
    grouped_final = {
        prefix: np.concatenate(index_lists)
        for prefix, index_lists in grouped.items()
    }

    return grouped_final