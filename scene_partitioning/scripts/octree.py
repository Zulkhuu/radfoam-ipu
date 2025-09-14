import json
import copy
import struct
import h5py
import math
import heapq

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


class OctreeNode:
    def __init__(self, indices, bbox_min, bbox_max, depth=0):
        self.indices = indices
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.children = []
        self.depth = depth
        
def build_octree_max_points(points, max_points_per_node=5000):
    """
    Build an Octree where no leaf has more than `max_points_per_node`.
    """
    N = len(points)
    root = OctreeNode(
        indices=np.arange(N),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        depth=0
    )

    leaf_nodes = []
    queue = deque([root])

    while queue:
        node = queue.popleft()

        if len(node.indices) <= max_points_per_node:
            leaf_nodes.append(node)
            continue

        mid = (node.bbox_min + node.bbox_max) / 2.0
        children_indices = [[] for _ in range(8)]

        for idx in node.indices:
            pt = points[idx]
            octant = 0
            if pt[0] > mid[0]: octant |= 1
            if pt[1] > mid[1]: octant |= 2
            if pt[2] > mid[2]: octant |= 4
            children_indices[octant].append(idx)

        for i in range(8):
            if not children_indices[i]:
                continue
            new_min = node.bbox_min.copy()
            new_max = node.bbox_max.copy()
            if i & 1: new_min[0] = mid[0]
            else:     new_max[0] = mid[0]
            if i & 2: new_min[1] = mid[1]
            else:     new_max[1] = mid[1]
            if i & 4: new_min[2] = mid[2]
            else:     new_max[2] = mid[2]

            child_node = OctreeNode(
                indices=np.array(children_indices[i]),
                bbox_min=new_min,
                bbox_max=new_max,
                depth=node.depth + 1
            )
            queue.append(child_node)

    return [node.indices for node in leaf_nodes]


def build_octree_fixed_depth(points, max_depth=5):
    """
    Build an Octree with fixed maximum depth. Stops subdivision at `max_depth`.
    Returns a list of leaf node indices.
    """
    N = len(points)
    root = OctreeNode(
        indices=np.arange(N),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        depth=0
    )

    leaf_nodes = []
    queue = deque([root])

    while queue:
        node = queue.popleft()

        if node.depth >= max_depth:
            leaf_nodes.append(node)
            continue

        mid = (node.bbox_min + node.bbox_max) / 2.0
        children_indices = [[] for _ in range(8)]

        for idx in node.indices:
            pt = points[idx]
            octant = 0
            if pt[0] > mid[0]: octant |= 1
            if pt[1] > mid[1]: octant |= 2
            if pt[2] > mid[2]: octant |= 4
            children_indices[octant].append(idx)

        for i in range(8):
            if not children_indices[i]:
                continue
            new_min = node.bbox_min.copy()
            new_max = node.bbox_max.copy()
            if i & 1: new_min[0] = mid[0]
            else:     new_max[0] = mid[0]
            if i & 2: new_min[1] = mid[1]
            else:     new_max[1] = mid[1]
            if i & 4: new_min[2] = mid[2]
            else:     new_max[2] = mid[2]

            child_node = OctreeNode(
                indices=np.array(children_indices[i]),
                bbox_min=new_min,
                bbox_max=new_max,
                depth=node.depth + 1
            )
            queue.append(child_node)

    return [node.indices for node in leaf_nodes]


class OctreeNode2:
    def __init__(self, indices, bbox_min, bbox_max, depth=0):
        self.indices = indices
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.depth = depth
        self.children = []

    def __len__(self):
        return len(self.indices)

    def is_leaf(self):
        return not self.children

def build_octree_max_partitions(points, target_partitions=1024):
    """
    Build an octree where nodes are subdivided until the number of leaf nodes reaches `target_partitions`.
    Always subdivides the leaf node with the most points.
    Returns a list of point indices per leaf node.
    """
    N = len(points)

    # Pad root box to a cube
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half_size = np.max(bbox_max - bbox_min) / 2.0
    cube_min = center - half_size
    cube_max = center + half_size

    # Root node
    root = OctreeNode2(indices=np.arange(N), bbox_min=cube_min, bbox_max=cube_max, depth=0)
    leaves = [root]

    # Priority queue: (-num_points, node_index)
    heap = [(-len(root), 0)]

    while len(leaves) < target_partitions and heap:
        _, idx = heapq.heappop(heap)
        node = leaves[idx]

        # Subdivide the node
        mid = (node.bbox_min + node.bbox_max) / 2.0
        children_indices = [[] for _ in range(8)]

        for i in node.indices:
            pt = points[i]
            octant = 0
            if pt[0] > mid[0]: octant |= 1
            if pt[1] > mid[1]: octant |= 2
            if pt[2] > mid[2]: octant |= 4
            children_indices[octant].append(i)

        new_nodes = []
        for i in range(8):
            new_min = node.bbox_min.copy()
            new_max = node.bbox_max.copy()
            if i & 1: new_min[0] = mid[0]
            else:     new_max[0] = mid[0]
            if i & 2: new_min[1] = mid[1]
            else:     new_max[1] = mid[1]
            if i & 4: new_min[2] = mid[2]
            else:     new_max[2] = mid[2]

            child = OctreeNode2(
                indices=np.array(children_indices[i]),
                bbox_min=new_min,
                bbox_max=new_max,
                depth=node.depth + 1
            )
            new_nodes.append(child)

        # Replace the parent node with its children
        leaves[idx] = new_nodes[0]
        for child in new_nodes[1:]:
            leaves.append(child)

        # Add new non-empty children to the priority queue
        for i, child in enumerate(new_nodes):
            if len(child) > 0:
                heapq.heappush(heap, (-len(child), leaves.index(child)))

    return [leaf.indices for leaf in leaves]
