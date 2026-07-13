from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class FrontierCluster:
    centroid: np.ndarray
    voxels: list[tuple[int, int, int]]
    normal: np.ndarray


class SparseVoxelMap:
    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = float(voxel_size)
        self.occupied: set[tuple[int, int, int]] = set()
        self.free: set[tuple[int, int, int]] = set()

    def _key(self, point_xyz: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(point_xyz, dtype=np.float64) / self.voxel_size).astype(int).tolist())

    def _center(self, key: tuple[int, int, int]) -> np.ndarray:
        return (np.asarray(key, dtype=np.float64) + 0.5) * self.voxel_size

    def integrate_points(self, origin_xyz: np.ndarray, points_xyz: np.ndarray, max_free_steps: int = 200):
        origin_xyz = np.asarray(origin_xyz, dtype=np.float64)
        if points_xyz.size == 0:
            return
        for point in np.asarray(points_xyz, dtype=np.float64):
            direction = point - origin_xyz
            dist = float(np.linalg.norm(direction))
            if dist < 1e-6:
                continue
            steps = min(max_free_steps, max(1, int(dist / (self.voxel_size * 0.75))))
            for step in range(steps):
                alpha = step / float(steps)
                key = self._key(origin_xyz + alpha * direction)
                if key not in self.occupied:
                    self.free.add(key)
            end_key = self._key(point)
            self.occupied.add(end_key)
            self.free.discard(end_key)

    def occupied_points(self) -> np.ndarray:
        if not self.occupied:
            return np.zeros((0, 3), dtype=np.float32)
        pts = [self._center(key) for key in sorted(self.occupied)]
        return np.asarray(pts, dtype=np.float32)

    def frontier_clusters(self) -> list[FrontierCluster]:
        frontier_keys = []
        nbrs = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        for key in self.free:
            has_unknown = False
            for dx, dy, dz in nbrs:
                nbr = (key[0] + dx, key[1] + dy, key[2] + dz)
                if nbr not in self.occupied and nbr not in self.free:
                    has_unknown = True
                    break
            # A frontier is the boundary between known free and unknown
            # space. Requiring occupied adjacency incorrectly collapses it
            # onto obstacle surfaces and produces occluded camera goals.
            if has_unknown:
                frontier_keys.append(key)

        frontier_set = set(frontier_keys)
        visited: set[tuple[int, int, int]] = set()
        clusters: list[FrontierCluster] = []
        for seed in frontier_keys:
            if seed in visited:
                continue
            queue = deque([seed])
            visited.add(seed)
            comp: list[tuple[int, int, int]] = []
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                            if nxt in frontier_set and nxt not in visited:
                                visited.add(nxt)
                                queue.append(nxt)
            pts = np.asarray([self._center(key) for key in comp], dtype=np.float64)
            centroid = pts.mean(axis=0)
            normal_accum = np.zeros(3, dtype=np.float64)
            for key in comp:
                p_free = self._center(key)
                for dx, dy, dz in nbrs:
                    nbr = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if nbr not in self.occupied and nbr not in self.free:
                        normal_accum += self._center(nbr) - p_free
            nrm = float(np.linalg.norm(normal_accum))
            if nrm < 1e-6:
                normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                normal = normal_accum / nrm
            clusters.append(FrontierCluster(centroid=centroid.astype(np.float32), voxels=comp, normal=normal.astype(np.float32)))
        return clusters
