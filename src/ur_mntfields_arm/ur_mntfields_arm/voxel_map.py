from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np


@dataclass
class FrontierCluster:
    centroid: np.ndarray
    voxels: list[tuple[int, int, int]]
    normal: np.ndarray


class SparseVoxelMap:
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2

    def __init__(
        self,
        voxel_size: float = 0.02,
        roi_min: np.ndarray | None = None,
        roi_max: np.ndarray | None = None,
        approach_envelope_m: float = 0.5,
    ):
        self.voxel_size = float(voxel_size)
        self.occupied: set[tuple[int, int, int]] = set()
        self.free: set[tuple[int, int, int]] = set()
        self.roi_min = None if roi_min is None else np.asarray(roi_min, dtype=np.float64).reshape(3)
        self.roi_max = None if roi_max is None else np.asarray(roi_max, dtype=np.float64).reshape(3)
        self.approach_envelope_m = max(0.0, float(approach_envelope_m))
        self.map_version = 0
        self._version_snapshot: dict[tuple[int, int, int], int] = {}

    def _key(self, point_xyz: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(point_xyz, dtype=np.float64) / self.voxel_size).astype(int).tolist())

    def _center(self, key: tuple[int, int, int]) -> np.ndarray:
        return (np.asarray(key, dtype=np.float64) + 0.5) * self.voxel_size

    def _within_mapping_bounds(self, point_xyz: np.ndarray) -> bool:
        if self.roi_min is None or self.roi_max is None:
            return True
        point = np.asarray(point_xyz, dtype=np.float64)
        return bool(np.all(point >= self.roi_min - self.approach_envelope_m) and np.all(point <= self.roi_max + self.approach_envelope_m))

    def state(self, point_xyz: np.ndarray) -> int:
        key = self._key(point_xyz)
        if key in self.occupied:
            return self.OCCUPIED
        if key in self.free:
            return self.FREE
        return self.UNKNOWN

    def is_observed_free(self, point_xyz: np.ndarray) -> bool:
        """Only observed or robot-verified free voxels may receive free labels."""
        return self.state(point_xyz) == self.FREE

    def integrate_known_free_points(
        self, points_xyz: np.ndarray, *, update_version: bool = True
    ) -> int:
        """Record points occupied by the known robot as environment-free.

        A self-filtered depth image cannot ray-observe the interiors of the
        robot links.  While the robot is physically at a configuration, each
        collision-sphere center is nevertheless known not to contain an
        external obstacle.  Recording those center voxels prevents every
        configuration from remaining unsupported while retaining the
        fail-closed coverage policy for the rest of unobserved workspace.
        """
        changed = 0
        for point in np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3):
            # Robot occupancy is direct free-space evidence even when a link
            # sits just outside the camera-mapping ROI/approach envelope.
            if not np.all(np.isfinite(point)):
                continue
            key = self._key(point)
            if key not in self.free or key in self.occupied:
                changed += 1
            self.occupied.discard(key)
            self.free.add(key)
        if changed and update_version:
            self._update_map_version()
        return changed

    def integrate_known_free_spheres(
        self,
        centers_xyz: np.ndarray,
        radii_m: np.ndarray,
        *,
        update_version: bool = True,
    ) -> int:
        """Carve the currently occupied robot volume as environment-free.

        Points inside the currently occupied robot volume are removed from the
        accumulated obstacle set, matching the per-frame robot self-filter.
        Filling each collision sphere, rather than only its center voxel, gives
        nearby sampled configurations support from the robot's observed swept
        volume while unknown space beyond that volume remains unsupported.
        """
        centers = np.asarray(centers_xyz, dtype=np.float64).reshape(-1, 3)
        radii = np.asarray(radii_m, dtype=np.float64).reshape(-1)
        if len(centers) != len(radii):
            raise ValueError("centers_xyz and radii_m must have equal length")
        changed = 0
        for center, radius in zip(centers, radii):
            if not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0.0:
                continue
            lo = np.floor((center - radius) / self.voxel_size).astype(int)
            hi = np.floor((center + radius) / self.voxel_size).astype(int)
            radius_sq = float(radius * radius)
            for x in range(int(lo[0]), int(hi[0]) + 1):
                for y in range(int(lo[1]), int(hi[1]) + 1):
                    for z in range(int(lo[2]), int(hi[2]) + 1):
                        key = (x, y, z)
                        point = self._center(key)
                        if float(np.dot(point - center, point - center)) > radius_sq:
                            continue
                        if key in self.occupied:
                            self.occupied.discard(key)
                            changed += 1
                        if key not in self.free:
                            self.free.add(key)
                            changed += 1
        if changed and update_version:
            self._update_map_version()
        return changed

    def set_roi(self, roi_min: np.ndarray, roi_max: np.ndarray, approach_envelope_m: float = 0.5) -> None:
        self.roi_min = np.asarray(roi_min, dtype=np.float64).reshape(3)
        self.roi_max = np.asarray(roi_max, dtype=np.float64).reshape(3)
        self.approach_envelope_m = max(0.0, float(approach_envelope_m))
        self.occupied = {key for key in self.occupied if self._within_mapping_bounds(self._center(key))}
        self.free = {key for key in self.free if self._within_mapping_bounds(self._center(key))}
        self._version_snapshot = {}
        self._update_map_version()

    def integrate_points(
        self,
        origin_xyz: np.ndarray,
        points_xyz: np.ndarray,
        max_free_steps: int = 200,
        *,
        update_version: bool = True,
    ):
        origin_xyz = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        if not len(points) or not np.all(np.isfinite(origin_xyz)):
            return

        # Accumulate the whole observation before mutating state. Endpoints
        # must win over free rays independent of pixel iteration order, while
        # ray-observed free space must clear stale occupied voxels left by
        # depth noise or earlier viewpoints.
        endpoint_keys: set[tuple[int, int, int]] = set()
        free_keys: set[tuple[int, int, int]] = set()
        for point in points:
            if not np.all(np.isfinite(point)) or not self._within_mapping_bounds(point):
                continue
            direction = point - origin_xyz
            dist = float(np.linalg.norm(direction))
            if dist < 1e-6:
                continue
            steps = min(max_free_steps, max(1, int(dist / (self.voxel_size * 0.75))))
            for step in range(steps):
                alpha = step / float(steps)
                ray_point = origin_xyz + alpha * direction
                if not self._within_mapping_bounds(ray_point):
                    continue
                free_keys.add(self._key(ray_point))
            endpoint_keys.add(self._key(point))

        free_keys.difference_update(endpoint_keys)
        self.occupied.difference_update(free_keys)
        self.free.update(free_keys)
        self.free.difference_update(endpoint_keys)
        self.occupied.update(endpoint_keys)
        if update_version:
            self._update_map_version()

    def update_map_version(self) -> bool:
        """Commit one version check after a compound map observation."""
        return self._update_map_version()

    def _roi_keys(self) -> list[tuple[int, int, int]]:
        if self.roi_min is None or self.roi_max is None:
            return sorted(self.occupied | self.free)
        lo = np.floor(self.roi_min / self.voxel_size).astype(int)
        hi = np.floor(self.roi_max / self.voxel_size).astype(int)
        return [
            (x, y, z)
            for x in range(int(lo[0]), int(hi[0]) + 1)
            for y in range(int(lo[1]), int(hi[1]) + 1)
            for z in range(int(lo[2]), int(hi[2]) + 1)
        ]

    def _update_map_version(self) -> bool:
        keys = self._roi_keys()
        if not keys:
            return False
        current = {
            key: self.OCCUPIED if key in self.occupied else self.FREE if key in self.free else self.UNKNOWN
            for key in keys
        }
        if not self._version_snapshot:
            changed_fraction = float(np.count_nonzero(np.fromiter(current.values(), dtype=np.int8))) / float(len(keys))
        else:
            changed_fraction = float(sum(current[key] != self._version_snapshot.get(key, self.UNKNOWN) for key in keys)) / float(len(keys))
        if changed_fraction > 0.01:
            self.map_version += 1
            self._version_snapshot = current
            return True
        return False

    def scene_signature(self) -> str:
        payload = {
            "voxel_size": self.voxel_size,
            "roi_min": None if self.roi_min is None else self.roi_min.tolist(),
            "roi_max": None if self.roi_max is None else self.roi_max.tolist(),
            "occupied": sorted(self.occupied),
            "free": sorted(self.free),
            "map_version": self.map_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            voxel_size=np.float64(self.voxel_size),
            occupied=np.asarray(sorted(self.occupied), dtype=np.int32).reshape(-1, 3),
            free=np.asarray(sorted(self.free), dtype=np.int32).reshape(-1, 3),
            roi_min=np.asarray([] if self.roi_min is None else self.roi_min, dtype=np.float64),
            roi_max=np.asarray([] if self.roi_max is None else self.roi_max, dtype=np.float64),
            approach_envelope_m=np.float64(self.approach_envelope_m),
            map_version=np.int64(self.map_version),
            scene_signature=np.asarray(self.scene_signature()),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SparseVoxelMap":
        with np.load(Path(path), allow_pickle=False) as data:
            roi_min = data["roi_min"]
            roi_max = data["roi_max"]
            result = cls(
                voxel_size=float(data["voxel_size"]),
                roi_min=None if roi_min.size == 0 else roi_min,
                roi_max=None if roi_max.size == 0 else roi_max,
                approach_envelope_m=float(data["approach_envelope_m"]),
            )
            result.occupied = {tuple(int(v) for v in row) for row in data["occupied"]}
            result.free = {tuple(int(v) for v in row) for row in data["free"]}
            result.map_version = int(data["map_version"])
            result._version_snapshot = {
                key: result.OCCUPIED if key in result.occupied else result.FREE
                for key in result.occupied | result.free
            }
            expected = str(data["scene_signature"])
        if result.scene_signature() != expected:
            raise ValueError(f"Voxel map signature mismatch: {path}")
        return result

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
