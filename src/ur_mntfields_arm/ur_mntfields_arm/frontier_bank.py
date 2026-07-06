from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ur_mntfields_arm.voxel_map import FrontierCluster


@dataclass
class FrontierRecord:
    frontier_id: int
    centroid: np.ndarray
    normal: np.ndarray
    voxels: list[tuple[int, int, int]]
    voxel_count: int
    status: str = "active"
    first_seen_step: int = 0
    last_seen_step: int = 0
    times_selected: int = 0
    times_failed: int = 0
    last_selected_step: int = -1


class FrontierBank:
    def __init__(self, match_radius_m: float = 0.18, visit_radius_m: float = 0.20, max_failures: int = 4):
        self.match_radius_m = float(match_radius_m)
        self.visit_radius_m = float(visit_radius_m)
        self.max_failures = int(max_failures)
        self._next_id = 1
        self.records: dict[int, FrontierRecord] = {}

    def active_records(self) -> list[FrontierRecord]:
        return [rec for rec in self.records.values() if rec.status == "active"]

    def local_active_records(self, step_idx: int, max_age_steps: int = 8) -> list[FrontierRecord]:
        step_idx = int(step_idx)
        max_age_steps = max(1, int(max_age_steps))
        return [
            rec
            for rec in self.records.values()
            if rec.status == "active" and (step_idx - rec.last_seen_step) <= max_age_steps
        ]

    def global_active_records(self, step_idx: int, min_age_steps: int = 9) -> list[FrontierRecord]:
        step_idx = int(step_idx)
        min_age_steps = max(0, int(min_age_steps))
        return [
            rec
            for rec in self.records.values()
            if rec.status == "active" and (step_idx - rec.last_seen_step) >= min_age_steps
        ]

    def update(self, clusters: list[FrontierCluster], step_idx: int):
        unmatched = set(self.records.keys())
        for cluster in clusters:
            best_id = None
            best_dist = float("inf")
            for frontier_id, rec in self.records.items():
                dist = float(np.linalg.norm(rec.centroid - cluster.centroid))
                if dist < self.match_radius_m and dist < best_dist:
                    best_dist = dist
                    best_id = frontier_id
            if best_id is None:
                frontier_id = self._next_id
                self._next_id += 1
                self.records[frontier_id] = FrontierRecord(
                    frontier_id=frontier_id,
                    centroid=cluster.centroid.copy(),
                    normal=cluster.normal.copy(),
                    voxels=list(cluster.voxels),
                    voxel_count=len(cluster.voxels),
                    first_seen_step=step_idx,
                    last_seen_step=step_idx,
                )
                continue
            rec = self.records[best_id]
            rec.centroid = cluster.centroid.copy()
            rec.normal = cluster.normal.copy()
            rec.voxels = list(cluster.voxels)
            rec.voxel_count = len(cluster.voxels)
            rec.last_seen_step = step_idx
            if rec.status == "retired":
                rec.status = "active"
            unmatched.discard(best_id)

        for frontier_id in unmatched:
            rec = self.records[frontier_id]
            if rec.status == "active" and (step_idx - rec.last_seen_step) > 50:
                rec.status = "retired"

    def mark_selected(self, frontier_id: int, step_idx: int | None = None):
        if frontier_id in self.records:
            self.records[frontier_id].times_selected += 1
            if step_idx is not None:
                self.records[frontier_id].last_selected_step = int(step_idx)

    def mark_failed(self, frontier_id: int):
        if frontier_id not in self.records:
            return
        rec = self.records[frontier_id]
        rec.times_failed += 1
        if rec.times_failed >= self.max_failures:
            rec.status = "retired"

    def mark_visited_near(self, xyz: np.ndarray):
        xyz = np.asarray(xyz, dtype=np.float64)
        for rec in self.records.values():
            if rec.status != "active":
                continue
            if float(np.linalg.norm(rec.centroid - xyz)) <= self.visit_radius_m:
                rec.status = "visited"
