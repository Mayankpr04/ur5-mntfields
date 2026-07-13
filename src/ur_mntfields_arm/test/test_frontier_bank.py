import numpy as np

from ur_mntfields_arm.frontier_bank import FrontierBank
from ur_mntfields_arm.voxel_map import FrontierCluster, SparseVoxelMap


def _cluster(xyz, normal):
    return FrontierCluster(
        centroid=np.asarray(xyz, dtype=np.float32),
        normal=np.asarray(normal, dtype=np.float32),
        voxels=[(0, 0, 0)],
    )


def test_nearby_clusters_do_not_share_one_record_in_an_update():
    bank = FrontierBank(match_radius_m=0.5, normal_match_cos=-1.0)
    bank.update([_cluster((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))], step_idx=1)

    bank.update(
        [
            _cluster((0.02, 0.0, 0.0), (1.0, 0.0, 0.0)),
            _cluster((0.04, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ],
        step_idx=2,
    )

    assert len(bank.records) == 2
    assert sorted(record.last_seen_step for record in bank.records.values()) == [2, 2]


def test_opposite_face_normal_creates_a_distinct_frontier_record():
    bank = FrontierBank(match_radius_m=0.5, normal_match_cos=0.35)
    bank.update([_cluster((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))], step_idx=1)
    bank.update([_cluster((0.02, 0.0, 0.0), (-1.0, 0.0, 0.0))], step_idx=2)

    assert len(bank.records) == 2


def test_repeated_candidate_failure_retires_unreachable_frontier():
    bank = FrontierBank(match_radius_m=0.2, max_failures=3)
    bank.update([_cluster((0.4, 0.0, 0.5), (1.0, 0.0, 0.0))], step_idx=1)
    frontier_id = next(iter(bank.records))

    bank.mark_failed(frontier_id)
    bank.mark_failed(frontier_id)
    assert bank.records[frontier_id].status == "active"
    bank.mark_failed(frontier_id)

    assert bank.records[frontier_id].status == "retired"
    assert bank.active_records() == []


def test_frontier_is_free_unknown_boundary_without_occupied_neighbor():
    voxel_map = SparseVoxelMap(voxel_size=1.0)
    voxel_map.free = {(0, 0, 0), (1, 0, 0)}
    voxel_map.occupied = {(-3, 0, 0)}

    clusters = voxel_map.frontier_clusters()

    frontier_voxels = {key for cluster in clusters for key in cluster.voxels}
    assert frontier_voxels == voxel_map.free
