# Budgeted field anchors

`field_anchor` augments the trained field with one environment-derived
transition configuration. It is not a growing waypoint graph and it does not
insert a new anchor for every query.

## Selection

Selection runs once when the first test plan is requested:

1. Compute the occupied scene bounds from analytic scene boxes, or from robust
   point-cloud quantiles when boxes are unavailable.
2. Select the robot-facing vertical opening. The gravity axis is explicitly
   excluded, so the cabinet floor or ceiling cannot become a retreat face.
3. Place one camera pose at the opening's lateral centre, slightly below its
   vertical midpoint, and 0.35 m outside the opening. Point optical z into the
   scene and keep optical y aligned with gravity-down.
4. Load a small set of IK seeds from the replay artifact plus the known startup
   and test states. Solve at most eight IK branches for that single pose.
5. Reject branches with insufficient clearance, camera-position error, or
   optical-axis error. Choose the remaining branch using clearance and distance
   from the replay's typical joint region.
6. Detect thin internal scene partitions, such as shelf boards. Runtime motions
   whose endpoints lie inside the structure on opposite sides of a partition
   are routed through the shared anchor. Approach and same-region motions stay
   direct.

There is no uniform joint sampling, workspace lattice, greedy set cover, or
candidate-by-probe edge matrix. Runtime planning evaluates only the direct
field route for same-region motions and only the anchored route for detected
cross-partition motions; it does not query the collision checker.

## Running

After building and sourcing the workspace:

```bash
CHECKPOINT=/absolute/path/to/model/weights_final.pt
./src/ur_mntfields_arm_sim/scripts/run_planner_baseline.sh \
  field_anchor "$CHECKPOINT"
```

The latest `samples/step_*.npz` beside the checkpoint is discovered
automatically. An explicit replay file may be supplied as the third argument:

```bash
./src/ur_mntfields_arm_sim/scripts/run_planner_baseline.sh \
  field_anchor "$CHECKPOINT" /absolute/path/to/samples/step_000300.npz
```

Direct launch parameters include:

- `planner_type:=field_anchor`
- `budgeted_anchor_count:=1`
- `budgeted_anchor_sample_path:=...`

Do not also enable `fixed_goal_anchor_routing_enabled`; that is the older,
manually specified test sequence and is separate from this automatic method.

## Safety boundary

Anchors improve global routing only when the learned field recognizes a weak or
blocked direct transition. They cannot correct a false-positive field prediction
that assigns high speed to a colliding direct edge. `field_anchor` is intended
to expose the learned planner without runtime geometric checking, so it is not a
formal collision guarantee. Use `field_collision` or final trajectory
validation for hardware execution until the field's held-out edge calibration
meets the required safety target.
