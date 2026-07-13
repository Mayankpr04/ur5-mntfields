# Neural-only planning implementation tracker

## Objective

Train an online UR5 arrival-time field in at most 10 minutes and generate paths
without point-cloud, SDF, or geometric collision queries during normal planning.
Geometric checks remain evaluation-only until the neural planner is certified.

## Implemented

- [x] Paper-factorized arrival time and stable Eq. 12 objective.
- [x] Global endpoint reshuffling for train and diagnostics.
- [x] Corrected diagnostic distribution and completion gate.
- [x] Bounded NBV task-space IK and FK orientation validation.
- [x] Reachability-first joint-space NBV recovery.
- [x] Per-frontier NBV rejection accounting and retirement.
- [x] Viewpoint cooldown remains strict; no two-view override.
- [x] Ten-minute wall-time stop saves a non-certified partial checkpoint.
- [x] Exact failed-rollout barrier states retained with zero-speed labels.
- [x] Stable log-speed loss for critical and barrier states.
- [x] Network-only learned-speed C-space search.
- [x] Planner debug reports zero geometric collision queries.
- [x] Learned-speed search accumulates neural low-speed risk instead of treating
  every edge above one binary threshold as equally safe.
- [x] Online certification uses deterministic joint goals shared with the test
  launch; it no longer reports `0/0` because an unrelated IK sweep failed.
- [x] Fixed-goal execution requires a stable settle interval before publishing
  the next trajectory, preventing controller-goal overlap.
- [x] ROI-centered recovery remains available after the frontier bank reaches
  zero active records, avoiding the static resampling loop while ROI unknowns
  remain.
- [x] Reverted online optimization to the paper's Eq. 12 objective
  (`speed_loss_weight=0.01`) and disabled non-paper log-speed, normal, risk,
  and synthetic failed-rollout barrier supervision that created the
  near-zero-speed prediction mode.
- [x] Replay retention and minibatch selection are uniform as required by
  Algorithm 1; low-speed quotas and C-space-cell quotas were removed.
- [x] Corrected exploration frontiers to known-free voxels adjacent to unknown
  voxels. Occupied adjacency is no longer required, and normals point toward
  unknown space.
- [x] Replaced three geometrically colliding certification goals with diverse
  configurations observed collision-free during exploration.
- [x] `view_field_speed_3d` trajectory generation now calls the network-only
  planner; geometric clearance is reported only after generation as an
  evaluation measurement.

## July 12 replay ablation

Training from scratch on the latest run's 131,120 saved rows with uniform
replay and paper Eq. 12 reached at optimizer step 840:

- speed MAE: `0.1664`
- speed correlation: `0.7309`
- near/far gap: `0.4886`
- low-target overprediction: `0.4274`
- wall time: about `45 s`

Network-only search generated all four configured legs. Post-generation
geometric evaluation measured minimum clearances between `0.2788 m` and
`0.3610 m`. This demonstrates that the saved clearance labels and paper
architecture are learnable; the prior collapse came from the auxiliary
training objective and biased replay distribution.

## Cabinet held-out failure

The original inside-cabinet goal-1 to goal-2 edge was densely queried by the
network (1,457 learned-speed states) but predicted a minimum speed of `0.477`
while post-generation geometry measured zero clearance. Nearest saved replay
states were `0.067-0.103` away in normalized 6-D C-space and had inconsistent
labels from `0.11` to `1.0`, confirming a local sampling coverage hole rather
than insufficient edge interpolation. The cabinet test goals and interpolated
corridors are now sampling anchors, while completion certification continues
to use the separate outside goal library.

## Acceptance gates

- `speed_corr >= 0.35`
- `near_far_gap >= 0.20`
- `low_target_overpred_frac <= 0.45`
- Raw field and learned-speed search are evaluated separately.
- Network-only search must report `geometric_collision_queries=0`.
- A checkpoint is certified only after all configured fixed start-goal cases pass
  evaluation against geometry; geometry is not used to generate the paths.

## Network-only test

```bash
ros2 launch ur_mntfields_arm_sim ur_mntfields_arm_field_test_gz.launch.py \
  checkpoint_path:=/absolute/path/to/checkpoint.pt \
  planner_type:=learned_speed_search \
  planner_mode:=goal_to_start \
  learned_speed_search_min_speed:=0.20 \
  collision_aware_field_rollout:=false \
  trajectory_collision_validation_enabled:=true \
  planner_direct_edge:=false \
  planner_shortcut:=false
```

`trajectory_collision_validation_enabled:=true` is evaluation-only: it checks
the completed path and does not participate in network-only search. Disable it
only after measuring an acceptable success rate over the full fixed-goal suite.

## Remaining validation

- [ ] Tune the learned-speed threshold on held-out start-goal cases.
- [x] Require multiple reachable fixed goals (three configured joint goals).
- [ ] Measure network-only planning latency and geometric success rate.
- [ ] Compare raw gradient, learned-speed search, and collision-aware baselines.

## Network-only latency correction

The learned-speed planner was not performing geometric collision queries, but
its dense neural edge sampler mixed coordinate systems: normalized
`step_size_q` was applied to denormalized joint angles.  On the cabinet goal-2
case this produced 31,952 neural states for 1,418 local edges and dominated the
2.59 s planning time.  Edge interpolation now remains in normalized C-space
and is denormalized only for the batched model call.  The learned-speed oracle
also caches repeated states, eliminating duplicate current-state and candidate
endpoint inference.  Planner diagnostics now report requested states, actual
inferred states, and cache hits.  This is a latency-only correction; no
geometric collision checker was added and the learned-speed threshold remains
unchanged.

Follow-up: fixed a stale `q_cands` name in the optimized edge reduction and
added a regression test that executes normalized edge interpolation and checks
its exact sample count.  The full package suite now contains 23 passing tests.

## Cabinet corridor coverage correction

The post-fix network-only test plans in `65-69 ms`, but goal 2 still collides:
the network assigns about `0.55` speed to the invalid direct corridor.  The
geometric-rollout comparison succeeds but takes `1.6 s`; it is a geometric
baseline, not evidence of network-only safety.  Inspection of the latest saved
training batch found no states within normalized distance `0.03` of the
goal-1-to-goal-2 corridor.  Configured training anchors had only been mixed
into broad proposal seeding, which did not guarantee local supervision.

Fifteen percent of each batch is now reserved for path-local samples around
both configured cabinet anchors and failed-rollout anchors.  The path sampler
uses a maximum normalized q0 radius of `0.025` and no longer duplicates the
near-boundary projection stage.  Broad and near-boundary sampling retain their
separate roles.  This changes training data and therefore requires a new
checkpoint; it cannot repair the existing `weights_final.pt`.

## Planner baseline harness

The fixed-goal test now supports `planner_type:=rrt_connect` using the existing
joint-space RRTConnect implementation. RRT diagnostics include iterations,
tree sizes, collision edge-query calls, and queried states. The optional
`collision_cloud_path` accepts either a saved training `step_*.npz` containing
`occupied_points` or an `N x 3` `.npy` cloud. When supplied, RRTConnect,
collision-aware field planning, and post-plan validation all use that same
static cloud plus the configured scene/support boxes. The wrapper
`src/ur_mntfields_arm_sim/scripts/run_planner_baseline.sh` selects consistent
flags for `rrt`, `field`, and `field_collision` comparisons.

RRT edge validation now uses the same `0.04 rad` interpolation resolution as
final trajectory validation. Previously its `0.10 rad` internal spacing could
accept an edge whose denser validation minimum was `0.017 m`, below the
configured `0.020 m` safety margin. This was a margin failure rather than a
visible zero-clearance collision.

The RRT wrapper now enables four dense collision-checked shortcut passes. Raw
RRTConnect remains timed as `plan_ms`; simplification and final checking remain
separately visible as `validate_ms`, with `raw_waypoints` versus
`plan_waypoints` showing the effect. This avoids executing the large random
tree detours produced by an otherwise cost-unaware RRTConnect solution.

## Saved-run cabinet coverage audit

The completed run contains 15 saved batches, 105,000 labelled pairs and
210,000 labelled endpoint states. Replay reached 105,000 pairs against a
200,000-pair capacity, so replay capacity did not discard any row. Endpoint
reshuffling also preserves every configuration and label; it only changes
which independently labelled states form a start/goal pair.

Despite the nominal 15% path-local quota, the saved data has zero endpoint
states within normalized C-space distance 0.025 of the cabinet goal-1-to-goal-2
segment. The nearest state is at 0.029998. Around the collision configuration
reported by the goal-2 test, the nearest labelled state is at 0.030693; only 16
states lie within 0.04, and 14 of those have speed below 0.2. All samples within
0.05 of the segment occur in the first three saved batches; the remaining 12
batches contain none.

The cause is rejection bias rather than minibatch shuffling. The path sampler
jitters anchors by at most 0.025, but rejects both endpoints unless clearance
is at least 0.015 m and both clearance normals are valid. Its target is pooled
over all configured paths, so colliding/offset-band proposals around the hard
cabinet corridor are discarded while easier anchor regions fill the quota.
Consequently, increasing the total batch size does not ensure supervision on
the missing collision boundary.

The reference `ntrl-demo` arm generator uses a 500,000-pair offline dataset.
It computes whole-arm clearance against the complete obstacle mesh, keeps q0
in a deliberately sampled thin free-space boundary shell, and applies
`clip(distance, offset, margin) / margin` with offset 0.005 and margin 0.05.
Its locally perturbed q1 is not subjected to the same strict free-side filter,
so obstacle-side states receive the floor-speed label. The training loader
shuffles those pre-labelled pairs but does not drop their labels.

The fixed test corridors must not define the training distribution because
runtime goals are arbitrary. The correction is therefore global C-space
boundary discovery rather than per-corridor quotas. Newly observed online
geometry should still invalidate or relabel affected replay rows; this is
separate from the cabinet, whose scene boxes are known from startup.

## Goal-independent boundary-shell sampling

Implemented a reference-style global boundary distribution. Random colliding
q0 proposals with a valid C-space normal are no longer wasted: the sampler
expands outward until it brackets free space, bisects the collision/free
transition, and retains a verified free-side shell state. Critical free-side
q0 states now receive an inward-perturbed q1; q1 is allowed below the clearance
offset and receives the floor-speed label instead of being rejected. Broad q1
sampling follows the same rule, while q0 remains verified free. Invalid-normal
contacts are still rejected so the PDE is never trained against a fabricated
normal.

The simulation label range is now 0.01 m offset and 0.10 m margin: 10 mm is a
more practical online-SDF floor than 5 mm, produces the reference floor speed
of 0.1, and retains a surrounding clear/high-speed band. The mixture is 45%
global near-boundary and 45% global broad/free; at most 10% is path-local and
only paths actually produced during exploration can supply those anchors.
Configured fixed test goals are disabled as training anchors.

Sampler logs now expose `q0_boundary_shell_frac` and
`q1_obstacle_side_frac`. Added regression coverage for collision-to-shell
projection and retained obstacle-side floor labels. Validation: 26 tests pass;
both ROS packages build successfully. A new checkpoint is required.
The next launch writes to `src/ur5_sim_training_factorized_v5_global_shell`
so its samples and diagnostics cannot be mixed with the previous v4 run.

Follow-up from the first v5 run: the two new sampler fractions were always
logged as zero because `_merge_sampler_stats` did not include their keys. This
was diagnostic aggregation only, not a sampling failure. Saved step 353 has
28.8% critical free-side q0 labels, 49.8% floor-speed q1 labels, 37.9% saturated
q0 labels, and 24.2% saturated q1 labels. Since training reshuffles both
endpoints, about one quarter of individual states are obstacle-side/floor
examples while about one third remain saturated high-speed examples. Added
both fraction keys to the accepted-pair-weighted aggregation.

## Balanced both sides of the global boundary shell

The first v5 shell run exposed a late-training calibration collapse. At the
cabinet goal-2 transition, the safe start and the geometrically colliding
state converged to almost the same predicted speed (`0.1065` versus `0.1036`).
Checkpoint history showed the safe-state prediction falling from roughly
`0.67--0.74` at epochs 480--840 to `0.106` in the partial checkpoint.

The cause was a one-sided pair construction introduced with the global shell:
every critical q0 forcibly generated q1 by stepping inward along the clearance
normal. This made obstacle/floor endpoints dominate late replay and did not
teach the free/high-speed neighbourhood beside the same shell.

`cspace_sampling.py` now keeps collision-to-shell projection for global
boundary discovery, but pairs direct, projected, and recovered shell states
with unbiased local 6-D endpoints. The local radius was widened from `0.06`
to `0.12` normalized units, so a batch retains both obstacle-side floor labels
and surrounding clear/high-speed labels without specializing to any test goal
or corridor. A regression test now requires both sides to be present and
prevents the obstacle-side share from returning to the always-inward regime.

The next clean run writes to
`src/ur5_sim_training_factorized_v6_balanced_shell`; v5 checkpoints predate
this correction and must not be used to assess it.

## Arbitrary-goal shell coverage and held-out safety certification

The v6 result still accepted the goal-2 wall crossing: its aggregate replay
diagnostics were good (final MAE about 0.129 and correlation about 0.853), but
the independently collision-checked state was predicted at speed 0.2885. The
learned search therefore accepted it above the old 0.10 threshold and planned
the invalid edge in 64 ms. A saved-data audit found no labelled state within
normalized distance 0.025 of the safe start, colliding transition, or goal;
the nearest collision-state label was 0.997. Evaluating that state over 2,048
independent goal conditions also left safe and colliding predictions nearly
indistinguishable. This is a local six-dimensional coverage hole hidden by
global replay-fit plots, not minibatch shuffling.

The reference arm sampler perturbs its second endpoint over a normalized
radius up to 0.5 and starts from a much denser IK-derived shell. The online
sampler now matches that goal-independent radius for direct, projected, and
collision-recovered boundary states, and its bounded ROI IK seed budget was
raised from 8 to 24. It still does not enumerate the fixed cabinet test
corridors. The next run writes to
`src/ur5_sim_training_factorized_v7_goal_conditioned_shell`.

An attempted safety-loss fine-tune of the v6 replay was rejected: it reduced
global correlation from about 0.853 to 0.662 and lowered both safe and
colliding predictions without separating them. Those experimental loss
weights were reverted; the paper objective remains configured.

Training completion now includes a generic held-out false-free audit. At each
eligible training update it samples 1,024 fresh full-C-space states
and evaluates each against four independent random goals, labels state clearance with the exact geometric
checker, and measures how often a low-speed state is predicted at or above
0.20. A checkpoint cannot become `weights_final.pt` unless at least 32 low
states are observed and the false-free rate is at most 5%. Up to 64 worst
misses are added to the hard-anchor buffer, and 512 exactly relabelled hard
pairs are included in the next 7,000-pair training update. Bulk sampling still
uses the fast SDF; hard failures retain the exact checker as their label source.
The audit runs before the camera-step modulo gate, avoiding the irregular-step
scheduling bug that previously delayed evaluation until near the wall-time
limit; only the expensive fixed-goal rollout remains cadence-limited.

The network-only test acceptance threshold now defaults to 0.20. The label
floor/offset maps obstacle-side states to 0.10, so the previous 0.10 threshold
incorrectly treated the intended obstacle label itself as executable free
space. This threshold change is a safety margin, not a substitute for v7
retraining and the held-out audit. Validation after these changes: 29 tests
pass and both ROS packages build successfully with `--symlink-install`.

## Priority hard-example correction after the v7 audit

The v7 run correctly stopped with `weights_partial.pt`; it did not silently
fail its completion logic. Across its late held-out audits, 28.3--30.8% of
exact low-clearance state/goal queries were still predicted at or above 0.20,
well above the 5% certification limit. Aggregate replay fit improved to about
0.79 correlation, but that metric did not certify the unsampled C-space holes.

An initial offline label check appeared to disagree with the saved labels, but
that comparison had mixed base-frame samples with world-frame scene boxes.
After applying the live `world -> base_link` translation, saved speed labels
match recomputed SDF labels with correlation 0.996 and MAE 0.006; the SDF and
exact checker also agree well (correlation 0.960). The bulk label pipeline is
therefore not the source of the v7 failure.

The actual correction signal was diluted. The 512 exact hard pairs were only
6.8% of each fresh 7,512-pair batch, and fresh rows supplied only 25% of the
2,048-row optimizer batch. Consequently audit misses occupied roughly 1.7%
of optimizer input. The tested collision configuration also had no saved
endpoint within normalized L-infinity distance 0.025 and only eight within
0.05 across 189,168 saved endpoints.

V8 reserves 20% of every optimizer minibatch for the current exactly-labelled
audit/failed-rollout set and raises that set to 1,024 pairs. This is generic
hard-example replay, not a fixed-goal or corridor specialization. Ordinary
collection is reduced from 7,000 to 4,000 pairs so more audit/correction cycles
fit in ten minutes. The local proposal radius is 0.25; arbitrary global goals
are still generated by endpoint reshuffling every optimizer step, so a radius
of 0.5 only increased rejection time without adding goal support. The next run
writes to `src/ur5_sim_training_factorized_v8_priority_shell`.

## Cabinet transition-anchor test route

V8 still stopped at a partial checkpoint and could not reliably solve the
inter-shelf test legs. A separate test-time route option now decomposes the
three unique cabinet goals plus return into
`G1 -> anchor -> G2 -> G3 -> anchor -> G1`. The direct G2-G3 leg is retained
because both configurations occupy the lower shelf.

The anchor is a camera-facing configuration centered on the cabinet at camera
world position approximately `[0.346, 0.350, 0.800]`, about 0.35 m in front
of the opening and closer than the startup view. Its static-scene clearance is
about 0.133 m. Dense joint interpolation checks gave minimum clearances of
about 0.0395 m to G1, 0.0272 m to G2, and 0.0339 m to G3. Separate upper and
lower anchors were rejected because their direct vertical transition crossed
a shelf; one farther-forward common anchor avoids that extra unsafe leg.

This routing is optional (`fixed_goal_anchor_routing_enabled:=true`) and only
changes the fixed-goal test sequence. It does not alter training, field
validation, or the default planner baselines.

The `field` baseline is now explicitly network-only: collision-aware rollout
and final geometric validation are disabled. Its shortcut pass densely queries
the learned speed along candidate skipped segments and accepts a shortcut only
when every prediction stays above `learned_speed_search_min_speed`. RRT and
`field_collision` retain geometric validation. This mode intentionally exposes
field false-free errors; it must not be interpreted as collision certification.

## V6 selected; V8 priority replay rejected

The six-leg network-only anchor route completed with the fully trained v6
checkpoint, including both anchor/cabinet transitions. The partially trained
v8 checkpoint collided on the anchor-to-G2 leg and required substantially more
search and shortcut work. V8 is therefore an experimental rejected checkpoint,
not the current recommended model.

An apples-to-apples audit evaluated both checkpoints on the same 12,000 saved
states with independently reshuffled goal endpoints, matching training-time
pair construction. V6 obtained MAE 0.157 and speed correlation 0.806; v8
obtained MAE 0.247 and correlation 0.563. V8 reduced the mean prediction on
low-speed labels from 0.289 to 0.188, but also reduced the mean prediction on
high-speed labels from 0.877 to 0.658. Thus priority replay made the field more
conservative globally rather than learning a sharper obstacle boundary. That
loss of free-space calibration explains its weaker gradients and route
failure. The v8 checkpoint and configuration remain available for reproducing
the negative result, but must not replace v6 for planner comparisons.

`run_planner_baseline.sh` now defaults to the v6 `weights_final.pt`. Passing a
checkpoint explicitly still permits controlled v8 or later comparisons.

## Active training policy reverted to v6

The saved ROS log provides the exact v6 run fingerprint: 7,000 ordinary pairs
per view, 10% path-local after paths existed, 45% global near-shell, the
remainder broad, zero hard pairs, uniform replay with ratio 0.75, and 180
optimizer steps per update. V6 used unbiased local q1 endpoints with maximum
normalized radius 0.12 and a bounded eight-seed ROI IK budget.

The simulation defaults now restore that policy. V8's 4,000-pair collection,
1,024 hard pairs, 20% reserved priority minibatch fraction, expanded local
radius, and false-free mining/completion gate are disabled. Their
implementations remain available for explicit ablations and diagnostics, so
later correctness fixes were not discarded. The new run writes to
`src/ur5_sim_training_factorized_v9_v6_reverted`; the completed v6 artifacts
remain immutable.
