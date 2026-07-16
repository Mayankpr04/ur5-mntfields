#!/usr/bin/env bash
set -eo pipefail

mode="${1:-}"
checkpoint="${2:-/home/mayank/ur_ws/src/ur5_sim_training_factorized_v6_balanced_shell/model/weights_final.pt}"
collision_cloud="${3:-}"
anchor_route="${4:-false}"

if [[ -z "${mode}" ]]; then
  echo "usage: $0 {rrt|field|field_anchor|field_collision} [checkpoint.pt] [step_NNNNNN.npz|cloud.npy] [true|false anchor route]" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source /home/mayank/ur_ws/install/setup.bash
set -u

common=(
  checkpoint_path:="${checkpoint}"
  fixed_goal_anchor_routing_enabled:="${anchor_route}"
)
if [[ -n "${collision_cloud}" ]]; then
  common+=(collision_cloud_path:="${collision_cloud}")
  if [[ "${collision_cloud}" == *.npz ]]; then
    common+=(budgeted_anchor_sample_path:="${collision_cloud}")
  fi
fi

case "${mode}" in
  rrt)
    planner=(
      planner_type:=rrt_connect
      planner_direct_edge:=true
      field_precheck_enabled:=false
      collision_aware_field_rollout:=true
      trajectory_collision_validation_enabled:=true
      planner_shortcut:=true
      rrt_step_size_q:=0.20
      rrt_max_iters:=4000
      rrt_goal_bias:=0.20
      rrt_edge_check_step_rad:=0.04
      path_shortcut_max_passes:=4
    )
    ;;
  field)
    planner=(
      planner_type:=field
      planner_mode:=forward
      planner_direct_edge:=true
      field_precheck_enabled:=false
      collision_aware_field_rollout:=false
      trajectory_collision_validation_enabled:=false
      planner_shortcut:=true
      path_shortcut_max_passes:=4
    )
    ;;
  field_anchor)
    planner=(
      planner_type:=field_anchor
      allow_uncertified_anchor_checkpoint:=true
      planner_mode:=forward
      planner_direct_edge:=true
      field_precheck_enabled:=false
      collision_aware_field_rollout:=false
      trajectory_collision_validation_enabled:=false
      planner_shortcut:=true
      path_shortcut_max_passes:=4
      budgeted_anchor_count:=1
      budgeted_anchor_force_first_goal:=true
      fixed_goal_return_to_first:=false
      fixed_goal_joint_positions_csv:="0.407674,-0.573785,1.257008,-3.851825,-1.572289,0.006915,-0.60558,-0.76626,1.46891,-3.82272,-1.56999,0.0,-0.26256,-1.21705,0.11368,-2.11169,-1.56999,0.0,-0.75557,-1.21983,0.11368,-2.06169,-1.56999,0.0,-0.75557,-1.01123,0.75179,-2.44670,-1.56999,0.0,0.51738,-0.95577,0.75179,-2.62537,-1.56999,0.0"
    )
    ;;
  field_collision)
    planner=(
      planner_type:=field_search
      planner_mode:=forward
      planner_direct_edge:=true
      # This is the explicit partial-checkpoint diagnostic mode. Let the
      # travel-time field attempt a plan even when the conservative v3 state
      # head fails closed; exact collision-aware rollout and post-validation
      # below still prevent an unchecked trajectory from being executed.
      field_precheck_enabled:=false
      collision_aware_field_rollout:=true
      trajectory_collision_validation_enabled:=true
      planner_shortcut:=true
      path_shortcut_max_passes:=0
    )
    ;;
  *)
    echo "unknown mode '${mode}'; expected rrt, field, field_anchor, or field_collision" >&2
    exit 2
    ;;
esac

exec ros2 launch ur_mntfields_arm_sim ur_mntfields_arm_field_test_gz.launch.py \
  "${common[@]}" "${planner[@]}"
