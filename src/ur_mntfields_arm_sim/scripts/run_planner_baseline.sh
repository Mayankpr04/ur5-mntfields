#!/usr/bin/env bash
set -eo pipefail

mode="${1:-}"
checkpoint="${2:-/home/mayank/ur_ws/src/ur5_sim_training_factorized_v6_balanced_shell/model/weights_final.pt}"
collision_cloud="${3:-}"
anchor_route="${4:-false}"

if [[ -z "${mode}" ]]; then
  echo "usage: $0 {rrt|field|field_collision} [checkpoint.pt] [step_NNNNNN.npz|cloud.npy] [true|false anchor route]" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source /home/mayank/ur_ws/install/setup.bash
set -u

common=(
  checkpoint_path:="${checkpoint}"
  planner_direct_edge:=false
  fixed_goal_anchor_routing_enabled:="${anchor_route}"
)
if [[ -n "${collision_cloud}" ]]; then
  common+=(collision_cloud_path:="${collision_cloud}")
fi

case "${mode}" in
  rrt)
    planner=(
      planner_type:=rrt_connect
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
      planner_mode:=bidirectional
      collision_aware_field_rollout:=false
      trajectory_collision_validation_enabled:=false
      planner_shortcut:=true
      path_shortcut_max_passes:=4
    )
    ;;
  field_collision)
    planner=(
      planner_type:=field_search
      planner_mode:=bidirectional
      collision_aware_field_rollout:=true
      trajectory_collision_validation_enabled:=true
      planner_shortcut:=true
      path_shortcut_max_passes:=0
    )
    ;;
  *)
    echo "unknown mode '${mode}'; expected rrt, field, or field_collision" >&2
    exit 2
    ;;
esac

exec ros2 launch ur_mntfields_arm_sim ur_mntfields_arm_field_test_gz.launch.py \
  "${common[@]}" "${planner[@]}"
