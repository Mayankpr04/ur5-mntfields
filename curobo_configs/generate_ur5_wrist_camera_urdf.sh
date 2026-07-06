#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/mayank/ur_ws"
OUT="${ROOT}/curobo_configs/ur5_wrist_camera.urdf"
XACRO_FILE="${ROOT}/src/ur_mntfields_arm_sim/urdf/ur_with_wrist_camera.urdf.xacro"
INITIAL_POSITIONS="${ROOT}/src/ur_mntfields_arm_sim/config/initial_positions.yaml"
CONTROLLERS="${ROOT}/src/ur_mntfields_arm_sim/config/gz_controllers.yaml"

source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash" 2>/dev/null || true
set -u

xacro "${XACRO_FILE}" \
  ur_type:=ur5 \
  base_z:=0.50 \
  sim_ignition:=true \
  use_fake_hardware:=false \
  fake_sensor_commands:=false \
  simulation_controllers:="${CONTROLLERS}" \
  initial_positions_file:="${INITIAL_POSITIONS}" \
  > "${OUT}"

echo "Wrote ${OUT}"
