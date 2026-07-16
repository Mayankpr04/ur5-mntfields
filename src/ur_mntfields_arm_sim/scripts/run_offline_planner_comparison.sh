#!/usr/bin/env bash
# ROS Humble's generated setup files probe optional variables that may be
# unset, so enable nounset only after sourcing the environments.
set -eo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 CHECKPOINT [CLEARANCE_MM=30] [REPEATS=5]" >&2
  exit 2
fi

CHECKPOINT="$1"
CLEARANCE_MM="${2:-30}"
REPEATS="${3:-5}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_DIR="$(cd -- "$(dirname -- "${CHECKPOINT}")" && pwd)"
CLEARANCE_M="$(python3 -c 'import sys; print(float(sys.argv[1]) / 1000.0)' "${CLEARANCE_MM}")"
TAG="margin${CLEARANCE_MM}mm_${REPEATS}x"
DIRECT_JSON="${MODEL_DIR}/planner_comparison_direct_${TAG}.json"
ANCHOR_JSON="${MODEL_DIR}/planner_comparison_anchor_${TAG}.json"
DIRECT_LOG="${MODEL_DIR}/planner_comparison_direct_${TAG}.log"
ANCHOR_LOG="${MODEL_DIR}/planner_comparison_anchor_${TAG}.log"

source /opt/ros/humble/setup.bash
source "${WS_ROOT}/install/setup.bash"
set -u
export PYTHONPATH="${WS_ROOT}/src/ur_mntfields_arm${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
  local routing="$1"
  local output="$2"
  local log="$3"
  python3 -u -m ur_mntfields_arm.offline_planner_benchmark \
    --checkpoint "${CHECKPOINT}" \
    --goal-set teleop6_safe \
    --case-set sequence6 \
    --routing "${routing}" \
    --repeats "${REPEATS}" \
    --collision-margin "${CLEARANCE_M}" \
    --output "${output}" >"${log}" 2>&1
}

echo "Running direct comparison on ${CLEARANCE_MM} mm margin (${REPEATS} repeats)..." >&2
run_one direct "${DIRECT_JSON}" "${DIRECT_LOG}"
echo "Running anchor comparison on ${CLEARANCE_MM} mm margin (${REPEATS} repeats)..." >&2
run_one anchor "${ANCHOR_JSON}" "${ANCHOR_LOG}"

python3 - "${DIRECT_JSON}" "${ANCHOR_JSON}" <<'PY'
import json
import sys

for routing, filename in zip(("direct", "anchor"), sys.argv[1:]):
    with open(filename, "r", encoding="utf-8") as stream:
        report = json.load(stream)
    print(f"{routing}:")
    for planner in ("field", "field_collision", "rrt_connect"):
        rows = [row for row in report["raw"] if row["planner"] == planner]
        summary = next(row for row in report["summary"] if row["planner"] == planner)
        attempts = len(rows)
        planned = int(summary["planned"])
        collision_free = int(summary["collision_free"])
        margin_safe = int(summary["safe"])
        clearance_mm = min(float(row["min_clearance_m"]) for row in rows) * 1000.0
        success_rate = 100.0 * margin_safe / attempts if attempts else 0.0
        print(
            f"  {planner}: [plans={planned}/{attempts}, "
            f"clearance={clearance_mm:.1f} mm min, "
            f"mean_time={float(summary['mean_ms']):.1f} ms, "
            f"collision_free={collision_free}/{attempts}, "
            f"success_rate={success_rate:.1f}%]"
        )
    if routing == "anchor":
        print(f"  anchor_selection_time={float(report['anchor_selection_ms']):.1f} ms (one-time)")

print(f"reports: {sys.argv[1]} {sys.argv[2]}")
PY
