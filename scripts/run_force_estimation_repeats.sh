#!/usr/bin/env bash
# Collect N scripted force-estimation repeats into force_estimation_runs/.
# Uses --headless so batch runs do not open (and crash) MuJoCo viewer windows.
# Usage:
#   ./scripts/run_force_estimation_repeats.sh [N] [scenario ...]
# Examples:
#   ./scripts/run_force_estimation_repeats.sh 5
#   ./scripts/run_force_estimation_repeats.sh 3 hit_floor push_block
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N="${1:-5}"
shift || true
if [[ "$#" -gt 0 ]]; then
  SCENARIOS=("$@")
else
  SCENARIOS=(hit_floor push_block peg_in_hole)
fi

# Headless needs no viewer; prefer plain python over mjpython.
PYTHON="${FORCE_EST_PYTHON:-${PYTHON:-python3}}"
HEADLESS_DURATION="${HEADLESS_DURATION:-10}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "N must be a positive integer (got: $N)" >&2
  exit 1
fi

echo "Writing $N repeats each for: ${SCENARIOS[*]}"
echo "Output root: $ROOT/force_estimation_runs"
echo "Launcher: $PYTHON (headless, ${HEADLESS_DURATION}s sim)"

for scenario in "${SCENARIOS[@]}"; do
  for i in $(seq 1 "$N"); do
    run_id=$(printf "run_%02d" "$i")
    out="force_estimation_runs/${scenario}/${run_id}"
    mkdir -p "$out"
    echo
    echo "=== ${scenario} / ${run_id} ==="
    case "$scenario" in
      hit_floor|push_block|peg_in_hole)
        "$PYTHON" main.py \
          --scenario "$scenario" \
          --results-dir "$out" \
          --headless \
          --headless-duration "$HEADLESS_DURATION"
        ;;
      *)
        echo "Unknown scenario: $scenario" >&2
        exit 1
        ;;
    esac
  done
done

echo
echo "Done. Build the report with:"
echo "  python3 analysis.py --force-estimation-report"
