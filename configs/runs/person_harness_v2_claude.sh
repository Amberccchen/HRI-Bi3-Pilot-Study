#!/usr/bin/env bash
# Person harness (preset harness_v2, planned person) with Claude Sonnet 5.5, thinking on at medium effort, 3 samples per trip, both baselines. Add --overrides prediction=true_future for the recorded futures, and check=2 for the plan check.
# The Anthropic API key is read from ~/.config/api_keys/anthropic for this one command only. Extra run options and overrides pass through.
# Run folder: /persistent/experiments/HRI/bi3_traj_gen/trips_v3/<YYYYMMDD>_person_harness_v2_claude_<short commit>
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
name="$(basename "${BASH_SOURCE[0]}" .sh)"
run="/persistent/experiments/HRI/bi3_traj_gen/trips_v3/$(date +%Y%m%d)_${name}_$(git rev-parse --short=7 HEAD)"
python="${PYTHON:-/persistent/conda/envs/hri-vllm-sm120/bin/python}"
export PYTHONPATH="$repo"
export RES_OWNER="${RES_OWNER:-hri-trajgen/$name}"
ANTHROPIC_API_KEY="$(cat ~/.config/api_keys/anthropic)" /persistent/tools/res_run --ram 3G --cpus 2 --label "$name" -- \
  "$python" -m traj_gen.run --run "$run" run --harness harness_v2 --backend claude --model claude-sonnet-5-5 --effort medium --thinking think \
  --samples 3 --concurrency 8 "$@"
