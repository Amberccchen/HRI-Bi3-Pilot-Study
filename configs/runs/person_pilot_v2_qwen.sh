#!/usr/bin/env bash
# Person pilot (preset pilot_v2, planned person) with the local Qwen model, thinking on, 3 samples per trip, both baselines.
# The vLLM Qwen server must already be running on :8000. Extra run options and overrides pass through, for example:
#   configs/runs/person_pilot_v2_qwen.sh --overrides prediction=true_future
# Run folder: /persistent/experiments/HRI/bi3_traj_gen/trips_v3/<YYYYMMDD>_person_pilot_v2_qwen_<short commit>
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
name="$(basename "${BASH_SOURCE[0]}" .sh)"
run="/persistent/experiments/HRI/bi3_traj_gen/trips_v3/$(date +%Y%m%d)_${name}_$(git rev-parse --short=7 HEAD)"
python="${PYTHON:-/persistent/conda/envs/hri-vllm-sm120/bin/python}"
export PYTHONPATH="$repo"
export RES_OWNER="${RES_OWNER:-hri-trajgen/$name}"
/persistent/tools/res_run --ram 3G --cpus 2 --label "$name" -- \
  "$python" -m traj_gen.run --run "$run" run --harness pilot_v2 --thinking think --samples 3 "$@"
