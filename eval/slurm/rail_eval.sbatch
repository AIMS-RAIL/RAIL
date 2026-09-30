#!/bin/bash
# Run inference + scoring + report for one Hugging Face model on one GPU.
# Pass your cluster's account/partition on the command line, e.g.
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,MODEL=Qwen/Qwen2-Audio-7B-Instruct,OUT=$PWD/runs/qwen2_audio,\
#   PYTHON=/path/to/venv/bin/python,EXTRA_ARGS="--limit-per-task 5" slurm/rail_eval.sbatch
#
# MODEL       Hub id or local checkpoint directory (required)
# OUT         run directory (required)
# PYTHON      interpreter with torch + transformers + librosa (default: python)
# EXTRA_ARGS  passed through to `run_rail_eval.py all` (e.g. --adapter generic --trust-remote-code)
# MODULES     optional `module load` list, e.g. "GCCcore/11.3.0 Python/3.10.4 FFmpeg/5.0.1"
#SBATCH --job-name=rail_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=rail_eval_%j.out

set -euo pipefail
: "${MODEL:?set MODEL to a Hub id or checkpoint directory}"
: "${OUT:?set OUT to the run directory}"
PYTHON="${PYTHON:-python}"

for m in ${MODULES:-}; do
  module load "$m" || echo "warning: module load $m failed" >&2
done

EVAL_DIR="${EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/run_rail_eval.py" ]]; then
  EVAL_DIR="$SLURM_SUBMIT_DIR"
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} model=$MODEL out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$OUT"
# shellcheck disable=SC2086
"$PYTHON" "$EVAL_DIR/run_rail_eval.py" all --model "$MODEL" --out "$OUT" ${EXTRA_ARGS:-}
