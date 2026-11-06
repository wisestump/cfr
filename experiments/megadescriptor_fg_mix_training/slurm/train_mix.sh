#!/usr/bin/env bash
# Train one foreground-mix model on one GPU. Account and partition come from
# the sbatch command line. A run takes several days on one GPU, so resubmit the
# same command after a time limit; training resumes from checkpoint.pth.

#SBATCH --job-name=megad_fg_mix
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the folder with images/size-518 and metadata/combined.}"
FOREGROUND_PROB="${FOREGROUND_PROB:?Set FOREGROUND_PROB, e.g. 0, 0.25, 0.5 or 0.75.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the output folder of this run.}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-megad-mix}"
fi

cd "${SCRIPT_DIR}"
python train_mix.py \
  --data-root "${DATA_ROOT}" \
  --foreground-prob "${FOREGROUND_PROB}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"
