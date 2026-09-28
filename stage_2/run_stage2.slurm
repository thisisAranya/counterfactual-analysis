#!/bin/bash
#SBATCH --job-name=stage2_rollout
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=stage_2/stage2-%j.log

# Needs stage 1 outputs for the chosen split. Submit from the project root:
#   SMOKE=1 sbatch stage_2/run_stage2.slurm     # 1 query, K_max=16
#   sbatch stage_2/run_stage2.slurm             # full run (20 queries, K_max=256)
set -euo pipefail

module purge
module load "${PYTHON_MODULE:-python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-x86_64}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
if [ ! -f config.yaml ]; then
    echo "Submit from the project root (config.yaml not found in $PWD)" >&2
    exit 1
fi
source "${VENV_DIR:-$SLURM_SUBMIT_DIR/venv-qwen}/bin/activate"

nvidia-smi
python -u stage_2/run_stage2.py --config config.yaml --split "${SPLIT:-original}" ${SMOKE:+--smoke}
