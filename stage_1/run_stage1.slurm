#!/bin/bash
#SBATCH --job-name=stage1_baseline
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=stage_1/stage1-%j.log

# Submit from the project root:
#   sbatch stage_1/run_stage1.slurm             # full run (20 queries)
#   SMOKE=1 sbatch stage_1/run_stage1.slurm     # 1 query, 64 tokens
#   SPLIT=counterfactual sbatch stage_1/run_stage1.slurm   # counterfactual queries
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

module purge
module load "${PYTHON_MODULE:-python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-x86_64}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f config.yaml ]; then
    echo "Submit from the project root (config.yaml not found in $PWD)" >&2
    exit 1
fi
VENV_DIR="${VENV_DIR:-$SLURM_SUBMIT_DIR/venv-qwen}"
source "$VENV_DIR/bin/activate"

# Check the cache; download only if the model is missing or incomplete.
# Best practice: download once on the login node first (see download_model.py),
# so this step only confirms the cache. Xet is disabled inside download_model.py.
HF_HUB_OFFLINE=0 python -u download_model.py --model "$MODEL"

export HF_HUB_OFFLINE=1
nvidia-smi
python -u stage_1/run_stage1.py --config config.yaml --split "${SPLIT:-original}" ${SMOKE:+--smoke}
