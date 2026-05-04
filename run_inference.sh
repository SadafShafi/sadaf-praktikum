#!/bin/bash
#SBATCH --job-name=llama-inference
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=intern_agc_emmy
#SBATCH --output=slurm_logs/inference_%j.out
#SBATCH --error=slurm_logs/inference_%j.err

#SBATCH --export=NONE

# We must NOT inherit the login node's environment!
#SBATCH --export=NONE

set -eo pipefail

# --- Setup ---
APP_DIR="/user/sshafi/u20715/sadaf_praktikum"
cd "${APP_DIR}"
mkdir -p slurm_logs

# 1. Initialize environment for the compute node
source /etc/profile || true
module purge
module use /sw/modules/gwdg/rev/25.04/rome_mofed_cuda80_rocky8
module load gcc/13.2.0 python/3.11.9

# 2. Point to the existing pip packages in .venv_new so we don't need internet!
SITE_PKGS="${APP_DIR}/.venv_new/lib/python3.11/site-packages"
export PYTHONPATH="${SITE_PKGS}"

# 3. Tell HuggingFace to never try to reach the internet
export HF_HUB_OFFLINE=1

# 4. Make CUDA visible to PyTorch:
#    libcuda.so.1 (the GPU driver) lives at /usr/lib64 on compute nodes.
#    The CUDA toolkit libs are bundled in pip's nvidia/ packages.
NVIDIA_LIBS=""
for d in "${SITE_PKGS}"/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$d"
done
export LD_LIBRARY_PATH="/usr/lib64:/usr/local/cuda/lib64:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Node:          ${SLURM_NODELIST:-N/A}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:    $(date)"
echo "Working dir:   ${APP_DIR}"
echo "nvidia-smi   $(nvidia-smi)"
echo "============================================"

# --- Run inference ---
if [[ "$1" == *.py ]]; then
    SCRIPT="$1"
    shift
else
    SCRIPT="inference.py"
fi

python3 -u "$SCRIPT" "$@"

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "============================================"
