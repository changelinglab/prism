#!/bin/bash
#SBATCH -A bbjs-dtai-gh
#SBATCH -p ghx4-interactive
#SBATCH -J main
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 72
#SBATCH --mem=120G
#SBATCH -t 30:00
#SBATCH -o exp/slurm_logs/%A_%a.out
#SBATCH -e exp/slurm_logs/%A_%a.out

##########################
echo "RUNNING ON NODE: $(hostname)"
echo "START TIME: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "RunningCMD: python src/main.py $*"
# === Directory Setup ===
PROJECT_HOME=/work/nvme/bbjs/sbharadwaj/powsm/PRiSM
cd $PROJECT_HOME
mkdir -p "exp/slurm_logs"
# === Environment setup ===
source ~/.bashrc
conda deactivate
source setup_uv.sh .venv_dai requirements-dai.txt
# for w2v2ph these must be pre-built
export HF_HOME="${PROJECT_HOME}/exp/cache/hf"
export PHONEMIZER_ESPEAK_LIBRARY="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/espeak-ng-data"
###########################
###########################
get_free_port_atomic() {
    local port
    local lock_base_dir="/dev/shm/vllm_port_locks"
    mkdir -p "$lock_base_dir"
    while true; do
        port=$(shuf -i 20000-60000 -n 1)
        if ! ss -lnt | grep -q ":$port "; then
            if mkdir "$lock_base_dir/$port" 2>/dev/null; then
                echo $port
                return 0
            fi
        fi
    done
}
###########################
###########################
nvidia-smi
FREE_PORT=$(get_free_port_atomic)
echo "Using Port: $FREE_PORT"
VLLM_EXECUTABLE=exp/download/vllm_arm.sif
# MODEL=Qwen/Qwen3-Omni-30B-A3B-Thinking
# TOTAL_LENGTH=8192
# MODEL=nvidia/audio-flamingo-3-hf
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
TOTAL_LENGTH=8192
###########################
###########################
apptainer exec --cleanenv --nv \
  --env HF_HOME="$HF_HOME" \
  -B $PROJECT_HOME \
  "${VLLM_EXECUTABLE}" \
  vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port $FREE_PORT \
    --trust-remote-code \
    --max-model-len ${TOTAL_LENGTH} \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enforce-eager 