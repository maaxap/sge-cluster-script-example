#!/bin/bash
#$ -N vllm_bench
#$ -o $JOB_NAME.$JOB_ID.out
#$ -e $JOB_NAME.$JOB_ID.err
#$ -l matylda4=10
#$ -q all.q@@gpu
#$ -pe smp 4
#$ -l gpu=0.5,gpu_ram=48G,ram_free=4G,mem_free=4G,tmp_free=125M
#$ -R y

ulimit -t 14400

PROJECT_DIR=/mnt/matylda4/iaparovich/Workspace/sge-cluster-script-example
cd "$PROJECT_DIR"

source /mnt/matylda4/iaparovich/miniconda3/etc/profile.d/conda.sh
conda activate vllm

set -euo pipefail

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

NCCL_IB_DISABLE=1 python scripts/benchmark_vllm.py

# TODO:
# - Submit: qsub run_benchmark_vllm.sh
# - Watch: qstat -u $USER
# - Logs: tail -f vllm_bench.<JOB_ID>.out   /   .err
# - Results JSON:  benchmark_vllm_<timestamp>.json
