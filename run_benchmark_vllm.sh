#!/bin/bash
#$ -N vllm_bench
#$ -o $JOB_NAME.$JOB_ID.out
#$ -e $JOB_NAME.$JOB_ID.err
#$ -q all.q@@gpu
#$ -pe smp 4
#$ -l gpu=2,gpu_ram=48G,ram_free=8G,mem_free=8G,tmp_free=10G
#$ -R y

set -euo pipefail

ulimit -t 10800

PROJECT_DIR=/mnt/minerva1/nlp-2/homes/xaparo00/workspace/projects/sge-cluster-script-example
cd "$PROJECT_DIR"

# TODO: setup conda env on the server once I get access to the GPU harness
source /home/xaparo00/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

NCCL_IB_DISABLE=1 python scripts/benchmark_vllm.py

# TODO:
# - Submit: qsub run_benchmark_vllm.sh
# - Watch: qstat -u $USER
# - Logs: tail -f vllm_bench.<JOB_ID>.out   /   .err
# - Results JSON:  benchmark_vllm_<timestamp>.json
