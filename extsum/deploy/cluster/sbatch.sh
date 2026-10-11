#!/bin/bash
# Submit one model's grid on one experiment to SLURM (one model per job; each
# job is resume-safe, so submit several for the whole grid).
#
#   EXPERIMENT=aclsum MODEL=gemma4_e2b sbatch deploy/cluster/sbatch.sh
#   EXPERIMENT=aclsum MODEL=qwen35_27b GRID=aspect_large sbatch deploy/cluster/sbatch.sh
#
# This cluster uses --gpus-per-node (not --gres=gpu:) with -N 1, so several GPUs
# land on ONE node. The checkout and the docker image must exist on that node
# (no shared filesystem): pin with -w <node> if unsure.

#SBATCH --job-name=extsum
#SBATCH -N 1
#SBATCH --gpus-per-node=nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail
: "${EXPERIMENT:?set EXPERIMENT=<name> (see: extsum list)}"
: "${MODEL:?set MODEL=<alias> (see: extsum list models)}"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST | $MODEL / $EXPERIMENT | $(date)"
bash deploy/cluster/run_docker.sh run -e "$EXPERIMENT" -m "$MODEL" ${GRID:+-g "$GRID"} ${EXTRA_ARGS:-}
echo "end $(date)"
