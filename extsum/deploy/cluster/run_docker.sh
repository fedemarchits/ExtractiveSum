#!/bin/bash
# Run an `extsum` command inside the GPU container, with the project mounted at
# /workspace so configs, data and results are the ones in this checkout.
#
#   bash deploy/cluster/run_docker.sh run -e aclsum -m qwen35_4b
#
# Credentials come from the project's .env (read by extsum itself).
# Env: IMAGE_NAME (default extsum-gpu:latest), GPUS (override the SLURM allocation).
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-extsum-gpu:latest}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# `docker --gpus all` bypasses SLURM's cgroup, so forward exactly the GPUs SLURM
# allocated. Inside the container they are renumbered 0..N-1.
HOST_GPUS="${GPUS:-${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-${CUDA_VISIBLE_DEVICES:-0}}}}"
NGPU="$(awk -F',' '{print NF}' <<< "$HOST_GPUS")"
CONTAINER_GPUS="$(seq 0 $((NGPU - 1)) | paste -sd, -)"

echo "image: $IMAGE_NAME | project: $PROJECT_ROOT | gpus: host [$HOST_GPUS] -> container [$CONTAINER_GPUS]"
echo "command: extsum $*"

docker run --rm \
    --gpus "\"device=${HOST_GPUS}\"" \
    --shm-size=16g \
    -v "$PROJECT_ROOT":/workspace \
    -e CUDA_VISIBLE_DEVICES="$CONTAINER_GPUS" \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMAGE_NAME" \
    python -u -m extsum "$@"
