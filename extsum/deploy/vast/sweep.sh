#!/bin/bash
# Run several models over several experiments on one rented GPU, unattended.
#
#   EXPERIMENTS="billsum" bash deploy/vast/sweep.sh
#   EXPERIMENTS="cnndm billsum" MODELS="qwen35_9b gemma4_12b" bash deploy/vast/sweep.sh
#
# Start it under nohup or tmux so it survives the SSH session:
#   nohup bash deploy/vast/sweep.sh > logs/sweep.out 2>&1 &
#
# Resume-safe: finished variants are skipped, so re-running after a crash or a
# stopped instance continues where it left off. While one model runs, the next
# one's weights are downloaded in the background; a model's weights are deleted
# when it finishes, so peak disk stays at about two models.
#
# Env:
#   EXPERIMENTS    space-separated experiment names         (default: billsum)
#   MODELS         space-separated model aliases            (default: the 6 local models)
#   GRID           grid name, applied to every experiment   (default: each experiment's own)
#   LIMIT          only the first N documents (smoke test)
#   KEEP_WEIGHTS=1 do not purge weights between models (needs a large disk)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    . /venv/main/bin/activate
fi

EXPERIMENTS="${EXPERIMENTS:-billsum}"
MODELS="${MODELS:-qwen35_2b qwen35_4b qwen35_9b gemma4_e2b gemma4_e4b gemma4_12b}"

export EXTSUM_PROFILE="${EXTSUM_PROFILE:-vast}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The xet/CAS download backend fails on some dataset repos; use plain HTTP.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Parallel chunked downloads when the xet backend is enabled (HF_HUB_DISABLE_XET=0):
# a single connection is often throttled to a few MB/s on rented boxes.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
# Fail fast and retry on a throttled endpoint instead of hanging for minutes.
export OPENROUTER_TIMEOUT="${OPENROUTER_TIMEOUT:-90}"

EXTRA=()
[ -n "${GRID:-}" ] && EXTRA+=(--grid "$GRID")
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")

# Silver files first: cheap, CPU-only, and every model must see the same ones.
# (the listing is captured first: piping it into `grep -q` closes the pipe early,
# which under pipefail reads as a failure and would skip this step)
LISTING="$(extsum list experiments)"
for E in $EXPERIMENTS; do
    if grep -q "^$E .*task=document" <<< "$LISTING"; then
        extsum silver --experiment "$E" || { echo "silver generation failed for $E" >&2; exit 1; }
    fi
done

# Hugging Face repo of a model alias, or nothing for endpoint-only models.
hf_repo() {
    python - "$1" <<'PY'
import sys
from extsum.backends import resolve_model
m = resolve_model(sys.argv[1])
print("" if m.get("backend") == "endpoint" else m["hf_id"])
PY
}

# Download a model's weights in the background, so the GPU is not idle while
# the next model is fetched. One download at a time; PREFETCH_PID tracks it.
PREFETCH_PID=""
prefetch() {
    local repo
    repo="$(hf_repo "$1")"
    [ -z "$repo" ] && return 0
    echo ">> prefetching $1 ($repo) in the background"
    hf download "$repo" >> logs/prefetch.log 2>&1 &
    PREFETCH_PID=$!
}

mkdir -p logs
# Recent huggingface_hub versions keep large files in one shared store,
# hub/blobs/, with a .refs file naming the model folders that use each one.
# Deleting a model's folder alone frees nothing, so also delete every shared
# file whose model folders are all gone. (No-op on the older per-model layout.)
purge_shared_blobs() {
    python - "$HF_HOME/hub" <<'PY'
import glob, os, sys
hub = sys.argv[1]
freed = 0
for refs in glob.glob(os.path.join(hub, "blobs", "*", "*.refs")):
    repos = {line.split("/")[0] for line in open(refs).read().split()}
    if repos and not any(os.path.isdir(os.path.join(hub, r)) for r in repos):
        blob = refs[: -len(".refs")]
        for path in (blob, refs, blob + ".lock"):
            if os.path.exists(path):
                freed += os.path.getsize(path)
                os.remove(path)
if freed:
    print(f">> freed {freed / 1e9:.1f} GB of weights")
PY
}

MODEL_LIST=($MODELS)
FAILED=()
for i in "${!MODEL_LIST[@]}"; do
    M="${MODEL_LIST[$i]}"
    NEXT="${MODEL_LIST[$((i + 1))]:-}"

    # The weights of M were prefetched during the previous model (or are fetched
    # by the first run). Let that download finish, then start on the next one.
    if [ -n "$PREFETCH_PID" ]; then
        wait "$PREFETCH_PID" || echo "!! prefetch of $M failed; the run will download it itself"
        PREFETCH_PID=""
    fi
    [ -n "$NEXT" ] && prefetch "$NEXT"

    for E in $EXPERIMENTS; do
        mkdir -p "logs/$E"
        LOG="logs/$E/$M.log"
        echo "=================== $M / $E   (log: $LOG)   $(date +%H:%M:%S) ==================="
        extsum run --experiment "$E" --model "$M" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOG"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "!! $M / $E failed — see $LOG; continuing with the next one"
            FAILED+=("$M/$E")
        fi
    done

    # Free this model's weights only: the next model's are being downloaded.
    if [ "${KEEP_WEIGHTS:-0}" != "1" ]; then
        REPO="$(hf_repo "$M")"
        if [ -n "$REPO" ]; then
            rm -rf "$HF_HOME/hub/models--${REPO//\//--}" 2>/dev/null || true
            purge_shared_blobs
        fi
    fi
done

# shellcheck disable=SC2086
extsum report --experiment $EXPERIMENTS || true

echo "=== sweep complete ==="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
echo "Fetch the results from your laptop:  bash deploy/fetch_results.sh <ssh-port> <user@host>"
