#!/bin/bash
# Copy this project to a rented box over SSH, without going through git.
# Run it ON YOUR LAPTOP from the project root:
#
#   bash deploy/sync_to_box.sh 32672 root@213.5.72.235
#   bash deploy/sync_to_box.sh 32672 root@213.5.72.235 /workspace/extsum
#
# Sends code, configs, the cached reasoning traces and the silver label files,
# so every model on the box sees exactly the documents and labels used here.
# Does NOT send results, reports, logs, the virtualenv or .env: put the tokens
# on the box yourself (cp .env.example .env && nano .env).
set -euo pipefail

PORT="${1:?usage: sync_to_box.sh <ssh-port> <user@host> [remote project dir]}"
HOST="${2:?usage: sync_to_box.sh <ssh-port> <user@host> [remote project dir]}"
REMOTE="${3:-/workspace/extsum}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

$SSH "$HOST" "mkdir -p '$REMOTE'"
rsync -az --stats -e "$SSH" \
    --exclude '.venv/' --exclude '.env' --exclude '__pycache__/' --exclude '*.egg-info/' \
    --exclude '.pytest_cache/' --exclude '.DS_Store' \
    --exclude 'results/' --exclude 'reports/' --exclude 'logs/' \
    "$ROOT/" "$HOST:$REMOTE/"

echo
echo "synced to $HOST:$REMOTE"
echo "next, on the box:"
echo "  cd $REMOTE && cp .env.example .env && nano .env    # HF_TOKEN"
echo "  bash deploy/vast/setup.sh"
