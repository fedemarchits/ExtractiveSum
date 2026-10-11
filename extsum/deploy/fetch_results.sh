#!/bin/bash
# Copy results from a rented box to this machine and verify the copy.
# Run it ON YOUR LAPTOP, from the project root, before destroying the instance:
#
#   bash deploy/fetch_results.sh 32672 root@213.5.72.235
#   bash deploy/fetch_results.sh 32672 root@213.5.72.235 /workspace/repo/extsum
#
# Copies results/ (every raw model answer), reports/ and logs/. Nothing local is
# deleted. The final step re-compares checksums; "0 files differ" means the
# instance is safe to destroy.
set -euo pipefail

PORT="${1:?usage: fetch_results.sh <ssh-port> <user@host> [remote project dir]}"
HOST="${2:?usage: fetch_results.sh <ssh-port> <user@host> [remote project dir]}"
REMOTE="${3:-/workspace/ExtractiveSummarization/extsum}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

for DIR in results reports logs; do
    mkdir -p "$DIR"
    rsync -az --stats -e "$SSH" "$HOST:$REMOTE/$DIR/" "$DIR/"
done

echo
echo "verifying checksums..."
DIFF=0
for DIR in results reports logs; do
    N=$(rsync -azn --checksum --out-format='%n' -e "$SSH" "$HOST:$REMOTE/$DIR/" "$DIR/" | grep -vc '/$' || true)
    DIFF=$((DIFF + N))
done
echo "$DIFF files differ"
[ "$DIFF" -eq 0 ] && echo "copy verified: the instance can be destroyed." || exit 1
