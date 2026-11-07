#!/usr/bin/env bash
# Upload this repository (code, git history, prepared data/rl) to an AutoDL instance.
# Take host and port from the instance's SSH command, e.g. "ssh -p <port> root@<host>":
#   bash scripts/sync_to_autodl.sh <host> <port>
# Add --dry-run as a third argument to list what would be sent. ssh asks for the password.
# Never uploaded: .env (API key), virtualenvs, caches, raw downloads, databases, and anything
# listed in the local .git/info/exclude (personal files kept out of git).
set -euo pipefail
HOST=${1:?AutoDL host from the instance SSH command}
PORT=${2:?AutoDL SSH port}
DRY=${3:-}
DEST=${DEST:-/root/autodl-tmp/agentos}
cd "$(dirname "$0")/.."

EXCLUDES=(
  --include ".env.example" --exclude ".env" --exclude ".env.*"   # first match wins
  --exclude ".venv/" --exclude ".venv-*/" --exclude "__pycache__/" --exclude ".pytest_cache/" --exclude ".ruff_cache/"
  --exclude ".DS_Store" --exclude ".vscode/" --exclude "data/raw/" --exclude "*.db" --exclude "*.db-shm" --exclude "*.db-wal"
  --exclude "experiments/results/**/stores/"
)
if [[ -f .git/info/exclude ]]; then EXCLUDES+=(--exclude-from .git/info/exclude); fi

# Flags kept to what macOS's bundled openrsync supports.
rsync -az --partial --progress ${DRY:+--dry-run} "${EXCLUDES[@]}" \
  -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" ./ "root@$HOST:$DEST/"
echo "Uploaded to root@$HOST:$DEST"
