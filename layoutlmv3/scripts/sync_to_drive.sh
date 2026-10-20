#!/usr/bin/env bash
# Upload a run directory to Google Drive with rclone. Uses `copy` (never deletes anything on Drive).
#
#   DRIVE_REMOTE=gdrive bash scripts/sync_to_drive.sh ~/runs/funsd-base-seed42
#
# Layout on Drive: <DRIVE_DIR>/<run dir name>/...   (DRIVE_DIR defaults to thesis/layoutlmv3)
set -euo pipefail

SRC="${1:?usage: $0 <local_run_dir>}"
REMOTE="${DRIVE_REMOTE:-gdrive}"
DEST="${DRIVE_DIR:-thesis/layoutlmv3}/$(basename "$SRC")"

command -v rclone >/dev/null || { echo "rclone not installed"; exit 1; }
rclone lsd "$REMOTE:" >/dev/null 2>&1 || { echo "rclone remote '$REMOTE' not working (run: rclone config)"; exit 1; }

rclone copy "$SRC" "$REMOTE:$DEST" \
  --transfers 4 --checkers 8 --drive-chunk-size 128M --retries 5 --low-level-retries 20 --progress
rclone check "$SRC" "$REMOTE:$DEST" --one-way && echo "Uploaded and verified: $REMOTE:$DEST"

