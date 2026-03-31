#!/usr/bin/env bash
# Download all results from the EMBL cluster job directory.
# Requires sshpass: brew install sshpass

set -euo pipefail

REMOTE_USER="mirza"
REMOTE_HOST="login1.cluster.embl.de"
REMOTE_PASS="detemrinedE123$"
REMOTE_DIR="/g/torres-hd/mirza/other_projects/pinn_models"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/results"

mkdir -p "$LOCAL_DIR"

echo "Downloading from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} ..."

sshpass -p "$REMOTE_PASS" rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" \
    "$LOCAL_DIR/"

echo "Done. Files saved to: $LOCAL_DIR"
