#!/bin/bash
set -euo pipefail

# ─── Config ────────────────────────────────────────────────────────

CONTAINER_SSH_PORT=2023
IMAGE_NAME="${USER}/poplar3.3_dev"
BUILD_ARGS=()

# ─── Parse Arguments ───────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--image NAME] [docker build options like --no-cache --pull ...]"
            exit 0
            ;;
        *)
            BUILD_ARGS+=("$1")
            shift
            ;;
    esac
done

# ─── Build the Docker image ───────────────────────────────────────

echo "🐳 Building Docker image '${IMAGE_NAME}'..."
docker build -t "${IMAGE_NAME}" "${BUILD_ARGS[@]}" \
    --build-arg UNAME="$USER" \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg CUSTOM_SSH_PORT="$CONTAINER_SSH_PORT" \
    .
