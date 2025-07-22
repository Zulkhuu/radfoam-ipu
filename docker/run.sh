#!/bin/bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────

POPLAR_SDK_PATH="/home/zuk/Downloads/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7"
DOCKER_IMAGE="${USER}/poplar3.3_dev"
CONTAINER_NAME="${USER}_poplar3.3_docker"
HOST_PORT=5000
CONTAINER_PORT=5000

# ─── Load Poplar SDK (if not already enabled) ─────────────────────

if [[ -z "${POPLAR_SDK_ENABLED:-}" ]]; then
    source "${POPLAR_SDK_PATH}/enable"
fi

# ─── Check if container exists ────────────────────────────────────

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🟡 Container '${CONTAINER_NAME}' already exists."

    # If it's not running, start it
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "▶️ Starting existing container..."
        docker start "${CONTAINER_NAME}"
    fi
else
    echo "🚀 Creating and starting new container '${CONTAINER_NAME}'..."
    gc-docker -- \
        -it --rm --detach \
        --user root \
        -p 127.0.0.1:${HOST_PORT}:${CONTAINER_PORT} \
        --name "${CONTAINER_NAME}" \
        -v "$(realpath ..):/home/${USER}/radfoam_ipu/" \
        -w /home/${USER}/radfoam_ipu \
        --tmpfs /tmp/exec \
        "${DOCKER_IMAGE}"
fi

# ─── Enter Bash in the container ──────────────────────────────────

echo "🔧 Entering bash in container '${CONTAINER_NAME}'..."
docker exec -it "${CONTAINER_NAME}" bash
