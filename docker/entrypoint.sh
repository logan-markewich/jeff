#!/usr/bin/env bash
# Downloads the GLiFormer weights into $JEFF_MODEL on first start (they live
# on the "models" volume, so this only runs once per volume), then hands off
# to the requested command.
set -euo pipefail

MODEL_DIR="${JEFF_MODEL:-models/gliformer-large-v1}"
MODEL_REPO="${JEFF_MODEL_REPO:-knowledgator/gliformer-large-v1}"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "[entrypoint] ${MODEL_DIR} is empty, downloading ${MODEL_REPO} ..."
    mkdir -p "${MODEL_DIR}"
    uv run hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
else
    echo "[entrypoint] found existing weights in ${MODEL_DIR}, skipping download."
fi

exec uv run "$@"
