#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Disable proxies for local model service.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="10.0.0.18,100.69.30.24,100.124.83.129,localhost,127.0.0.1"

INPUT_DATA="${INPUT_DATA:-/nvme2/zzr/lzy/swe-m/test/output/output.json}"
IMAGE_DIR="${IMAGE_DIR:-/nvme2/zzr/lzy/swe-m/test/image}"
OUTPUT_DIR="${OUTPUT_DIR:-/nvme2/zzr/lzy/GALA/output/qwen35b_a3b_gala_test}"
REPO_PATH="${REPO_PATH:-/nvme2/zzr/lzy/swe-m/test/repo}"

#INPUT_DATA="${INPUT_DATA:-/nvme2/zzr/lzy/swe-m/dev/output/output.json}"
#IMAGE_DIR="${IMAGE_DIR:-/nvme2/zzr/lzy/swe-m/dev/image}"
#OUTPUT_DIR="${OUTPUT_DIR:-/nvme2/zzr/lzy/GALA/output/check00}"
#REPO_PATH="${REPO_PATH:-/nvme2/zzr/lzy/swe-m/dev/repo}"
INSTANCE_ID="${INSTANCE_ID:-}"
CFUSE_STREAM_MODE="${CFUSE_STREAM_MODE:-no-stream}"

# Use the same model endpoint for both LLM and VLM stages by default.
# Override MODEL_NAME / MODEL_BASE_URL in the environment if you need a
# different endpoint for image-capable models.
MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://10.0.0.18:46400/v1}"

# Local OpenAI-compatible services usually do not enforce API keys, but the code requires env vars.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export VLM_API_KEY="${VLM_API_KEY:-dummy}"

case "${CFUSE_STREAM_MODE}" in
  stream)
    export CFUSE_STREAM_FLAG="--stream"
    ;;
  no-stream)
    export CFUSE_STREAM_FLAG="--no-stream"
    ;;
  *)
    echo "Invalid CFUSE_STREAM_MODE: ${CFUSE_STREAM_MODE}"
    echo "Use CFUSE_STREAM_MODE=stream or CFUSE_STREAM_MODE=no-stream"
    exit 1
    ;;
esac

if [[ "${INPUT_DATA}" == "/path/to/input_data.json" || "${IMAGE_DIR}" == "/path/to/image_dir" || "${OUTPUT_DIR}" == "/path/to/output_dir" || "${REPO_PATH}" == "/path/to/repo" ]]; then
  echo "Please provide real paths via env vars:"
  echo "INPUT_DATA=/abs/path/input.json IMAGE_DIR=/abs/path/images OUTPUT_DIR=/abs/path/results REPO_PATH=/abs/path/repos bash scripts/full_run.sh"
  exit 1
fi

echo "CFUSE stream mode: ${CFUSE_STREAM_MODE} (${CFUSE_STREAM_FLAG})"

## Optional: download images first if your input_data uses remote URLs.
# python src/utils/download_image.py --input_data="${INPUT_DATA}" --image_dir="${IMAGE_DIR}"

# Full pipeline (image graph only):
# 1) generate-image-graph -> 2) localization -> 3) generate-patch -> 4) validation -> 5) process-result
python main.py full-run \
  --input_data "${INPUT_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --repo_path "${REPO_PATH}" \
  --vlm_model "${MODEL_NAME}" \
  --vlm_url "${MODEL_BASE_URL}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers 12 \
  --temperature 0.0 \
  ${INSTANCE_ID:+--instance_id "${INSTANCE_ID}"}
