#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Disable proxies for local model service.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="10.0.0.18,100.69.30.24,100.124.83.129,localhost,127.0.0.1"

# Reuse the same defaults as scripts/full_run.sh.
INPUT_DATA="${INPUT_DATA:-/nvme2/zzr/lzy/swe-m/test/output/output.json}"
IMAGE_DIR="${IMAGE_DIR:-/nvme2/zzr/lzy/swe-m/test/image}"
OUTPUT_DIR="${OUTPUT_DIR:-/nvme2/zzr/lzy/GALA/output/qwen122b_small_align}"
REPO_PATH="${REPO_PATH:-/nvme2/zzr/lzy/swe-m/test/repo}"
INSTANCE_ID="${INSTANCE_ID:-}"
CFUSE_STREAM_MODE="${CFUSE_STREAM_MODE:-no-stream}"

MODEL_NAME="${MODEL_NAME:-qwen3.5-122b}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://10.0.0.18:46400/v1}"
MAX_WORKERS="${MAX_WORKERS:-12}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROJECT_NAME="${PROJECT_NAME:-gala}"

IMAGE_IR_PATH="${IMAGE_IR_PATH:-${OUTPUT_DIR}/image_ir_data.json}"

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

if [[ ! -f "${IMAGE_IR_PATH}" ]]; then
  echo "Missing image IR file: ${IMAGE_IR_PATH}"
  exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Missing output directory: ${OUTPUT_DIR}"
  exit 1
fi

echo "Resume pipeline after interrupted validation"
echo "INPUT_DATA=${INPUT_DATA}"
echo "IMAGE_DIR=${IMAGE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REPO_PATH=${REPO_PATH}"
echo "IMAGE_IR_PATH=${IMAGE_IR_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_BASE_URL=${MODEL_BASE_URL}"
echo "MAX_WORKERS=${MAX_WORKERS}"
echo "CFUSE stream mode: ${CFUSE_STREAM_MODE} (${CFUSE_STREAM_FLAG})"

python main.py validation \
  --image_ir_path "${IMAGE_IR_PATH}" \
  --result_path "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --repo_path "${REPO_PATH}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers "${MAX_WORKERS}"

FAILED_DATA_PATH="${OUTPUT_DIR}/all_validation_failed_instance.json"
if [[ ! -f "${FAILED_DATA_PATH}" ]]; then
  echo "Missing failed-instance file after validation: ${FAILED_DATA_PATH}"
  exit 1
fi

python main.py redo-after-align-code-graph \
  --result_path "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --repo_path "${REPO_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --vlm_model "${MODEL_NAME}" \
  --vlm_url "${MODEL_BASE_URL}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers "${MAX_WORKERS}" \
  --temperature "${TEMPERATURE}"

python main.py process-result \
  --result_path "${OUTPUT_DIR}" \
  --project_name "${PROJECT_NAME}"
