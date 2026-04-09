#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="100.69.30.24,localhost,127.0.0.1"

OUTPUT_DIR="${OUTPUT_DIR:-/nvme2/zzr/lzy/GALA/output/full_run_dev}"
REDO_DIR="${REDO_DIR:-${OUTPUT_DIR}/redo_round}"
IMAGE_IR_PATH="${IMAGE_IR_PATH:-${OUTPUT_DIR}/image_ir_data.json}"
IMAGE_DIR="${IMAGE_DIR:-/nvme2/zzr/lzy/swe-m/dev/image}"
REPO_PATH="${REPO_PATH:-/nvme2/zzr/lzy/swe-m/dev/repo}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://100.69.30.24:46402/v1/}"
MAX_WORKERS="${MAX_WORKERS:-9}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROJECT_NAME="${PROJECT_NAME:-gala}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export VLM_API_KEY="${VLM_API_KEY:-dummy}"

if [[ ! -f "${IMAGE_IR_PATH}" ]]; then
  echo "Missing image IR file: ${IMAGE_IR_PATH}"
  exit 1
fi

rm -rf "${REDO_DIR}"
find "${OUTPUT_DIR}" -mindepth 2 \
  \( -name '*_script.sh' \
  -o -name 'user_prompt_*.txt' \
  -o -name 'test_log' \
  -o -name 'cropped_*' \
  -o -name 'nvme2-zzr-lzy-swe-m-dev-repo-*' \) \
  -exec rm -rf {} +
rm -f "${OUTPUT_DIR}/model_response_validation_failed.json"
rm -f "${OUTPUT_DIR}/model_response_validation_success.json"
rm -f "${OUTPUT_DIR}/agent_validation_failed_instance.json"
rm -f "${OUTPUT_DIR}/all_validation_failed_instance.json"
rm -f "${OUTPUT_DIR}/swebench_image_cropped_instance.json"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*result*.json' -delete

python main.py validation \
  --image_ir_path "${IMAGE_IR_PATH}" \
  --result_path "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers "${MAX_WORKERS}" \
  --repo_path "${REPO_PATH}"

REDO_IR_PATH="${OUTPUT_DIR}/all_validation_failed_instance.json"
if [[ ! -f "${REDO_IR_PATH}" ]]; then
  echo "Missing failed-instance file: ${REDO_IR_PATH}"
  exit 1
fi

python main.py build-code-graph \
  --repo_path "${REPO_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --result_path "${OUTPUT_DIR}" \
  --input_data "${REDO_IR_PATH}"

python main.py align-code-graph \
  --repo_path "${REPO_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --result_path "${OUTPUT_DIR}" \
  --input_data "${REDO_IR_PATH}"

python main.py generate-patch \
  --image_ir_path "${REDO_IR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --repo_path "${REPO_PATH}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers "${MAX_WORKERS}" \
  --temperature "${TEMPERATURE}"

python main.py process-result \
  --result_path "${OUTPUT_DIR}" \
  --project_name "${PROJECT_NAME}"
