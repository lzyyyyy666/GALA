#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Disable proxies for local model service.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="100.69.30.24,localhost,127.0.0.1"

# Reuse the same defaults as scripts/full_run.sh where possible.
SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:-/nvme2/zzr/lzy/GALA/output/output_new_alignment_10_15}"
OUTPUT_DIR="${OUTPUT_DIR:-/nvme2/zzr/lzy/GALA/output/output_new_alignment_prompt}"
REPO_PATH="${REPO_PATH:-/nvme2/zzr/lzy/swe-m/dev/repo}"
SOURCE_IMAGE_IR_PATH="${SOURCE_IMAGE_IR_PATH:-${SOURCE_OUTPUT_DIR}/image_ir_data.json}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://100.69.30.24:46406/v1/}"
MAX_WORKERS="${MAX_WORKERS:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROJECT_NAME="${PROJECT_NAME:-gala}"

# Local OpenAI-compatible services usually do not enforce API keys, but the code requires env vars.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export VLM_API_KEY="${VLM_API_KEY:-dummy}"

if [[ ! -f "${SOURCE_IMAGE_IR_PATH}" ]]; then
  echo "Missing source image IR file: ${SOURCE_IMAGE_IR_PATH}"
  exit 1
fi

if [[ ! -d "${SOURCE_OUTPUT_DIR}" ]]; then
  echo "Missing source output directory: ${SOURCE_OUTPUT_DIR}"
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
IMAGE_IR_PATH="${OUTPUT_DIR}/image_ir_data.json"

echo "Re-running pipeline from generate-patch"
echo "SOURCE_OUTPUT_DIR=${SOURCE_OUTPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REPO_PATH=${REPO_PATH}"
echo "SOURCE_IMAGE_IR_PATH=${SOURCE_IMAGE_IR_PATH}"
echo "IMAGE_IR_PATH=${IMAGE_IR_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_BASE_URL=${MODEL_BASE_URL}"

# Seed the new output directory with the existing per-instance context files.
find "${SOURCE_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -exec cp -r {} "${OUTPUT_DIR}/" \;
cp -f "${SOURCE_IMAGE_IR_PATH}" "${IMAGE_IR_PATH}"

# Clean validation/result artifacts so downstream outputs are fresh.
find "${OUTPUT_DIR}" -mindepth 2 \
  \( -name 'res_patch_*.patch' \
  -o -name '*_script.sh' \
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
rm -f "${OUTPUT_DIR}/${PROJECT_NAME}_result_path.json"
rm -f "${OUTPUT_DIR}/${PROJECT_NAME}_result_path.jsonl"

python main.py generate-patch \
  --image_ir_path "${IMAGE_IR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --repo_path "${REPO_PATH}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers "${MAX_WORKERS}" \
  --temperature "${TEMPERATURE}"

python main.py validation \
  --image_ir_path "${IMAGE_IR_PATH}" \
  --result_path "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers "${MAX_WORKERS}" \
  --repo_path "${REPO_PATH}"

python main.py process-result \
  --result_path "${OUTPUT_DIR}" \
  --project_name "${PROJECT_NAME}"
