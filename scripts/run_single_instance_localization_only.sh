#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="10.0.0.18,100.69.30.24,100.124.83.129,localhost,127.0.0.1"

INPUT_DATA="${INPUT_DATA:-/nvme2/zzr/lzy/swe-m/dev/output/output.json}"
IMAGE_DIR="${IMAGE_DIR:-/nvme2/zzr/lzy/swe-m/dev/image}"
OUTPUT_DIR="${OUTPUT_DIR:-/nvme2/zzr/lzy/GALA/output/single_instance_localization_only}"
REPO_PATH="${REPO_PATH:-/nvme2/zzr/lzy/swe-m/dev/repo}"
INSTANCE_ID="${INSTANCE_ID:-}"

MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://100.69.30.24:46406/v1/}"

TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-${MODEL_NAME}}"
TEXT_BASE_URL="${TEXT_BASE_URL:-${MODEL_BASE_URL}}"
TEXT_API_KEY="${TEXT_API_KEY:-${OPENAI_API_KEY:-dummy}}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export VLM_API_KEY="${VLM_API_KEY:-dummy}"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Please provide INSTANCE_ID for a single validation instance."
  echo "Example:"
  echo "INSTANCE_ID=Automattic__wp-calypso-21409 bash scripts/run_single_instance_localization_only.sh"
  exit 1
fi

if [[ "${INPUT_DATA}" == "/path/to/input_data.json" || "${IMAGE_DIR}" == "/path/to/image_dir" || "${OUTPUT_DIR}" == "/path/to/output_dir" || "${REPO_PATH}" == "/path/to/repo" ]]; then
  echo "Please provide real paths via env vars:"
  echo "INPUT_DATA=/abs/path/input.json IMAGE_DIR=/abs/path/images OUTPUT_DIR=/abs/path/results REPO_PATH=/abs/path/repos INSTANCE_ID=<id> bash scripts/run_single_instance_localization_only.sh"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

SINGLE_INPUT_PATH="$(python - "${INPUT_DATA}" "${OUTPUT_DIR}" "${INSTANCE_ID}" <<'PY'
import json
import os
import re
import sys

input_data_path, output_dir, instance_id = sys.argv[1:4]

with open(input_data_path, "r", encoding="utf-8") as infile:
    loaded = json.load(infile)

target = str(instance_id).strip()
if not target:
    raise ValueError("empty instance_id")

filtered = {}
if isinstance(loaded, dict):
    if target in loaded:
        filtered[target] = loaded[target]
    else:
        for key, value in loaded.items():
            if isinstance(value, dict) and str(value.get("instance_id") or "").strip() == target:
                filtered[target] = value
                break
elif isinstance(loaded, list):
    for item in loaded:
        if isinstance(item, dict) and str(item.get("instance_id") or "").strip() == target:
            filtered[target] = item
            break
else:
    raise ValueError("input_data must be dict or list JSON")

if not filtered:
    raise ValueError(f"instance_id not found: {target}")

safe_instance = re.sub(r"[^A-Za-z0-9._-]+", "_", target)
output_path = os.path.join(output_dir, f"single_instance_{safe_instance}.json")
with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(filtered, outfile, indent=4, ensure_ascii=False)

print(output_path)
PY
)"

echo "Single-instance mode enabled: ${INSTANCE_ID}"
echo "Filtered input saved to: ${SINGLE_INPUT_PATH}"
echo "Running localization-only pipeline:"
echo "1) generate-image-ir -> 2) build-code-graph -> 3) align-code-graph"

python main.py generate-image-ir \
  --input_data "${SINGLE_INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --result_path "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --max_workers 1

python main.py build-code-graph \
  --input_data "${SINGLE_INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --repo_path "${REPO_PATH}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --result_path "${OUTPUT_DIR}" \
  --text_model_name "${TEXT_MODEL_NAME}" \
  --text_base_url "${TEXT_BASE_URL}" \
  --text_api_key "${TEXT_API_KEY}"

python main.py align-code-graph \
  --input_data "${SINGLE_INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --repo_path "${REPO_PATH}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --result_path "${OUTPUT_DIR}" \
  --text_model_name "${TEXT_MODEL_NAME}" \
  --text_base_url "${TEXT_BASE_URL}" \
  --text_api_key "${TEXT_API_KEY}"

echo "Localization-only run complete."
echo "Artifacts:"
echo "  image_ir_data.json: ${OUTPUT_DIR}/image_ir_data.json"
echo "  instance dir: ${OUTPUT_DIR}/${INSTANCE_ID}"
echo "  code graph: ${OUTPUT_DIR}/${INSTANCE_ID}/code_graph_${INSTANCE_ID}.json"
