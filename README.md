# GALA: Graph-Aligned Visual Repair for Multimodal Program Repair

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

GALA is a multimodal automated program repair pipeline for issues that include visual artifacts (for example screenshots).  
It converts visual inputs into structured image graphs, aligns them with repository code context, generates patches, validates them, and exports evaluation-ready results.

## Highlights

- End-to-end pipeline: image graph generation -> code graph localization -> patch generation -> validation -> result packaging
- Stage-by-stage CLI for debugging and ablation
- Batch processing support with configurable workers
- Compatible with OpenAI-style API endpoints for VLM/LLM services

## Repository Structure

- `main.py`: CLI entrypoint
- `src/`: core pipeline modules
- `src/utils/`: shared helpers (LLM client, image utils, data prep)
- `src/run_cmd/`: command templates used by patch/validation flow
- `prompt/`: prompt templates (`gala_prompt.py`, `image_prompt.py`, `test_prompt.py`, etc.)
- `scripts/`: runnable pipeline scripts

## Installation

```bash
git clone https://github.com/lzyyyyy666/GALA.git
cd GALA

export PYTHONPATH=./
pip install -r requirements.txt
```

## Data Preparation

Process SWE-bench Multimodal parquet and download repos:

```bash
python src/utils/process_data_dl_repo.py \
  --parquet-path /path/to/test-00000-of-00001.parquet \
  --output-path /path/to/output.json \
  --repo-dir /path/to/repo_dir
```

Optional image download step:

```bash
python src/utils/download_image.py \
  --input_data /path/to/output.json \
  --image_dir /path/to/image_dir
```

## Quick Start

Run the full pipeline with the provided script:

```bash
bash scripts/full_run.sh
```

Or run directly via CLI:

```bash
python main.py full-run \
  --input_data /path/to/input.json \
  --output_dir /path/to/results \
  --repo_path /path/to/repos \
  --image_dir /path/to/images \
  --vlm_model <vlm_model_name> \
  --vlm_url <vlm_base_url> \
  --model_name <llm_model_name> \
  --base_url <llm_base_url> \
  --project_name gala
```

## Environment Variables

```bash
export PYTHONPATH=./
export OPENAI_API_KEY="your-agent-api-key"
export VLM_API_KEY="your-vlm-api-key"
```

## CLI Commands

### `full-run`

Run complete GALA pipeline.

```bash
python main.py full-run \
  --input_data PATH \
  --output_dir PATH \
  --repo_path PATH \
  --image_dir PATH \
  --vlm_model MODEL \
  --vlm_url URL \
  --model_name MODEL \
  --base_url URL \
  [--temperature 0.0] \
  [--max_workers 4] \
  [--copy_repo] \
  [--project_name gala] \
  [--text_model_name MODEL] \
  [--text_base_url URL] \
  [--text_api_key KEY] \
  [--no_checkout_base_commit] \
  [--instance_id ID]
```

### `generate-image-graph`

Generate image graph IR from visual artifacts.

```bash
python main.py generate-image-graph \
  --model_name MODEL \
  --base_url URL \
  --input_data PATH \
  --image_dir PATH \
  --result_path PATH \
  --output_dir PATH \
  [--max_workers 4]
```

Compatibility alias:

```bash
python main.py generate-image-ir ...
```

### `build-code-graph`

Build repository snapshot and seed-file localization.

```bash
python main.py build-code-graph \
  --repo_path PATH \
  --image_dir PATH \
  --output_dir PATH \
  --base_url URL \
  --result_path PATH \
  [--model_name MODEL] \
  [--input_data PATH] \
  [--no_checkout_base_commit] \
  [--text_model_name MODEL] \
  [--text_base_url URL] \
  [--text_api_key KEY] \
  [--force_rebuild]
```

### `align-code-graph`

Build final aligned code graph from existing seed artifacts.

```bash
python main.py align-code-graph \
  --repo_path PATH \
  --image_dir PATH \
  --output_dir PATH \
  --base_url URL \
  --result_path PATH \
  [--model_name MODEL] \
  [--input_data PATH] \
  [--no_checkout_base_commit] \
  [--text_model_name MODEL] \
  [--text_base_url URL] \
  [--text_api_key KEY] \
  [--force_rebuild]
```

### `generate-patch`

Generate patch files from image/code graph context.

```bash
python main.py generate-patch \
  --image_ir_path PATH \
  --output_dir PATH \
  --repo_path PATH \
  --base_url URL \
  [--model_name MODEL] \
  [--temperature 0.0] \
  [--max_workers 4] \
  [--copy_repo]
```

### `validation`

Run validation on generated patches.

```bash
python main.py validation \
  --image_ir_path PATH \
  --result_path PATH \
  --output_dir PATH \
  --base_url URL \
  [--model_name MODEL] \
  [--max_workers 4] \
  [--repo_path PATH] \
  [--copy_repo]
```

### `redo-after-align-code-graph`

Rerun patch generation for validation-failed instances using existing code-graph artifacts.

```bash
python main.py redo-after-align-code-graph \
  --result_path PATH \
  --output_dir PATH \
  --repo_path PATH \
  --image_dir PATH \
  --vlm_url URL \
  --base_url URL \
  [--vlm_model MODEL] \
  [--model_name MODEL] \
  [--temperature 0.0] \
  [--copy_repo] \
  [--max_workers 4]
```

### `process-result`

Export final result files for evaluation.

```bash
python main.py process-result \
  --result_path PATH \
  [--project_name gala] \
  [--result_tag round1]
```

## Output Layout

Typical output structure:

```text
results/
├── image_ir_data.json
├── <instance_id>/
│   ├── code_graph_<instance_id>.json
│   ├── res_patch_<instance_id>.patch
│   └── resp_<instance_id>.json
├── all_validation_failed_instance.json
├── gala_round1_result_path.json
├── gala_round1_result_path.jsonl
├── gala_final_result_path.json
└── gala_final_result_path.jsonl
```

## Notes for Open-Source Usage

- Do not hardcode API keys or internal URLs in committed code.
- Prefer environment variables for credentials.
- If you publish benchmark numbers, ensure they match your released code/config.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
