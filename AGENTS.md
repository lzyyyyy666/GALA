# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entrypoint for all GALA stages (`full-run`, `generate-image-graph`, `generate-image-ir`, `generate-patch`, `validation`, `build-code-graph`, `align-code-graph`, `redo-after-align-code-graph`, `process-result`).
- `src/`: core pipeline modules (`generate_image_ir.py`, `main_docker_run.py`, `validation.py`, `image_repo_localization.py`, `process_result.py`).
- `src/utils/`: shared helpers for config, logging, image download/processing, and dataset preparation.
- `src/run_cmd/`: command helpers used by repair/validation flow.
- `prompt/`: prompt templates for IR generation and validation (`gala_prompt.py`, `image_prompt.py`, `test_prompt.py`).
- `scripts/full_run.sh`: end-to-end example script.

## Build, Test, and Development Commands
- Install dependencies:
  ```bash
  export PYTHONPATH=./
  pip install -r requirements.txt
  ```
- Run full pipeline:
  ```bash
  python main.py full-run --input_data <input.json> --output_dir <results_dir> --repo_path <repos_dir> --image_dir <images_dir> --vlm_url <vlm_api> --base_url <llm_api>
  ```
- Run staged commands individually:
  - `python main.py generate-image-graph ...`
  - `python main.py generate-image-ir ...`
  - `python main.py generate-patch ...`
  - `python main.py validation ...`
  - `python main.py build-code-graph ...`
  - `python main.py align-code-graph ...`
- Dataset/repo preparation:
  - `python src/utils/process_data_dl_repo.py --parquet-path <file> --output-path <output.json> --repo-dir <repo_dir>`
  - `python src/utils/download_image.py --input_data <input.json> --image_dir <image_dir>`

## Coding Style & Naming Conventions
- Target Python 3.10+; follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables/files, `PascalCase` for classes, and explicit module names mirroring pipeline stage purpose.
- Keep CLI flags long-form and descriptive (for example, `--output_dir`, `--max_workers`).
- Prefer small, composable functions in `src/utils/` for reusable logic.

## Testing Guidelines
- No dedicated unit-test suite is currently checked in; validate changes via pipeline commands on a small sample dataset.
- For patch-quality checks, follow prompt rules in `prompt/test_prompt.py` (format validity, no runtime/model errors, downstream test success).
- When adding tests, place them under a new `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
- Current history uses short, imperative commit subjects (for example, `add scripts`, `rename`, `remove DS_Store`).
- Recommended commit format: `<scope>: <imperative summary>` (example: `validation: handle empty failed set`).
- PRs should include:
  - What changed and why.
  - Commands run to verify behavior.
  - Required env vars/API settings (`OPENAI_API_KEY`, `VLM_API_KEY`, `PYTHONPATH`).
  - Sample output paths or logs for non-trivial pipeline changes.

## Security & Configuration Tips
- Never hardcode keys or internal API URLs in code or scripts.
- Use environment variables for credentials and keep local paths out of committed defaults.
