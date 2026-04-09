import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.code_graph_builder import CodeGraphBuilder
from src.utils.repo_resolver import resolve_repo_dir


def normalize_chat_completions_url(base_url: str) -> str:
    stripped = (base_url or "").strip()
    if not stripped:
        return ""
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/"):
        return stripped + "chat/completions"
    return stripped + "/chat/completions"


def resolve_input_data_path(result_path: str, input_data: str) -> str:
    if input_data:
        return input_data
    preferred = os.path.join(result_path, "image_ir_data.json")
    if os.path.exists(preferred):
        return preferred
    fallback = os.path.join(result_path, "all_validation_failed_instance.json")
    return fallback


def load_input_data(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input data not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError(f"Input data must be a dict keyed by instance_id: {path}")
    return data


def load_instance_doc_from_dir(instance_dir: str) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(instance_dir):
        return None

    json_files = sorted(glob.glob(os.path.join(instance_dir, "*.json")))
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        if (
            file_name.startswith("code_graph_")
            or file_name.startswith("subgraph_")
            or file_name.startswith("repo_structure_")
            or file_name.startswith("seed_nodes_")
        ):
            continue
        try:
            with open(json_file, "r", encoding="utf-8-sig") as infile:
                loaded = json.load(infile)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            continue
    return None


def require_nonempty_field(doc: Dict[str, Any], field: str, instance_id: str) -> str:
    value = str(doc.get(field) or "").strip()
    if not value:
        raise ValueError(f"Missing required field `{field}` for instance: {instance_id}")
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=4, ensure_ascii=False)


def build_seed_failure_message(builder: CodeGraphBuilder, repo_dir: str) -> str:
    diagnostics = builder.last_seed_diagnostics if isinstance(builder.last_seed_diagnostics, dict) else {}
    reason = str(diagnostics.get("reason") or "unknown")
    all_files_count = diagnostics.get("all_files_count")
    model_name = str(diagnostics.get("model_name") or "")
    base_url = str(diagnostics.get("base_url") or "")
    api_key_present = bool(diagnostics.get("api_key_present"))

    lines = ["Seed nodes are empty (seed_files=0)."]

    if reason == "no_candidate_files_indexed":
        lines.append(
            "Cause: the code graph builder indexed 0 candidate files from the target repo."
        )
        lines.append(
            f"Repo dir: {repo_dir}. This builder currently supports JS/TS source files plus file-level candidates such as `.scss`, `.json`, `.html`, `.md`, `.mdx`, `.svg`, `.frag`, `.vert`, and no-extension files."
        )
    elif reason == "missing_llm_config":
        lines.append(
            "Cause: missing LLM configuration. `text_model_name`, `text_base_url`, and `text_api_key` are all required."
        )
    elif reason == "llm_request_failed":
        lines.append(
            f"Cause: LLM request failed with exception: {diagnostics.get('exception', '')}"
        )
    elif reason == "llm_output_unparseable_or_empty":
        lines.append(
            "Cause: the LLM returned empty output or output that could not be parsed into a fenced code block of candidate file paths."
        )
    elif reason == "llm_returned_non_matching_paths":
        lines.append(
            "Cause: the LLM returned file paths, but none matched indexed candidate files in the repository."
        )
    else:
        lines.append("Cause: unable to determine automatically.")

    lines.append(
        f"Diagnostics: model={model_name or '<empty>'}, base_url={base_url or '<empty>'}, api_key_present={api_key_present}, indexed_candidate_files={all_files_count}."
    )

    if diagnostics.get("model_found_files"):
        lines.append(f"LLM first-pass files: {diagnostics['model_found_files']}")
    if diagnostics.get("found_files_after_first_pass"):
        lines.append(f"Matched first-pass files: {diagnostics['found_files_after_first_pass']}")
    if diagnostics.get("model_found_files_after_format_fix"):
        lines.append(
            f"LLM format-fix files: {diagnostics['model_found_files_after_format_fix']}"
        )
    if diagnostics.get("found_files_after_format_fix"):
        lines.append(
            f"Matched format-fix files: {diagnostics['found_files_after_format_fix']}"
        )
    if diagnostics.get("reflection_files"):
        lines.append(f"LLM reflection files: {diagnostics['reflection_files']}")
    if diagnostics.get("reflection_files_in_repo"):
        lines.append(
            f"Matched reflection files: {diagnostics['reflection_files_in_repo']}"
        )

    return " ".join(lines)


def run_single_instance(args: argparse.Namespace) -> int:
    instance_id = args.instance_id.strip()
    if not instance_id:
        raise ValueError("`--instance_id` cannot be empty")

    input_data_path = resolve_input_data_path(args.result_path, args.input_data or "")
    input_data = load_input_data(input_data_path)

    instance_dir = os.path.join(args.result_path, instance_id)
    doc = input_data.get(instance_id)
    if not isinstance(doc, dict):
        doc = load_instance_doc_from_dir(instance_dir)
    if not isinstance(doc, dict):
        raise KeyError(
            f"Instance `{instance_id}` not found in input data and no usable json found under {instance_dir}"
        )

    repo_identifier = require_nonempty_field(doc, "repo", instance_id)
    base_commit = require_nonempty_field(doc, "base_commit", instance_id)
    problem_statement = require_nonempty_field(doc, "problem_statement", instance_id)

    repo_dir = resolve_repo_dir(args.repo_path, repo_identifier)
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"Repo directory not found: {repo_dir}")

    text_model_name = (args.text_model_name or os.getenv("TEXT_MODEL_NAME", "")).strip()
    text_base_url = normalize_chat_completions_url(
        args.text_base_url or os.getenv("TEXT_BASE_URL", "")
    )
    text_api_key = (
        args.text_api_key
        or os.getenv("TEXT_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()

    builder = CodeGraphBuilder(
        model_name=text_model_name,
        base_url=text_base_url,
        api_key=text_api_key,
        top_n_seed_files=5,
    )

    payload = builder.build_instance_graph(
        instance_id=instance_id,
        repo_identifier=repo_identifier,
        base_commit=base_commit,
        repo_dir=repo_dir,
        problem_statement=problem_statement,
        checkout_base_commit=args.checkout_base_commit,
    )

    fallback_reason = str(payload.get("fallback_reason") or "")
    if args.checkout_base_commit and fallback_reason.startswith("checkout_failed"):
        raise RuntimeError(f"Checkout base_commit failed: {fallback_reason}")

    code_graph = payload.get("code_graph", {})
    if not isinstance(code_graph, dict):
        raise RuntimeError("Invalid graph payload: missing `code_graph` dict")

    seed_files = code_graph.get("seed_files", [])
    if not isinstance(seed_files, list):
        raise RuntimeError("Invalid graph payload: `seed_files` is not a list")
    if len(seed_files) == 0:
        raise RuntimeError(build_seed_failure_message(builder, repo_dir))

    repo_structure = payload.get("repo_structure", {})
    if not isinstance(repo_structure, dict):
        raise RuntimeError("Invalid graph payload: missing `repo_structure` dict")

    output_dir = args.output_dir or os.path.join(args.result_path, instance_id)
    os.makedirs(output_dir, exist_ok=True)

    repo_structure_path = os.path.join(output_dir, f"debug_repo_structure_{instance_id}.json")
    code_graph_path = os.path.join(output_dir, f"debug_code_graph_{instance_id}.json")

    write_json(repo_structure_path, repo_structure)
    write_json(code_graph_path, code_graph)

    print(f"instance_id: {instance_id}")
    print(f"input_data: {input_data_path}")
    print(f"repo_dir: {repo_dir}")
    print(f"checkout_base_commit: {args.checkout_base_commit}")
    print(f"repo_structure saved: {repo_structure_path}")
    print(f"code_graph saved: {code_graph_path}")
    print(f"seed_file_count: {len(seed_files)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug helper for building single-instance CoSIL-style code graph artifacts."
    )
    parser.add_argument("--instance_id", required=True, help="Target instance id")
    parser.add_argument("--result_path", required=True, help="Result root containing instance folders")
    parser.add_argument("--repo_path", required=True, help="Repository root path")
    parser.add_argument(
        "--input_data",
        help="Input dataset json. Default: result_path/image_ir_data.json, fallback to result_path/all_validation_failed_instance.json",
    )
    parser.add_argument("--text_model_name", help="Text model for CoSIL-style seed localization")
    parser.add_argument("--text_base_url", help="Text model base URL")
    parser.add_argument("--text_api_key", help="Text model API key (or use env TEXT_API_KEY)")
    parser.add_argument(
        "--checkout_base_commit",
        action="store_true",
        help="Enable checkout to base_commit before parsing repository structure (default: disabled for debug runs)",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory. Default: result_path/<instance_id>",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_single_instance(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
