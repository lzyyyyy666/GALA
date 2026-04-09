import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

from src.generate_image_ir import GenerateIR
from src.image_repo_localization import ImageCodeLocalization, normalize_chat_completions_url
from src.utils.llm_client import get_usage_totals, reset_usage_totals, send_chat_completion

DEFAULT_INPUT_DATA = "/nvme2/zzr/lzy/swe-m/dev/output/output.json"
DEFAULT_IMAGE_DIR = "/nvme2/zzr/lzy/swe-m/dev/image"
DEFAULT_OUTPUT_DIR = "/nvme2/zzr/lzy/GALA/GUI_output/full_dev_no_align"
DEFAULT_REPO_PATH = "/nvme2/zzr/lzy/swe-m/dev/repo"
DEFAULT_MODEL_NAME = "qwen3.5-35b-a3b"
DEFAULT_MODEL_BASE_URL = "http://100.69.30.24:46406/v1/"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
    }


def _print_step_usage(step_name: str, start_ts: float, end_ts: float, usage: Dict[str, int]) -> None:
    print(
        f"[STEP] {step_name}: "
        f"duration={_format_duration(end_ts - start_ts)}, "
        f"requests={usage.get('requests', 0)}, "
        f"prompt_tokens={usage.get('prompt_tokens', 0)}, "
        f"completion_tokens={usage.get('completion_tokens', 0)}, "
        f"total_tokens={usage.get('total_tokens', 0)}"
    )


def _load_json_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as infile:
        loaded = json.load(infile)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return loaded


def _build_explanation(code_graph_payload: Dict[str, Any]) -> str:
    code_graph = code_graph_payload.get("code_graph", {})
    if not isinstance(code_graph, dict):
        return ""

    alignment_plan = code_graph.get("alignment_plan", {})
    if not isinstance(alignment_plan, dict):
        return ""

    subgraph_alignments = alignment_plan.get("subgraph_alignments", [])
    if not isinstance(subgraph_alignments, list):
        return ""

    reasons: List[str] = []
    for item in subgraph_alignments:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("why_it_matches") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)

    if reasons:
        return " ".join(reasons)

    matched_files = alignment_plan.get("matched_files", [])
    if isinstance(matched_files, list) and matched_files:
        return "Selected based on image-code alignment and repository structure analysis."

    return ""


def _extract_bug_files(code_graph_payload: Dict[str, Any]) -> List[str]:
    code_graph = code_graph_payload.get("code_graph", {})
    if not isinstance(code_graph, dict):
        return []

    for field_name in ("reranked_top15_files", "selected_files", "seed_files"):
        value = code_graph.get(field_name, [])
        if isinstance(value, list):
            files = [str(item).strip() for item in value if str(item).strip()]
            if files:
                return files
    return []


def _extract_text_content(response: Dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    parts.append(stripped)
        return "\n".join(parts).strip()
    return ""


def _generate_bug_file_explanation(
    raw_doc: Dict[str, Any],
    code_graph_payload: Dict[str, Any],
    bug_files: List[str],
    model_name: str,
    base_url: str,
    api_key: str,
) -> str:
    if not bug_files or not model_name or not base_url or not api_key:
        return ""

    issue_text = str(raw_doc.get("problem_statement") or "").strip()
    repo_structure = code_graph_payload.get("repo_structure", {})
    try:
        repo_structure_text = json.dumps(repo_structure, ensure_ascii=False, indent=2)
    except Exception:
        repo_structure_text = str(repo_structure)
    system_prompt = (
        "The user will provide the Bug Report and Repository Structure, and also a final ranked list of "
        "bug-related files that has already been identified."
    )
    user_prompt = f"""
I will give you the bug related information (i.e., Bug Report) for your references, and I will also give you the final suspicious bug related files that have already been selected from the code Repo.

1. Read the bug report to understand the bug scenario;
2. Look at the Repository Structure for context;
3. Based on the provided final bug related files, write Explanation of why these files are bug related.

* Bug Report
'''
{issue_text or "none"}
'''

* Repository Structure
'''
{repo_structure_text}
'''

* Final Bug Related Files
'''
{chr(10).join(bug_files)}
'''
""".strip()

    try:
        response = send_chat_completion(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
    except Exception as exc:
        print(f"Failed to generate bug file explanation: {exc}")
        return ""

    return _extract_text_content(response)


def _build_gui_record(
    raw_doc: Dict[str, Any],
    code_graph_payload: Dict[str, Any],
    explanation_model_name: str,
    explanation_base_url: str,
    explanation_api_key: str,
) -> Dict[str, Any]:
    bug_scenario = str(raw_doc.get("problem_statement") or "").strip()
    bug_files = _extract_bug_files(code_graph_payload)
    explanation = _generate_bug_file_explanation(
        raw_doc=raw_doc,
        code_graph_payload=code_graph_payload,
        bug_files=bug_files,
        model_name=explanation_model_name,
        base_url=explanation_base_url,
        api_key=explanation_api_key,
    )
    if not explanation:
        explanation = _build_explanation(code_graph_payload)
    if not explanation and bug_files:
        explanation = (
            "These files were selected as bug related because they are the strongest matches to the bug report "
            "after image-guided localization, code-graph alignment, and final file reranking."
        )
    return {
        "bug_scenario": bug_scenario,
        "bug_files": bug_files,
        "explanation": explanation,
    }


def _build_single_instance_input(
    input_data_path: str,
    output_dir: str,
    instance_id: str,
) -> str:
    data = _load_json_dict(input_data_path)
    target = str(instance_id or "").strip()
    if not target:
        raise ValueError("empty instance_id")
    if target not in data or not isinstance(data[target], dict):
        raise ValueError(f"instance_id not found: {target}")

    safe_instance = re.sub(r"[^A-Za-z0-9._-]+", "_", target)
    output_path = os.path.join(output_dir, f"single_instance_{safe_instance}.json")
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump({target: data[target]}, outfile, indent=4, ensure_ascii=False)
    return output_path


def _collect_gui_output(
    image_ir_path: str,
    result_path: str,
    explanation_model_name: str,
    explanation_base_url: str,
    explanation_api_key: str,
) -> Dict[str, Dict[str, Any]]:
    image_ir_data = _load_json_dict(image_ir_path)
    output: Dict[str, Dict[str, Any]] = {}
    for instance_id, raw_doc in image_ir_data.items():
        if not isinstance(raw_doc, dict):
            continue
        code_graph_path = os.path.join(result_path, instance_id, f"code_graph_{instance_id}.json")
        if not os.path.exists(code_graph_path):
            continue
        code_graph_payload = _load_json_dict(code_graph_path)
        output[instance_id] = {
            "1": _build_gui_record(
                raw_doc,
                code_graph_payload,
                explanation_model_name=explanation_model_name,
                explanation_base_url=explanation_base_url,
                explanation_api_key=explanation_api_key,
            )
        }
    return output


def run_gui_pipeline(args: argparse.Namespace) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    input_data_path = args.input_data
    if str(args.instance_id or "").strip():
        input_data_path = _build_single_instance_input(
            input_data_path=input_data_path,
            output_dir=args.output_dir,
            instance_id=args.instance_id,
        )

    step_start_ts = time.time()
    usage_before = get_usage_totals()
    image_ir = GenerateIR(args.vlm_model, args.vlm_url)
    image_ir.process_batch(
        result_path=args.output_dir,
        input_data=input_data_path,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        max_workers=args.max_workers,
    )
    image_ir_usage = _usage_delta(usage_before, get_usage_totals())
    _print_step_usage("image_ir", step_start_ts, time.time(), image_ir_usage)
    image_ir_path = os.path.join(args.output_dir, "image_ir_data.json")

    step_start_ts = time.time()
    usage_before = get_usage_totals()
    localizer = ImageCodeLocalization(
        model_name=args.vlm_model,
        base_url=args.vlm_url,
        repo_path=args.repo_path,
        checkout_base_commit=not args.no_checkout_base_commit,
        text_model_name=(args.text_model_name or args.model_name or ""),
        text_base_url=(args.text_base_url or args.base_url or ""),
        text_api_key=args.text_api_key or "",
    )
    localizer.process_batch(
        data_path=image_ir_path,
        result_path=args.output_dir,
        image_dir=args.image_dir,
        max_workers=args.max_workers,
    )
    localization_usage = _usage_delta(usage_before, get_usage_totals())
    _print_step_usage("localization", step_start_ts, time.time(), localization_usage)

    gui_output = _collect_gui_output(
        image_ir_path=image_ir_path,
        result_path=args.output_dir,
        explanation_model_name=(args.text_model_name or args.model_name or ""),
        explanation_base_url=normalize_chat_completions_url(args.text_base_url or args.base_url or ""),
        explanation_api_key=(
            args.text_api_key
            or os.getenv("TEXT_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("VLM_API_KEY", "")
        ),
    )
    final_output_path = args.gui_output_path or os.path.join(args.output_dir, "gui_bug_files.json")
    with open(final_output_path, "w", encoding="utf-8") as outfile:
        json.dump(gui_output, outfile, indent=4, ensure_ascii=False)
    return final_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GALA image-IR + localization, then export GUI-style file localization JSON."
    )
    parser.add_argument("--input_data", default=DEFAULT_INPUT_DATA, help="Input dataset json")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--repo_path", default=DEFAULT_REPO_PATH, help="Repository root path")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Image directory")
    parser.add_argument("--vlm_model", default=DEFAULT_MODEL_NAME, help="VLM model name")
    parser.add_argument("--vlm_url", default=DEFAULT_MODEL_BASE_URL, help="VLM base URL")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Text model name, aligned with full_run.sh/main.py")
    parser.add_argument("--base_url", default=DEFAULT_MODEL_BASE_URL, help="Text model base URL, aligned with full_run.sh/main.py")
    parser.add_argument("--text_model_name", help="Text model used for file localization")
    parser.add_argument("--text_base_url", help="Text model base URL")
    parser.add_argument("--text_api_key", help="Text model API key")
    parser.add_argument("--max_workers", type=int, default=6, help="Max workers")
    parser.add_argument(
        "--instance_id",
        default="",
        help="Run GUI pipeline for one instance only; default empty means run the full dataset",
    )
    parser.add_argument(
        "--gui_output_path",
        help="Final GUI-format JSON path. Default: <output_dir>/gui_bug_files.json",
    )
    parser.add_argument(
        "--no_checkout_base_commit",
        action="store_true",
        help="Disable checkout to base_commit before code graph parsing",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ.setdefault("VLM_API_KEY", "dummy")
    reset_usage_totals()
    start_ts = time.time()
    output_path = run_gui_pipeline(args)
    usage = get_usage_totals()
    end_ts = time.time()
    print(f"GUI-format localization saved: {output_path}")
    print(
        "Runtime summary: "
        f"duration={_format_duration(end_ts - start_ts)}, "
        f"requests={usage.get('requests', 0)}, "
        f"prompt_tokens={usage.get('prompt_tokens', 0)}, "
        f"completion_tokens={usage.get('completion_tokens', 0)}, "
        f"total_tokens={usage.get('total_tokens', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
