import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict
from datetime import datetime

from src.generate_image_ir import GenerateIR
from src.main_docker_run import PatchGeneration
from src.validation import Validation
from src.agent_validation import AgentBaseValidation
from src.process_validation import process_val
from src.image_repo_localization import ImageCodeLocalization
from src.process_result import process_git_diff
from src.utils.llm_client import get_usage_totals, reset_usage_totals


def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _record_step_timing(records: list[dict[str, Any]], step_name: str, start_ts: float, end_ts: float) -> None:
    duration = end_ts - start_ts
    records.append(
        {
            "step": step_name,
            "start": start_ts,
            "end": end_ts,
            "duration": duration,
        }
    )
    print(
        f"[TIMING] {step_name} finished at {_format_timestamp(end_ts)} "
        f"(duration: {_format_duration(duration)})"
    )


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
    }


_CFUSE_TOKEN_PATTERN = re.compile(
    r"Completion finished: Tokens\(prompt=(\d+), completion=(\d+), total=(\d+)\)"
)


def _merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    merged = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for usage in usages:
        if not usage:
            continue
        for key in merged:
            merged[key] += int(usage.get(key, 0) or 0)
    return merged


def _collect_cfuse_usage(*base_dirs: str | None) -> dict[str, int]:
    usage = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    seen_paths: set[str] = set()
    for base_dir in base_dirs:
        if not base_dir:
            continue
        abs_base_dir = os.path.abspath(base_dir)
        if not os.path.isdir(abs_base_dir):
            continue
        for root, _, files in os.walk(abs_base_dir):
            for file_name in files:
                if file_name != "main.log":
                    continue
                log_path = os.path.join(root, file_name)
                if log_path in seen_paths:
                    continue
                seen_paths.add(log_path)
                try:
                    with open(log_path, "r", encoding="utf-8") as infile:
                        for line in infile:
                            match = _CFUSE_TOKEN_PATTERN.search(line)
                            if not match:
                                continue
                            prompt_tokens = int(match.group(1))
                            completion_tokens = int(match.group(2))
                            total_tokens = int(match.group(3))
                            usage["requests"] += 1
                            usage["prompt_tokens"] += prompt_tokens
                            usage["completion_tokens"] += completion_tokens
                            usage["total_tokens"] += total_tokens
                except OSError:
                    continue
    return usage


def _record_step_metrics(
    records: list[dict[str, Any]],
    step_name: str,
    start_ts: float,
    end_ts: float,
    usage_before: dict[str, int] | None,
    usage_after: dict[str, int] | None,
    token_status: str = "available",
    usage_override: dict[str, int] | None = None,
) -> None:
    duration = end_ts - start_ts
    if usage_override is not None:
        usage = {
            key: int(usage_override.get(key, 0) or 0)
            for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
        }
    else:
        usage = _usage_delta(usage_before or {}, usage_after or {})
    records.append(
        {
            "step": step_name,
            "start": start_ts,
            "end": end_ts,
            "duration": duration,
            "usage": usage,
            "token_status": token_status,
        }
    )
    usage_text = (
        f"requests={usage['requests']}, prompt={usage['prompt_tokens']}, "
        f"completion={usage['completion_tokens']}, total={usage['total_tokens']}"
        if token_status == "available"
        else "token_usage=unavailable"
    )
    print(
        f"[TIMING] {step_name} finished at {_format_timestamp(end_ts)} "
        f"(duration: {_format_duration(duration)}; {usage_text})"
    )


def _print_timing_summary(records: list[dict[str, Any]], total_start_ts: float) -> None:
    _ = records
    _ = total_start_ts


def _build_single_instance_input(
    input_data_path: str,
    output_dir: str,
    instance_id: str,
) -> str:
    with open(input_data_path, "r", encoding="utf-8") as infile:
        loaded = json.load(infile)

    target = str(instance_id or "").strip()
    if not target:
        raise ValueError("empty instance_id")

    filtered: Dict[str, Any] = {}
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
    return output_path


def run_gala(
        input_data: str,
        output_dir: str,
        repo_path: str,
        project_name: str = "gala",
        vlm_model: str = "Qwen3-VL-235B-A22B-Instruct",
        vlm_url: str = "https://xxx/v1",
        image_dir: str = None,
        model_name: str = "Kimi-K2-Instruct-0905",
        base_url: str = "https://xxx/v1",
        temperature: float = 0,
        copy_repo: bool = False,
        max_workers: int = 4,
        text_model_name: str = "",
        text_base_url: str = "",
        text_api_key: str = "",
        checkout_base_commit: bool = True,
):
    timing_records: list[dict[str, Any]] = []
    total_start_ts = time.time()
    reset_usage_totals()

    # step 1. generate image graph
    image_ir_path = os.path.join(output_dir, "image_ir_data.json")
    if os.path.exists(image_ir_path):
        print(f"Reuse existing image graph: {image_ir_path}")
    else:
        step_start_ts = time.time()
        usage_before = get_usage_totals()
        image_ir = GenerateIR(vlm_model, vlm_url)
        image_ir.process_batch(output_dir, input_data, output_dir, image_dir, max_workers=max_workers)
        print(f"Image graph generated: {image_ir_path}")
        _record_step_metrics(timing_records, "generate-image-graph", step_start_ts, time.time(), usage_before, get_usage_totals())

    # step 2. build repo structure snapshot
    step_start_ts = time.time()
    usage_before = get_usage_totals()
    localizer = ImageCodeLocalization(
        vlm_model,
        vlm_url,
        repo_path=repo_path,
        checkout_base_commit=checkout_base_commit,
        text_model_name=text_model_name or model_name,
        text_base_url=text_base_url or base_url,
        text_api_key=text_api_key,
    )
    localizer.process_batch_build_code_graph(image_ir_path, output_dir, image_dir, max_workers=max_workers)
    _record_step_metrics(timing_records, "build-code-graph", step_start_ts, time.time(), usage_before, get_usage_totals())

    # step 3. align code graph (before patch generation)
    step_start_ts = time.time()
    usage_before = get_usage_totals()
    localizer.process_batch_align_code_graph(image_ir_path, output_dir, image_dir, max_workers=max_workers)
    _record_step_metrics(timing_records, "align-code-graph", step_start_ts, time.time(), usage_before, get_usage_totals())

    # step 4. generate patch
    step_start_ts = time.time()
    usage_before = get_usage_totals()
    cfuse_usage_before = _collect_cfuse_usage(output_dir)
    patch_gen = PatchGeneration(model_name, base_url, max_workers, temperature)
    patch_gen.process_batch(image_ir_path, output_dir, repo_path=repo_path, copy_repo=copy_repo)
    print(f"Patches generated: {output_dir}")
    combined_usage = _merge_usage(
        _usage_delta(usage_before, get_usage_totals()),
        _usage_delta(cfuse_usage_before, _collect_cfuse_usage(output_dir)),
    )
    _record_step_metrics(
        timing_records,
        "generate-patch",
        step_start_ts,
        time.time(),
        None,
        None,
        usage_override=combined_usage,
    )
    # Export a first-round, testable patch package before validation/redo.
    process_git_diff(output_dir, project_name, result_tag="round1")

    # step 5. validation
    step_start_ts = time.time()
    usage_before = get_usage_totals()
    cfuse_usage_before = _collect_cfuse_usage(output_dir)
    validation = Validation()
    failed_path, _ = validation.filtering_result(image_ir_path, output_dir)
    agent_val = AgentBaseValidation(repo_path=repo_path, model_name=model_name, base_url=base_url, max_workers=max_workers)
    agent_val.process_batch(failed_path, output_dir, repo_path=repo_path, copy_repo=copy_repo)
    process_val(output_dir)
    combined_usage = _merge_usage(
        _usage_delta(usage_before, get_usage_totals()),
        _usage_delta(cfuse_usage_before, _collect_cfuse_usage(output_dir)),
    )
    _record_step_metrics(
        timing_records,
        "validation",
        step_start_ts,
        time.time(),
        None,
        None,
        usage_override=combined_usage,
    )

    # step 6. retry patch generation for validation-failed subset
    step_start_ts = time.time()
    usage_before = get_usage_totals()
    cfuse_usage_before = _collect_cfuse_usage(output_dir)
    failed_data_path = os.path.join(output_dir, "all_validation_failed_instance.json")
    failed_instances: Dict[str, Any] = {}
    if os.path.exists(failed_data_path):
        try:
            with open(failed_data_path, "r", encoding="utf-8") as infile:
                loaded_failed = json.load(infile)
            if isinstance(loaded_failed, dict):
                failed_instances = loaded_failed
        except Exception as exc:
            print(f"WARNING: failed to parse failed-instance file: {exc}")

    if failed_instances:
        # Reuse validation-failed subset directly.
        # Redo only patch generation, and reuse existing localization artifacts.
        redo_ir_path = failed_data_path
        patch_gen.process_batch(redo_ir_path, output_dir, repo_path=repo_path, copy_repo=copy_repo)
        print(f"Redo patches generated for {len(failed_instances)} failed instances")
    else:
        print("No validation-failed instances detected; skip redo patch generation")
    combined_usage = _merge_usage(
        _usage_delta(usage_before, get_usage_totals()),
        _usage_delta(cfuse_usage_before, _collect_cfuse_usage(output_dir)),
    )
    _record_step_metrics(
        timing_records,
        "redo-after-validation",
        step_start_ts,
        time.time(),
        None,
        None,
        usage_override=combined_usage,
    )

    # step 7. process result
    step_start_ts = time.time()
    usage_before = get_usage_totals()
    process_git_diff(output_dir, project_name, result_tag="final")
    _record_step_metrics(timing_records, "process-result", step_start_ts, time.time(), usage_before, get_usage_totals())
    _print_timing_summary(timing_records, total_start_ts)


def main():
    parser = argparse.ArgumentParser(description="GALA command line tool")
    subparsers = parser.add_subparsers(dest='cmd', help='Available commands')

    def add_generate_image_graph_args(cmd_parser: argparse.ArgumentParser) -> None:
        cmd_parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", required=True, help="Model name")
        cmd_parser.add_argument("--base_url", required=True, help="Base URL for API")
        cmd_parser.add_argument("--input_data", required=True, help='Path to input data')
        cmd_parser.add_argument("--image_dir", required=True, help='Path to image directory')
        cmd_parser.add_argument("--result_path", required=True, help='Path to result data')
        cmd_parser.add_argument("--output_dir", required=True, help='Output directory')
        cmd_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')

    # generate-image-graph command
    gen_graph_parser = subparsers.add_parser('generate-image-graph', help='Generate image graph')
    add_generate_image_graph_args(gen_graph_parser)

    # generate-image-ir command (backward compatibility alias)
    gen_ir_parser = subparsers.add_parser('generate-image-ir', help='Generate image graph (compat alias)')
    add_generate_image_graph_args(gen_ir_parser)
    
    # generate-patch command
    gen_patch_parser = subparsers.add_parser('generate-patch', help='Generate patches')
    gen_patch_parser.add_argument("--image_ir_path", required=True, help='Path to image IR data')
    gen_patch_parser.add_argument("--temperature", default=0, type=float)
    gen_patch_parser.add_argument("--output_dir", required=True, help='Output directory')
    gen_patch_parser.add_argument("--repo_path", required=True, help='Repository path')
    gen_patch_parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", help='Model name')
    gen_patch_parser.add_argument("--base_url", required=True, help="Base URL for API")
    gen_patch_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')
    gen_patch_parser.add_argument("--copy_repo", action="store_true", help='Copy repository')
    
    # validation command
    validation_parser = subparsers.add_parser('validation', help='Validate patches')
    validation_parser.add_argument("--image_ir_path", required=True, help='Path to image IR data')
    validation_parser.add_argument("--result_path", required=True, help='Path to result data')
    validation_parser.add_argument("--output_dir", required=True, help='Output directory')
    validation_parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", help='Model name')
    validation_parser.add_argument("--base_url", required=True, help="Base URL for API")
    validation_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')
    validation_parser.add_argument("--repo_path", type=str, default="data/swe_bench_mm/repos", help="Repository base path")
    validation_parser.add_argument("--copy_repo", action="store_true", help='Copy repository')

    def add_code_graph_stage_args(cmd_parser: argparse.ArgumentParser) -> None:
        cmd_parser.add_argument("--repo_path", required=True, help='Repository path')
        cmd_parser.add_argument("--image_dir", required=True)
        cmd_parser.add_argument("--output_dir", required=True, help='Output directory')
        cmd_parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", help='Model name')
        cmd_parser.add_argument("--base_url", required=True, help="Base URL for API")
        cmd_parser.add_argument("--result_path", required=True)
        cmd_parser.add_argument("--input_data", help="Path to full input dataset json (default: result_path/image_ir_data.json)")
        cmd_parser.add_argument("--no_checkout_base_commit", action="store_true", help="Disable checkout to base_commit before code graph parsing")
        cmd_parser.add_argument("--text_model_name", help="Text model used for CoSIL-style file localization in code graph")
        cmd_parser.add_argument("--text_base_url", help="Base URL for text model API")
        cmd_parser.add_argument("--text_api_key", help="Text model API key (optional; prefer env TEXT_API_KEY)")
        cmd_parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild outputs for the selected stage")

    build_code_graph_parser = subparsers.add_parser('build-code-graph', help='Build repo snapshot and select candidate/seed files')
    add_code_graph_stage_args(build_code_graph_parser)

    align_code_graph_parser = subparsers.add_parser('align-code-graph', help='Build final code graphs and alignment from existing seed-file artifacts')
    add_code_graph_stage_args(align_code_graph_parser)

    # redo-after-align-code-graph command
    redo_parser = subparsers.add_parser(
        'redo-after-align-code-graph',
        help='Rerun patch generation for validation-failed instances (reuse existing code-graph artifacts)'
    )
    redo_parser.add_argument("--result_path", required=True, help='Directory containing previous run outputs')
    redo_parser.add_argument("--output_dir", required=True, help='Output directory')
    redo_parser.add_argument("--repo_path", required=True, help='Repository path')
    redo_parser.add_argument("--image_dir", required=True, help='Path to image directory')
    redo_parser.add_argument("--vlm_model", default="Qwen3-VL-235B-A22B-Instruct", help='VLM model name')
    redo_parser.add_argument("--vlm_url", required=True, help="Base URL for VLM API")
    redo_parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", help='Patch model name')
    redo_parser.add_argument("--base_url", required=True, help="Base URL for patch API")
    redo_parser.add_argument("--temperature", default=0, type=float, help='Temperature for model')
    redo_parser.add_argument("--copy_repo", action="store_true", help='Copy repository')
    redo_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')

    # process-result command
    process_result_parser = subparsers.add_parser('process-result', help='Process final patch results')
    process_result_parser.add_argument("--result_path", required=True, help='Result directory')
    process_result_parser.add_argument("--project_name", type=str, default="gala", help='Project name')
    process_result_parser.add_argument("--result_tag", type=str, default="", help='Optional result filename tag, e.g. round1/final')
    
    # full-run command (original complete pipeline)
    full_run_parser = subparsers.add_parser('full-run', help='Run complete GALA pipeline')
    full_run_parser.add_argument("--input_data", required=True, help='Path to input data')
    full_run_parser.add_argument("--output_dir", required=True, help='Output directory')
    full_run_parser.add_argument("--repo_path", required=True, help='Repository path')
    full_run_parser.add_argument("--image_dir", required=True, help='Path to image directory')
    full_run_parser.add_argument("--vlm_model", default="Qwen3-VL-235B-A22B-Instruct", help='Model name')
    full_run_parser.add_argument("--vlm_url", required=True, help="Base URL for API")
    full_run_parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", help='Model name')
    full_run_parser.add_argument("--base_url", required=True, help="Base URL for API")
    full_run_parser.add_argument("--temperature", default=0, type=float, help='Temperature for model')
    full_run_parser.add_argument("--copy_repo", action="store_true", help='Copy repository')
    full_run_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')
    full_run_parser.add_argument("--project_name", type=str, default="gala", help='Project name')
    full_run_parser.add_argument("--text_model_name", help="Text model used for CoSIL-style file localization in code graph")
    full_run_parser.add_argument("--text_base_url", help="Base URL for text model API")
    full_run_parser.add_argument("--text_api_key", help="Text model API key (optional; prefer env TEXT_API_KEY)")
    full_run_parser.add_argument("--no_checkout_base_commit", action="store_true", help="Disable checkout to base_commit before code graph parsing")
    full_run_parser.add_argument("--instance_id", help="Run pipeline for one instance only (for smoke test)")
    
    args = parser.parse_args()
    timing_records: list[dict[str, Any]] = []
    total_start_ts = time.time()
    reset_usage_totals()
    
    if args.cmd is None:
        parser.print_help()
        return
    
    if hasattr(args, "output_dir") and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.cmd in {'generate-image-graph', 'generate-image-ir'}:
        step_start_ts = time.time()
        usage_before = get_usage_totals()
        image_ir = GenerateIR(args.model_name, args.base_url)
        image_ir.process_batch(args.result_path, args.input_data, args.output_dir, args.image_dir, max_workers=args.max_workers)
        print(f"Image graph generated: {os.path.join(args.output_dir, 'image_ir_data.json')}")
        _record_step_metrics(timing_records, args.cmd, step_start_ts, time.time(), usage_before, get_usage_totals())
    
    elif args.cmd == 'generate-patch':
        step_start_ts = time.time()
        usage_before = get_usage_totals()
        cfuse_usage_before = _collect_cfuse_usage(args.output_dir)
        patch_gen = PatchGeneration(args.model_name, args.base_url, args.max_workers, args.temperature)
        patch_gen.process_batch(args.image_ir_path, args.output_dir, repo_path=args.repo_path, copy_repo=args.copy_repo)
        print(f"Patches generated: {args.output_dir}")
        combined_usage = _merge_usage(
            _usage_delta(usage_before, get_usage_totals()),
            _usage_delta(cfuse_usage_before, _collect_cfuse_usage(args.output_dir)),
        )
        _record_step_metrics(
            timing_records,
            "generate-patch",
            step_start_ts,
            time.time(),
            None,
            None,
            usage_override=combined_usage,
        )

    elif args.cmd == 'validation':
        step_start_ts = time.time()
        usage_before = get_usage_totals()
        cfuse_usage_before = _collect_cfuse_usage(args.result_path)
        validation = Validation()
        failed_path, _ = validation.filtering_result(args.image_ir_path, args.result_path)
        agent_val = AgentBaseValidation(repo_path=args.repo_path, model_name=args.model_name, base_url=args.base_url, max_workers=args.max_workers)
        agent_val.process_batch(failed_path, args.result_path, args.repo_path, copy_repo=args.copy_repo)

        process_val(args.result_path)
        combined_usage = _merge_usage(
            _usage_delta(usage_before, get_usage_totals()),
            _usage_delta(cfuse_usage_before, _collect_cfuse_usage(args.result_path)),
        )
        _record_step_metrics(
            timing_records,
            "validation",
            step_start_ts,
            time.time(),
            None,
            None,
            usage_override=combined_usage,
        )
    
    elif args.cmd in {'build-code-graph', 'align-code-graph'}:
        step_start_ts = time.time()
        usage_before = get_usage_totals()
        localizer = ImageCodeLocalization(
            args.model_name,
            args.base_url,
            repo_path=args.repo_path,
            checkout_base_commit=not args.no_checkout_base_commit,
            text_model_name=args.text_model_name or "",
            text_base_url=args.text_base_url or "",
            text_api_key=args.text_api_key or "",
        )
        all_input_path = args.input_data or os.path.join(args.result_path, "image_ir_data.json")
        if not os.path.exists(all_input_path):
            print(f"WARNING: all input data not found at {all_input_path}, fallback to all_validation_failed_instance.json")
            all_input_path = os.path.join(args.result_path, "all_validation_failed_instance.json")
        if args.cmd == 'build-code-graph':
            localizer.process_batch_build_code_graph(
                all_input_path,
                args.result_path,
                args.image_dir,
                force_rebuild=args.force_rebuild,
            )
        else:
            localizer.process_batch_align_code_graph(
                all_input_path,
                args.result_path,
                args.image_dir,
                force_rebuild=args.force_rebuild,
            )
        _record_step_metrics(timing_records, args.cmd, step_start_ts, time.time(), usage_before, get_usage_totals())

    elif args.cmd == 'redo-after-align-code-graph':
        step_start_ts = time.time()
        usage_before = get_usage_totals()
        cfuse_usage_before = _collect_cfuse_usage(args.result_path, args.output_dir)
        failed_data_path = os.path.join(args.result_path, "all_validation_failed_instance.json")
        if not os.path.exists(failed_data_path):
            print(f"WARNING: failed-instance file not found at {failed_data_path}")
            return 1
        try:
            with open(failed_data_path, "r", encoding="utf-8") as infile:
                loaded_failed = json.load(infile)
        except Exception as exc:
            print(f"WARNING: failed to parse failed-instance file: {exc}")
            return 1
        if not isinstance(loaded_failed, dict) or not loaded_failed:
            print("No validation-failed instances detected; skip redo-after-align-code-graph")
            return 0
        redo_ir_path = failed_data_path
        print(f"Reuse failed-instance data directly: {redo_ir_path}")

        patch_gen = PatchGeneration(args.model_name, args.base_url, args.max_workers, args.temperature)
        patch_gen.process_batch(redo_ir_path, args.output_dir, repo_path=args.repo_path, copy_repo=args.copy_repo)
        print(f"Patches generated again: {args.output_dir}")
        combined_usage = _merge_usage(
            _usage_delta(usage_before, get_usage_totals()),
            _usage_delta(cfuse_usage_before, _collect_cfuse_usage(args.result_path, args.output_dir)),
        )
        _record_step_metrics(
            timing_records,
            "redo-after-align-code-graph",
            step_start_ts,
            time.time(),
            None,
            None,
            usage_override=combined_usage,
        )

    elif args.cmd == 'process-result':
        step_start_ts = time.time()
        usage_before = get_usage_totals()
        process_git_diff(args.result_path, args.project_name, result_tag=args.result_tag)
        _record_step_metrics(timing_records, "process-result", step_start_ts, time.time(), usage_before, get_usage_totals())

    elif args.cmd == 'full-run':
        run_input_data = args.input_data
        if args.instance_id:
            try:
                run_input_data = _build_single_instance_input(
                    input_data_path=args.input_data,
                    output_dir=args.output_dir,
                    instance_id=args.instance_id,
                )
                print(f"Single-instance mode enabled: {args.instance_id}")
                print(f"Filtered input saved to: {run_input_data}")
            except Exception as exc:
                print(f"Failed to prepare single-instance input: {exc}")
                return 1
        run_gala(
            input_data=run_input_data,
            output_dir=args.output_dir,
            repo_path=args.repo_path,
            vlm_model=args.vlm_model,
            vlm_url=args.vlm_url,
            image_dir=args.image_dir,
            model_name=args.model_name,
            base_url=args.base_url,
            temperature=args.temperature,
            copy_repo=args.copy_repo,
            max_workers=args.max_workers,
            project_name=args.project_name,
            text_model_name=args.text_model_name or "",
            text_base_url=args.text_base_url or "",
            text_api_key=args.text_api_key or "",
            checkout_base_commit=not args.no_checkout_base_commit,
        )

    _print_timing_summary(timing_records, total_start_ts)
    return 0


if __name__ == "__main__":
    # Main execution code
    sys.exit(main())


