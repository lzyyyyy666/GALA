#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from typing import Any, Dict, Optional

from src.code_graph_builder import CodeGraphBuilder
from src.image_repo_localization import normalize_chat_completions_url
from src.utils.repo_resolver import resolve_repo_dir


DEFAULT_SOURCE_OUTPUT_DIR = "/nvme2/zzr/lzy/GALA/output/seedtest1"
DEFAULT_SOURCE_INPUT_DIR = "/nvme2/zzr/lzy/GALA/output/new_agent"
DEFAULT_IMAGE_IR_PATH = os.path.join(DEFAULT_SOURCE_INPUT_DIR, "image_ir_data.json")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_SOURCE_OUTPUT_DIR, "seed_files_eval.json")
DEFAULT_REPO_PATH = "/nvme2/zzr/lzy/swe-m/dev/repo"
DEFAULT_MODEL_NAME = "qwen3.5-35b-a3b"
DEFAULT_MODEL_BASE_URL = "http://100.69.30.24:46406/v1/"


def _first_image_graph(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_graphs = doc.get("image_graphs")
    if not isinstance(image_graphs, list):
        return None
    for item in image_graphs:
        if isinstance(item, dict) and item:
            return item
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate seed_files only from an existing image_ir_data.json dataset."
    )
    parser.add_argument("--image_ir_path", default=DEFAULT_IMAGE_IR_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--repo_path", default=DEFAULT_REPO_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base_url", default=DEFAULT_MODEL_BASE_URL)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--top_n_seed_files", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--no_checkout_base_commit", action="store_true")
    args = parser.parse_args()

    image_ir_path = os.path.abspath(args.image_ir_path)
    output_path = os.path.abspath(args.output_path)
    repo_path = os.path.abspath(args.repo_path)
    base_url = normalize_chat_completions_url(args.base_url)

    if not os.path.exists(image_ir_path):
        raise FileNotFoundError(f"image_ir_data not found: {image_ir_path}")

    with open(image_ir_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at {image_ir_path}, got {type(data).__name__}")

    builder = CodeGraphBuilder(
        model_name=args.model_name,
        base_url=base_url,
        api_key=args.api_key,
        top_n_seed_files=args.top_n_seed_files,
    )

    results: Dict[str, Any] = {}
    total = len(data)

    def _run_single(instance_id: str, raw_doc: Any) -> Dict[str, Any]:
        doc = raw_doc if isinstance(raw_doc, dict) else {}
        repo_identifier = str(doc.get("repo") or "").strip()
        base_commit = str(doc.get("base_commit") or "").strip()
        problem_statement = str(doc.get("problem_statement") or "")
        image_graph = _first_image_graph(doc)

        record: Dict[str, Any] = {
            "instance_id": instance_id,
            "seed_files": [],
        }

        if not repo_identifier:
            return {"record": record, "status": "missing_repo"}

        repo_dir = resolve_repo_dir(repo_path, repo_identifier)
        if not os.path.isdir(repo_dir):
            return {"record": record, "status": f"missing_repo_dir - {repo_dir}"}

        repo_dir_abs = os.path.abspath(repo_dir)
        cache_key = f"{repo_dir_abs}@@{base_commit if not args.no_checkout_base_commit else 'working_tree'}"

        try:
            snapshot = builder._get_or_build_snapshot(  # noqa: SLF001 - intentional internal reuse
                cache_key=cache_key,
                repo_dir=repo_dir_abs,
                base_commit=base_commit,
                checkout_base_commit=not args.no_checkout_base_commit,
            )
            available_files = snapshot["available_file_paths"]
            prompt_tree = snapshot["prompt_tree"]
            seed_files = builder._select_seed_files_with_llm(  # noqa: SLF001 - intentional internal reuse
                problem_statement=problem_statement,
                structure=prompt_tree,
                all_files=available_files,
                image_graph=image_graph,
            )
            record.update(
                {
                    "seed_files": seed_files,
                }
            )
            return {"record": record, "status": f"{len(seed_files)} seed files", "ok": True}
        except Exception as exc:  # pragma: no cover - best-effort batch script
            return {"record": record, "status": f"failed - {exc}"}

    success = 0
    processed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_instance = {
            executor.submit(_run_single, instance_id, raw_doc): instance_id
            for instance_id, raw_doc in data.items()
        }
        for future in as_completed(future_to_instance):
            instance_id = future_to_instance[future]
            processed += 1
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - best-effort batch script
                results[instance_id] = {"instance_id": instance_id, "seed_files": []}
                print(f"[{processed}/{total}] {instance_id}: failed - {exc}")
                continue
            results[instance_id] = result["record"]
            if result.get("ok"):
                success += 1
            print(f"[{processed}/{total}] {instance_id}: {result['status']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2, ensure_ascii=False)

    print(f"Saved seed file results to: {output_path}")
    print(f"Successful instances: {success}/{total}")


if __name__ == "__main__":
    main()
