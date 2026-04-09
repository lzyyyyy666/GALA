import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.generate_image_ir import process_image_url_to_base64, resolve_local_image_path
from src.image_graph_pipeline import TypeAwareImageGraphPipeline


def _completion_url(base_url: str) -> str:
    return base_url + "chat/completions" if base_url.endswith("/") else base_url + "/chat/completions"


def _find_dataset_file(dataset_path: str) -> str:
    if os.path.isfile(dataset_path):
        return dataset_path

    preferred = [
        "output.json",
        "input.json",
        "swe_bench_mm_prompt_v3_dict.json",
        "swebench_mm_prompt_v3_dict.json",
        "dataset.json",
        "data.json",
    ]
    for name in preferred:
        candidate = os.path.join(dataset_path, name)
        if os.path.isfile(candidate):
            return candidate

    json_files: List[str] = []
    parquet_files: List[str] = []
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            full = os.path.join(root, filename)
            lowered = filename.lower()
            if lowered.endswith(".json"):
                json_files.append(full)
            elif lowered.endswith(".parquet"):
                parquet_files.append(full)

    if json_files:
        json_files.sort()
        return json_files[0]
    if parquet_files:
        parquet_files.sort()
        return parquet_files[0]

    raise FileNotFoundError(
        f"Cannot find dataset json/parquet under {dataset_path}. "
        "Please preprocess dataset first (e.g. process_data_dl_repo.py)."
    )


def _load_dataset_dict(dataset_file: str) -> Dict[str, Dict[str, Any]]:
    lowered = dataset_file.lower()
    if lowered.endswith(".json"):
        with open(dataset_file, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            converted: Dict[str, Dict[str, Any]] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                instance_id = str(item.get("instance_id") or "").strip()
                if instance_id:
                    converted[instance_id] = item
            return converted
        raise ValueError(f"Unsupported JSON structure in {dataset_file}")

    if lowered.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(dataset_file)
        converted: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            item = row.to_dict()
            instance_id = str(item.get("instance_id") or "").strip()
            if instance_id:
                converted[instance_id] = item
        return converted

    raise ValueError(f"Unsupported dataset file type: {dataset_file}")


def _parse_image_assets(doc: Dict[str, Any]) -> List[str]:
    raw_assets = doc.get("image_assets", "{}")
    try:
        assets = json.loads(raw_assets) if isinstance(raw_assets, str) else raw_assets
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(assets, dict):
        return []
    images = assets.get("problem_statement", [])
    if not isinstance(images, list):
        return []
    return [img for img in images if isinstance(img, str) and img]


def _resolve_image_dir(dataset_path: str, explicit_image_dir: str = "") -> str:
    if explicit_image_dir:
        return explicit_image_dir

    base_dir = dataset_path if os.path.isdir(dataset_path) else os.path.dirname(os.path.abspath(dataset_path))
    candidates = [
        os.path.join(base_dir, "images"),
        os.path.join(base_dir, "image"),
        os.path.join(base_dir, "img"),
        base_dir,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return base_dir


def _run_single_instance(
    pipeline: TypeAwareImageGraphPipeline,
    instance_id: str,
    doc: Dict[str, Any],
    image_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], int, int, int, int, int, int]:
    issue_text = str(doc.get("problem_statement") or "")
    image_urls = _parse_image_assets(doc)

    image_graphs: List[Dict[str, Any]] = []
    rooted_image_graphs: List[Dict[str, Any]] = []
    total_nodes = 0
    total_edges = 0
    total_roots = 0
    total_objects = 0
    total_relations = 0

    for image_url in image_urls:
        image_data_url = process_image_url_to_base64(image_url, image_dir=image_dir, instance_id=instance_id)
        if image_data_url is None:
            continue

        image_path = resolve_local_image_path(image_url, image_dir=image_dir, instance_id=instance_id)
        graph = pipeline.run_single(
            image_url=image_data_url,
            issue_text=issue_text,
            image_path=image_path,
        )
        image_graphs.append(
            {
                "image_path": graph.get("image_path", image_path),
                "image_type": graph.get("image_type", "generic_diagram"),
                "graph_type": graph.get("graph_type", "generic_scene_graph"),
                "nodes": graph.get("nodes", []),
                "edges": graph.get("edges", []),
            }
        )
        rooted_graph = graph.get("rooted_graph", {})
        rooted_output_graph = {
            "image_path": graph.get("image_path", image_path),
            "image_type": rooted_graph.get("image_type", graph.get("image_type", "generic_diagram")),
            "graph_type": graph.get("graph_type", "generic_scene_graph"),
            "root_objects": rooted_graph.get("root_objects", []),
            "objects": rooted_graph.get("objects", []),
            "relations": rooted_graph.get("relations", []),
        }
        rooted_image_graphs.append(rooted_output_graph)
        total_nodes += len(graph.get("nodes", []))
        total_edges += len(graph.get("edges", []))
        total_roots += len(rooted_output_graph.get("root_objects", []))
        total_objects += len(rooted_output_graph.get("objects", []))
        total_relations += len(rooted_output_graph.get("relations", []))

    return (
        {"instance_id": instance_id, "image_graphs": image_graphs},
        {"instance_id": instance_id, "image_graphs": rooted_image_graphs},
        len(image_graphs),
        total_nodes,
        total_edges,
        total_roots,
        total_objects,
        total_relations,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Image Graph JSON from SWE-bench Multimodal dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory or dataset file")
    parser.add_argument("--output_dir", required=True, help="Directory to store generated graph JSON files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance_id", help="Generate graph for a single instance")
    group.add_argument("--all", action="store_true", help="Generate graphs for all instances")

    parser.add_argument("--image_dir", default="", help="Optional image directory override")
    parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", help="VLM model name")
    parser.add_argument("--base_url", default=os.getenv("VLM_BASE_URL", "https://xxx/v1"), help="VLM base URL")
    args = parser.parse_args()

    dataset_file = _find_dataset_file(args.dataset_path)
    dataset_dict = _load_dataset_dict(dataset_file)
    image_dir = _resolve_image_dir(args.dataset_path, explicit_image_dir=args.image_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = TypeAwareImageGraphPipeline(
        model_name=args.model_name,
        base_url=_completion_url(args.base_url),
        api_key=os.getenv("VLM_API_KEY", ""),
    )

    if args.instance_id:
        if args.instance_id not in dataset_dict:
            raise KeyError(f"instance_id not found in dataset: {args.instance_id}")
        instance_ids = [args.instance_id]
    else:
        instance_ids = sorted(dataset_dict.keys())

    for instance_id in instance_ids:
        doc = dataset_dict[instance_id]
        (
            output_doc,
            rooted_output_doc,
            n_images,
            n_nodes,
            n_edges,
            n_roots,
            n_objects,
            n_relations,
        ) = _run_single_instance(
            pipeline=pipeline,
            instance_id=instance_id,
            doc=doc,
            image_dir=image_dir,
        )

        output_path = os.path.join(args.output_dir, f"{instance_id}.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(output_doc, outfile, indent=2, ensure_ascii=False)

        rooted_output_path = os.path.join(args.output_dir, f"{instance_id}_rooted_graph.json")
        with open(rooted_output_path, "w", encoding="utf-8") as outfile:
            json.dump(rooted_output_doc, outfile, indent=2, ensure_ascii=False)

        print(
            f"instance_id={instance_id} "
            f"number_of_images={n_images} "
            f"number_of_nodes={n_nodes} "
            f"number_of_edges={n_edges} "
            f"number_of_roots={n_roots} "
            f"number_of_objects={n_objects} "
            f"number_of_relations={n_relations}"
        )


if __name__ == "__main__":
    main()
