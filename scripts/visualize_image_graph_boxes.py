import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

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


def _normalize_bbox(raw_bbox: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)) or len(raw_bbox) != 4:
        return None

    values: List[int] = []
    for value in raw_bbox:
        try:
            values.append(int(round(float(value))))
        except (TypeError, ValueError):
            return None

    x1, y1, x2, y2 = values
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        if width <= 1:
            return None
        if x1 >= width - 1:
            x1 = max(0, width - 2)
            x2 = width - 1
        else:
            x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        if height <= 1:
            return None
        if y1 >= height - 1:
            y1 = max(0, height - 2)
            y2 = height - 1
        else:
            y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def _draw_boxes(image_path: str, rooted_graph: Dict[str, Any], output_path: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    root_ids = {
        str(item.get("id") or "").strip()
        for item in rooted_graph.get("root_objects", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    n_root_boxes = 0
    n_other_boxes = 0
    skipped_objects: List[Dict[str, Any]] = []
    for obj in rooted_graph.get("objects", []):
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("id") or "").strip()
        bbox = _normalize_bbox(obj.get("bbox"), width=width, height=height)
        if not object_id or bbox is None:
            skipped_objects.append(
                {
                    "id": object_id,
                    "type": str(obj.get("type") or "").strip(),
                    "text": str(obj.get("text") or "").strip(),
                    "bbox": obj.get("bbox"),
                }
            )
            continue

        is_root = object_id in root_ids
        color = "red" if is_root else "blue"
        draw.rectangle(bbox, outline=color, width=4)
        if is_root:
            n_root_boxes += 1
        else:
            n_other_boxes += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    return n_root_boxes, n_other_boxes, skipped_objects


def _clean_issue_text(issue_text: str) -> str:
    lines = issue_text.splitlines()
    cleaned_lines: List[str] = []
    image_line_pattern = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
    for line in lines:
        if image_line_pattern.match(line):
            continue
        if line.strip() == "Here's an example where it is implemented as a `Select` component:":
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    return cleaned_text


def _visualize_single_instance(
    pipeline: TypeAwareImageGraphPipeline,
    instance_id: str,
    doc: Dict[str, Any],
    image_dir: str,
    output_dir: str,
) -> None:
    issue_text = str(doc.get("problem_statement") or "")
    cleaned_issue_text = _clean_issue_text(issue_text)
    image_urls = _parse_image_assets(doc)

    if not image_urls:
        print(f"instance_id={instance_id} has no problem_statement images")
        return

    instance_output_dir = os.path.join(output_dir, instance_id)
    os.makedirs(instance_output_dir, exist_ok=True)

    for image_idx, image_url in enumerate(image_urls):
        image_data_url = process_image_url_to_base64(image_url, image_dir=image_dir, instance_id=instance_id)
        if image_data_url is None:
            print(f"instance_id={instance_id} image_index={image_idx} skipped: failed to load image data")
            continue

        image_path = resolve_local_image_path(image_url, image_dir=image_dir, instance_id=instance_id)
        graph = pipeline.run_single(
            image_url=image_data_url,
            issue_text=issue_text,
            image_path=image_path,
        )
        rooted_graph = graph.get("rooted_graph", {})
        root_count = len(rooted_graph.get("root_objects", [])) if isinstance(rooted_graph.get("root_objects", []), list) else 0
        object_count = len(rooted_graph.get("objects", [])) if isinstance(rooted_graph.get("objects", []), list) else 0
        output_filename = f"{image_idx:02d}_{os.path.basename(image_path)}"
        output_path = os.path.join(instance_output_dir, output_filename)
        json_output_path = os.path.join(
            instance_output_dir,
            f"{image_idx:02d}_{os.path.splitext(os.path.basename(image_path))[0]}_rooted_graph.json",
        )
        n_root_boxes, n_other_boxes, skipped_objects = _draw_boxes(
            image_path=image_path,
            rooted_graph=rooted_graph,
            output_path=output_path,
        )
        with open(json_output_path, "w", encoding="utf-8") as outfile:
            json.dump(
                {
                    "instance_id": instance_id,
                    "image_index": image_idx,
                    "image_path": image_path,
                    "image_type": rooted_graph.get("image_type", graph.get("image_type", "generic_diagram")),
                    "graph_type": graph.get("graph_type", "generic_scene_graph"),
                    "root_objects": rooted_graph.get("root_objects", []),
                    "objects": rooted_graph.get("objects", []),
                    "relations": rooted_graph.get("relations", []),
                },
                outfile,
                indent=2,
                ensure_ascii=False,
            )

        print(f"image_name={os.path.basename(image_path)}")
        print(f"graph_json={json_output_path}")
        print(f"root_objects={root_count}")
        print(f"objects={object_count}")
        print(f"drawn_boxes={n_root_boxes + n_other_boxes}")
        print(f"skipped_objects={len(skipped_objects)}")
        if skipped_objects:
            print("skipped_object_details:")
            for skipped in skipped_objects:
                print(
                    f"- id={skipped['id']} type={skipped['type']} "
                    f"text={skipped['text']} bbox={skipped['bbox']}"
                )
        print("issue_text:")
        print(cleaned_issue_text)
        print("")


def main():
    parser = argparse.ArgumentParser(description="Visualize rooted image-graph bounding boxes on source images")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory or dataset file")
    parser.add_argument("--output_dir", required=True, help="Directory to store annotated images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance_id", help="Visualize a single instance")
    group.add_argument("--all", action="store_true", help="Visualize all instances")

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
        _visualize_single_instance(
            pipeline=pipeline,
            instance_id=instance_id,
            doc=dataset_dict[instance_id],
            image_dir=image_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
