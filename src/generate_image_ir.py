import argparse
import base64
import io
import json
import mimetypes
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import tqdm
from PIL import Image, ImageFile, UnidentifiedImageError

from src.image_graph_pipeline import TypeAwareImageGraphPipeline


def resolve_local_image_path(img_url: str, image_dir: str = "", instance_id: str = "") -> str:
    parsed_url = urlparse(img_url)
    if parsed_url.scheme in ("http", "https", "ftp", "ftps"):
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"{instance_id}.jpg"
        return os.path.join(image_dir, f"{instance_id}_{filename}")
    return img_url


def _open_image_with_fallback(image_path: str) -> Image.Image:
    try:
        image = Image.open(image_path)
        image.load()
        return image
    except UnidentifiedImageError:
        original_flag = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            image = Image.open(image_path)
            image.load()
            return image
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = original_flag


def process_image_url_to_base64(img_url: str, image_dir: str = "", instance_id: str = "") -> Optional[str]:
    """Process image URL/local path and return a data URL. Return None if local file does not exist."""
    try:
        img_path = resolve_local_image_path(img_url, image_dir=image_dir, instance_id=instance_id)

        if not os.path.exists(img_path):
            print(f"Local file does not exist, skipping: {img_path}")
            return None

        is_gif = img_path.lower().endswith(".gif") or "gif" in img_path.lower()
        if is_gif:
            img = _open_image_with_fallback(img_path)
            img = img.convert("RGB")
            output_buffer = io.BytesIO()
            img.save(output_buffer, format="JPEG")
            img_str = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
            img.close()
            return f"data:image/jpeg;base64,{img_str}"

        with open(img_path, "rb") as img_file:
            img_data = img_file.read()
            img_str = base64.b64encode(img_data).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"
            return f"data:{mime_type};base64,{img_str}"
    except Exception as exc:
        print(f"Failed to process image URL: {img_url} - {exc}")
        return None


def _safe_parse_image_assets(data: Dict[str, Any]) -> List[str]:
    raw_assets = data.get("image_assets", "{}")
    try:
        assets = json.loads(raw_assets) if isinstance(raw_assets, str) else raw_assets
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(assets, dict):
        return []
    problem_images = assets.get("problem_statement", [])
    if not isinstance(problem_images, list):
        return []
    return [img for img in problem_images if isinstance(img, str) and img]


class GenerateIR:
    def __init__(self, model_name: str, base_url: str):
        self.vlm_model = model_name
        self.base_url = base_url + "chat/completions" if base_url.endswith("/") else base_url + "/chat/completions"
        self.api_key = os.getenv("VLM_API_KEY", "")

        self.image_graph_pipeline = TypeAwareImageGraphPipeline(
            model_name=self.vlm_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _generate_image_graph(self, image_data_url: str, issue_text: str, image_path: str) -> Dict[str, Any]:
        return self.image_graph_pipeline.run_single(
            image_url=image_data_url,
            issue_text=issue_text,
            image_path=image_path,
        )

    def _process_image_graph_batch(self, input_data: str, output_dir: str, image_dir: str, max_workers: int) -> None:
        with open(input_data, "r", encoding="utf-8") as infile:
            all_data = json.load(infile)

        tasks: List[Tuple[str, int, str, str, str]] = []
        skipped_count = 0
        for problem_id, data in all_data.items():
            image_list = _safe_parse_image_assets(data)
            for image_idx, img_url in enumerate(image_list):
                image_path = resolve_local_image_path(img_url, image_dir=image_dir, instance_id=problem_id)
                img_str = process_image_url_to_base64(img_url, image_dir=image_dir, instance_id=problem_id)
                if img_str is None:
                    skipped_count += 1
                    continue
                issue_text = str(data.get("problem_statement") or "")
                tasks.append((problem_id, image_idx, img_str, image_path, issue_text))

        if skipped_count > 0:
            print(f"Skipped {skipped_count} non-existent image files")
        print(f"Total images to process: {len(tasks)}")

        temp_results: Dict[str, List[Tuple[int, Any]]] = {}

        def process_single_image(task: Tuple[str, int, str, str, str]) -> Tuple[str, int, Any]:
            problem_id, image_idx, img_str, image_path, issue_text = task
            try:
                result = self._generate_image_graph(
                    image_data_url=img_str,
                    issue_text=issue_text,
                    image_path=image_path,
                )
                return problem_id, image_idx, result
            except Exception as exc:
                print(f"Error processing image for {problem_id}: {exc}")
                traceback.print_exc()
                return problem_id, image_idx, None

        bar = tqdm.tqdm(total=len(tasks), ncols=96)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_image, task) for task in tasks]
            for future in as_completed(futures):
                problem_id, image_idx, result = future.result()
                if result is not None:
                    temp_results.setdefault(problem_id, []).append((image_idx, result))
                bar.update()

        res_dict: Dict[str, Dict[str, Any]] = {}
        for problem_id, data in all_data.items():
            doc = dict(data)
            ordered_items = sorted(temp_results.get(problem_id, []), key=lambda pair: pair[0])

            image_graphs = [item for _, item in ordered_items]
            doc["image_graphs"] = image_graphs
            doc.pop("sub_image_graphs", None)
            doc.pop("image_graph_summary", None)
            doc.pop("image_caption", None)
            doc.pop("sub_graph_caption", None)
            res_dict[problem_id] = doc

        with open(os.path.join(output_dir, "image_ir_data.json"), "w", encoding="utf-8") as outf:
            outf.write(json.dumps(res_dict, indent=4, ensure_ascii=False))

    def process_batch(
        self,
        result_path: str,
        input_data: str,
        output_dir: str,
        image_dir: str,
        max_workers: int = 4,
    ) -> None:
        _ = result_path  # Kept for API compatibility; output is always written to output_dir.
        os.makedirs(output_dir, exist_ok=True)

        with open(input_data, "r", encoding="utf-8") as infile:
            loaded_data = json.load(infile)
        if isinstance(loaded_data, dict):
            print("total instance list: ", len(loaded_data))
        else:
            print("total instance list: unknown (input is not a dict)")

        self._process_image_graph_batch(
            input_data=input_data,
            output_dir=output_dir,
            image_dir=image_dir,
            max_workers=max_workers,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", required=True, help="VLM")
    parser.add_argument("--base_url", required=True, help="Base URL for API")
    parser.add_argument("--result_path")
    parser.add_argument("--max_workers", type=int, default=4, help="Max workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_ir = GenerateIR(args.model_name, args.base_url)
    generate_ir.process_batch(
        args.result_path,
        args.input_data,
        args.output_dir,
        args.image_dir,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
