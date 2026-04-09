import argparse
import glob
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from src.code_graph_builder import CodeGraphBuilder
from src.utils.repo_resolver import resolve_repo_dir


def normalize_chat_completions_url(base_url: str) -> str:
    stripped = (base_url or "").strip()
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/"):
        return stripped + "chat/completions"
    return stripped + "/chat/completions"


class ImageCodeLocalization:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        repo_path: str = "",
        checkout_base_commit: bool = True,
        text_model_name: str = "",
        text_base_url: str = "",
        text_api_key: str = "",
    ):
        self.repo_path = repo_path
        self.checkout_base_commit = checkout_base_commit
        self.vlm_model = model_name
        self.vlm_base_url = normalize_chat_completions_url(base_url)
        self.vlm_api_key = os.getenv("VLM_API_KEY", "")

        self.text_model_name = text_model_name or os.getenv("TEXT_MODEL_NAME", "") or self.vlm_model
        self.text_base_url = normalize_chat_completions_url(text_base_url) if text_base_url else self.vlm_base_url
        self.text_api_key = text_api_key or os.getenv("TEXT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or self.vlm_api_key
        self.code_graph_builder = CodeGraphBuilder(
            model_name=self.text_model_name,
            base_url=self.text_base_url,
            api_key=self.text_api_key,
            top_n_candidate_files=10,
            top_n_seed_files=5,
        )
    @staticmethod
    def find_existing_code_graph(instance_dir: str, instance_id: str) -> Optional[str]:
        graph_path = os.path.join(instance_dir, f"code_graph_{instance_id}.json")
        if os.path.exists(graph_path):
            return graph_path
        return None

    @staticmethod
    def find_existing_code_graph_snapshot(instance_dir: str, instance_id: str) -> Optional[str]:
        snapshot_path = os.path.join(instance_dir, f"repo_structure_{instance_id}.json")
        if os.path.exists(snapshot_path):
            return snapshot_path
        return None

    @staticmethod
    def _load_instance_doc(instance_dir: str) -> Optional[Dict[str, Any]]:
        json_files = sorted(glob.glob(os.path.join(instance_dir, "*.json")))
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            if file_name.startswith("code_graph_") or file_name.startswith("subgraph_"):
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as infile:
                    loaded = json.load(infile)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                continue
        return None

    @staticmethod
    def _collect_image_graph_input(doc: Dict[str, Any]) -> Optional[list[Dict[str, Any]]]:
        graphs: list[Dict[str, Any]] = []
        image_graphs = doc.get("image_graphs")
        if isinstance(image_graphs, list):
            for item in image_graphs:
                if isinstance(item, dict) and item:
                    graphs.append(item)
        legacy_graph = doc.get("image_graph")
        if isinstance(legacy_graph, dict) and legacy_graph:
            graphs.append(legacy_graph)
        return graphs or None

    def build_and_save_code_graph(
        self,
        instance_id: str,
        instance_dir: str,
        doc: Dict[str, Any],
        snapshot_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.repo_path:
            return None

        repo_identifier = str(doc.get("repo") or "").strip()
        if not repo_identifier:
            print(f"Skip code graph for {instance_id}: missing repo in instance json")
            return None

        repo_dir = resolve_repo_dir(self.repo_path, repo_identifier)
        if not os.path.isdir(repo_dir):
            print(f"Skip code graph for {instance_id}: repo dir not found - {repo_dir}")
            return None

        base_commit = str(doc.get("base_commit") or "").strip()
        problem_statement = str(doc.get("problem_statement") or "")
        image_graph = self._collect_image_graph_input(doc)
        try:
            graph_payload = self.code_graph_builder.build_instance_graph(
                instance_id=instance_id,
                repo_identifier=repo_identifier,
                base_commit=base_commit,
                repo_dir=repo_dir,
                problem_statement=problem_statement,
                image_graph=image_graph,
                checkout_base_commit=self.checkout_base_commit,
                snapshot_payload=snapshot_payload,
            )
            out_path = os.path.join(instance_dir, f"code_graph_{instance_id}.json")
            with open(out_path, "w", encoding="utf-8") as outfile:
                json.dump(graph_payload, outfile, indent=4, ensure_ascii=False)
            code_graph = graph_payload.get("code_graph", {})
            seed_files = code_graph.get("seed_files", []) if isinstance(code_graph, dict) else []
            if isinstance(seed_files, list):
                print(f"seed files ({len(seed_files)}): {seed_files}")
            print(f"code graph saved: {out_path}")
            return out_path
        except Exception as exc:
            print(f"Failed to build code graph for {instance_id}: {exc}")
            traceback.print_exc()
            return None

    def build_and_save_code_graph_snapshot(
        self,
        instance_id: str,
        instance_dir: str,
        doc: Dict[str, Any],
    ) -> Optional[str]:
        if not self.repo_path:
            return None

        repo_identifier = str(doc.get("repo") or "").strip()
        if not repo_identifier:
            print(f"Skip repo structure for {instance_id}: missing repo in instance json")
            return None

        repo_dir = resolve_repo_dir(self.repo_path, repo_identifier)
        if not os.path.isdir(repo_dir):
            print(f"Skip repo structure for {instance_id}: repo dir not found - {repo_dir}")
            return None

        base_commit = str(doc.get("base_commit") or "").strip()
        problem_statement = str(doc.get("problem_statement") or "")
        image_graph = self._collect_image_graph_input(doc)
        try:
            snapshot_payload = self.code_graph_builder.build_instance_seed_stage(
                instance_id=instance_id,
                repo_identifier=repo_identifier,
                base_commit=base_commit,
                repo_dir=repo_dir,
                problem_statement=problem_statement,
                image_graph=image_graph,
                checkout_base_commit=self.checkout_base_commit,
            )
            out_path = os.path.join(instance_dir, f"repo_structure_{instance_id}.json")
            with open(out_path, "w", encoding="utf-8") as outfile:
                json.dump(snapshot_payload, outfile, indent=4, ensure_ascii=False)
            code_graph_snapshot = snapshot_payload.get("code_graph_snapshot", {})
            seed_files = code_graph_snapshot.get("seed_files", []) if isinstance(code_graph_snapshot, dict) else []
            if isinstance(seed_files, list):
                print(f"seed files ({len(seed_files)}): {seed_files}")
            print(f"repo structure saved: {out_path}")
            return out_path
        except Exception as exc:
            print(f"Failed to build repo structure for {instance_id}: {exc}")
            traceback.print_exc()
            return None

    @staticmethod
    def _load_snapshot_payload(instance_dir: str, instance_id: str) -> Optional[Dict[str, Any]]:
        snapshot_path = os.path.join(instance_dir, f"repo_structure_{instance_id}.json")
        if not os.path.isfile(snapshot_path):
            return None
        try:
            with open(snapshot_path, "r", encoding="utf-8") as infile:
                payload = json.load(infile)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    @staticmethod
    def _snapshot_has_seed_files(snapshot_payload: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(snapshot_payload, dict):
            return False
        code_graph_snapshot = snapshot_payload.get("code_graph_snapshot")
        if not isinstance(code_graph_snapshot, dict):
            return False
        seed_files = code_graph_snapshot.get("seed_files")
        return isinstance(seed_files, list) and len(seed_files) > 0

    def _collect_tasks(
        self,
        data_path: str,
        result_path: str,
    ) -> list[tuple[str, str, Dict[str, Any]]]:
        with open(data_path, "r", encoding="utf-8") as infile:
            data_dict = json.load(infile)
        if not isinstance(data_dict, dict):
            print(f"Skip localization stage: input data is not a dict - {data_path}")
            return []

        os.makedirs(result_path, exist_ok=True)
        tasks = []
        for instance_id, raw_doc in data_dict.items():
            instance_dir = os.path.join(result_path, instance_id)
            os.makedirs(instance_dir, exist_ok=True)

            merged_doc: Dict[str, Any] = {}
            disk_doc = self._load_instance_doc(instance_dir)
            if isinstance(disk_doc, dict):
                merged_doc.update(disk_doc)
            if isinstance(raw_doc, dict):
                merged_doc.update(raw_doc)
            merged_doc["instance_id"] = instance_id
            tasks.append((instance_id, instance_dir, merged_doc))
        return tasks

    def process_batch_build_code_graph(
        self,
        data_path: str,
        result_path: str,
        image_dir: str,
        max_workers: int = 4,
        force_rebuild: bool = False,
    ) -> None:
        _ = image_dir  # kept for CLI compatibility
        tasks = []
        for instance_id, instance_dir, merged_doc in self._collect_tasks(data_path, result_path):
            existing_snapshot = self.find_existing_code_graph_snapshot(instance_dir, instance_id)
            if existing_snapshot is not None and not force_rebuild:
                snapshot_payload = self._load_snapshot_payload(instance_dir, instance_id)
                if self._snapshot_has_seed_files(snapshot_payload):
                    print(f"existing repo structure found: {existing_snapshot}")
                    continue
            tasks.append((instance_id, instance_dir, merged_doc))

        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_instance = {
                executor.submit(self.build_and_save_code_graph_snapshot, instance_id, instance_dir, doc): (instance_id, instance_dir)
                for instance_id, instance_dir, doc in tasks
            }
            for future in as_completed(future_to_instance):
                instance_id, instance_dir = future_to_instance[future]
                try:
                    saved_path = future.result()
                    if saved_path:
                        success += 1
                        print(f"Successfully processed instance: {instance_id}")
                    else:
                        failed += 1
                        print(f"Skipped instance: {instance_id}")
                except Exception as exc:
                    failed += 1
                    print(f"Failed to process instance {instance_id}: {exc}")
                    traceback.print_exc()

        print(f"Total processed {len(tasks)} instances")
        print(f"Successfully built repo structures: {success}")
        print(f"Failed/Skipped: {failed}")

    def process_batch_align_code_graph(
        self,
        data_path: str,
        result_path: str,
        image_dir: str,
        max_workers: int = 4,
        force_rebuild: bool = False,
    ) -> None:
        _ = image_dir  # kept for CLI compatibility
        tasks = []
        for instance_id, instance_dir, merged_doc in self._collect_tasks(data_path, result_path):
            existing_code_graph = self.find_existing_code_graph(instance_dir, instance_id)
            if existing_code_graph is not None and not force_rebuild:
                print(f"existing code graph found: {existing_code_graph}")
                continue
            snapshot_payload = self._load_snapshot_payload(instance_dir, instance_id)
            tasks.append((instance_id, instance_dir, merged_doc, snapshot_payload))

        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_instance = {
                executor.submit(
                    self.build_and_save_code_graph,
                    instance_id,
                    instance_dir,
                    doc,
                    snapshot_payload,
                ): (instance_id, instance_dir)
                for instance_id, instance_dir, doc, snapshot_payload in tasks
            }
            for future in as_completed(future_to_instance):
                instance_id, instance_dir = future_to_instance[future]
                try:
                    saved_path = future.result()
                    if saved_path:
                        success += 1
                        print(f"Successfully processed instance: {instance_id}")
                    else:
                        failed += 1
                        print(f"Skipped instance: {instance_id}")
                except Exception as exc:
                    failed += 1
                    print(f"Failed to process instance {instance_id}: {exc}")
                    traceback.print_exc()

        print(f"Total processed {len(tasks)} instances")
        print(f"Successfully aligned code graph: {success}")
        print(f"Failed/Skipped: {failed}")

    def process_batch(
        self,
        data_path: str,
        result_path: str,
        image_dir: str,
        max_workers: int = 4,
        force_rebuild: bool = False,
    ) -> None:
        self.process_batch_build_code_graph(
            data_path=data_path,
            result_path=result_path,
            image_dir=image_dir,
            max_workers=max_workers,
            force_rebuild=force_rebuild,
        )
        self.process_batch_align_code_graph(
            data_path=data_path,
            result_path=result_path,
            image_dir=image_dir,
            max_workers=max_workers,
            force_rebuild=force_rebuild,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--repo_path", default="data/swe_bench_mm/repos")
    parser.add_argument("--input_data", help="Path to full input dataset json (default: result_path/image_ir_data.json)")
    parser.add_argument("--no_checkout_base_commit", action="store_true", help="Disable checkout to base_commit before code graph parsing")
    parser.add_argument("--text_model_name", help="Text model used for CoSIL-style file localization in code graph")
    parser.add_argument("--text_base_url", help="Base URL for text model API")
    parser.add_argument("--text_api_key", help="Text model API key (optional; prefer env TEXT_API_KEY)")
    parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", required=True, help="VLM")
    parser.add_argument("--base_url", required=True, help="Base URL for API")
    args = parser.parse_args()

    data_path = args.input_data or os.path.join(args.result_path, "image_ir_data.json")
    if not os.path.exists(data_path):
        print(f"WARNING: all input data not found at {data_path}, fallback to all_validation_failed_instance.json")
        data_path = os.path.join(args.result_path, "all_validation_failed_instance.json")

    localization = ImageCodeLocalization(
        args.model_name,
        args.base_url,
        repo_path=args.repo_path,
        checkout_base_commit=not args.no_checkout_base_commit,
        text_model_name=args.text_model_name or "",
        text_base_url=args.text_base_url or "",
        text_api_key=args.text_api_key or "",
    )
    localization.process_batch(data_path, args.result_path, args.image_dir)


if __name__ == "__main__":
    main()
