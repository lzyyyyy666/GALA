#!/usr/bin/env python3
"""
SWE Bench MM Main Program

Refactored main program providing clear command-line interface and modular code structure.

Usage:
    python main.py --data_path <data_path> --output_dir <output_dir>

Args:
    data_path: Input JSON data file path
    output_dir: Output results directory
    model_name: Model name to use
    repo_path: Repository path

Example:
    python main.py --data_path data/test.json --output_dir results/
"""
import json
import os
import subprocess
import argparse
from typing import *
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import glob
import shutil
import time

from tqdm import tqdm
from jinja2 import Template

from prompt.image_prompt import SubGraphPrompt
from src.utils.logger import logger
from src.utils.config_manager import ConfigManager
from src.utils.repo_resolver import resolve_repo_dir
from src.run_cmd.run_cfuse import COMMAND

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')


class PatchGeneration:
    """Batch process multiple documents"""

    def __init__(self, model_name: str, base_url: str, max_workers: int = 4, temperature: float = 0.0):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = OPENAI_API_KEY
        self.max_workers = max_workers
        self.progress_lock = threading.Lock()
        self.temperature = temperature
        try:
            self.localization_retry = max(1, int(str(os.getenv("CFUSE_LOCALIZATION_MAX_RETRY", "2")).strip()))
        except Exception:
            self.localization_retry = 2

    @staticmethod
    def _normalize_rel_path(path: str) -> str:
        return str(path or "").replace("\\", "/").lstrip("./")

    @staticmethod
    def _load_code_graph(log_dir: str, instance_id: str) -> Dict[str, Any]:
        graph_path = os.path.join(log_dir, f"code_graph_{instance_id}.json")
        if not os.path.isfile(graph_path):
            return {}
        try:
            with open(graph_path, "r", encoding="utf-8") as infile:
                payload = json.load(infile)
            if not isinstance(payload, dict):
                return {}
            code_graph = payload.get("code_graph")
            return code_graph if isinstance(code_graph, dict) else {}
        except Exception:
            return {}

    def _collect_allowed_scope_files(self, log_dir: str, instance_id: str) -> List[str]:
        code_graph = self._load_code_graph(log_dir, instance_id)
        allowed_scope_files: List[str] = []

        def _append_path(raw_path: Any) -> None:
            normalized = self._normalize_rel_path(str(raw_path or ""))
            if normalized and normalized not in allowed_scope_files:
                allowed_scope_files.append(normalized)

        edit_targets = code_graph.get("edit_targets", [])
        seed_files = code_graph.get("seed_files", [])

        if isinstance(edit_targets, list):
            for item in edit_targets:
                if isinstance(item, dict):
                    _append_path(item.get("file"))

        if isinstance(seed_files, list):
            for file_path in seed_files:
                _append_path(file_path)

        return allowed_scope_files

    @staticmethod
    def _build_focused_visual_summary(image_graphs: Any) -> str:
        if not isinstance(image_graphs, list) or not image_graphs:
            return "none"

        def _normalize_summary_text(value: Any) -> str:
            return " ".join(str(value or "").split()).strip()

        def _node_text(node: Dict[str, Any]) -> str:
            for key in ("type", "text", "id"):
                value = str(node.get(key) or "").strip()
                if value:
                    return value
            return ""

        def _node_summary(node: Dict[str, Any]) -> str:
            label = ""
            for key in ("type", "id"):
                value = str(node.get(key) or "").strip()
                if value:
                    label = value
                    break
            text_value = _normalize_summary_text(node.get("text"))
            if label and text_value:
                return f"{label} [{text_value}]"
            if label:
                return label
            if text_value:
                return text_value
            return ""

        lines: List[str] = ["Focused Visual Graphs:"]
        graph_count = 0

        for index, graph in enumerate(image_graphs, start=1):
            if not isinstance(graph, dict) or not graph:
                continue

            nodes = graph.get("nodes")
            edges = graph.get("edges")
            if not isinstance(nodes, list):
                nodes = []
            if not isinstance(edges, list):
                edges = []

            root_nodes: List[Dict[str, Any]] = []
            for node in nodes:
                if isinstance(node, dict) and str(node.get("role") or "").strip().lower() == "root":
                    root_nodes.append(node)

            node_map: Dict[str, str] = {}
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id") or "").strip()
                if node_id:
                    node_map[node_id] = _node_summary(node) or _node_text(node)

            graph_count += 1
            lines.append(f"- Graph {index}:")

            root_text = ", ".join([_node_summary(node) for node in root_nodes[:3] if _node_summary(node)])
            lines.append(f"  Root objects: {root_text if root_text else 'none'}")

            relation_lines: List[str] = []
            for edge in edges[:3]:
                if not isinstance(edge, dict):
                    continue
                source = node_map.get(str(edge.get("source") or "").strip(), str(edge.get("source") or "").strip())
                target = node_map.get(str(edge.get("target") or "").strip(), str(edge.get("target") or "").strip())
                edge_type = str(edge.get("relation") or edge.get("type") or "").strip()
                if source and target and edge_type:
                    relation_lines.append(f"{source} --{edge_type}--> {target}")

            if relation_lines:
                lines.append("  Focused relations:")
                for relation in relation_lines:
                    lines.append(f"    - {relation}")

        return "\n".join(lines) if graph_count else "none"

    @staticmethod
    def _build_visual_prompt_context(doc: Dict[str, Any]) -> str:
        image_graphs = doc.get("image_graphs")
        if not isinstance(image_graphs, list) or not image_graphs:
            return "none"
        focused_visual_graph_summary = PatchGeneration._build_focused_visual_summary(image_graphs).strip() or "none"
        return focused_visual_graph_summary

    def _build_localization_prompt_context(self, log_dir: str, instance_id: str) -> str:
        code_graph = self._load_code_graph(log_dir, instance_id)
        seed_files = code_graph.get("seed_files", [])

        guidance_lines: List[str] = []
        if isinstance(seed_files, list) and seed_files:
            guidance_lines.append("Reranked files:")
            for file_path in seed_files:
                normalized = self._normalize_rel_path(str(file_path or ""))
                if normalized:
                    guidance_lines.append(f"- {normalized}")

        return "\n".join(guidance_lines).strip() if guidance_lines else "none"

    def _build_edit_targets_prompt_context(self, log_dir: str, instance_id: str) -> str:
        code_graph = self._load_code_graph(log_dir, instance_id)
        edit_targets = code_graph.get("edit_targets", [])

        guidance_lines: List[str] = []
        if isinstance(edit_targets, list) and edit_targets:
            guidance_lines.append("Suggested edit targets:")
            for item in edit_targets:
                if not isinstance(item, dict):
                    continue
                file_path = self._normalize_rel_path(str(item.get("file") or ""))
                function_name = str(item.get("function") or "").strip()
                role = str(item.get("role") or "").strip()
                if file_path and function_name:
                    if role:
                        guidance_lines.append(f"- {file_path} :: {function_name} [{role}]")
                    else:
                        guidance_lines.append(f"- {file_path} :: {function_name}")

        return "\n".join(guidance_lines).strip() if guidance_lines else "none"

    @staticmethod
    def _truncate_text(value: Any, max_len: int = 600) -> str:
        text = " ".join(str(value or "").split()).strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _build_previous_failure_reason(self, doc: Dict[str, Any]) -> str:
        candidates = [
            doc.get("_redo_failure_reason"),
            doc.get("validation_failure_reason"),
            doc.get("_agent_validation_failure_reason"),
            doc.get("_rule_validation_failure_reason"),
            doc.get("test_error"),
            doc.get("error"),
        ]
        for value in candidates:
            text = self._truncate_text(value, max_len=320)
            if text and text.lower() != "none":
                return text
        return "none"

    @staticmethod
    def get_processed_instances(output_dir: str) -> List[str]:
        if not os.path.exists(output_dir):
            return []

        processed = []
        for file_path in os.listdir(output_dir):
            if os.path.join(os.path.join(output_dir, file_path)):
                # Check if there are .patch suffix files in the directory
                patch_files = list(glob.glob(os.path.join(output_dir, file_path, "*.patch")))
                if patch_files:
                    processed.append(file_path)
        return processed

    def execute_single_task(self, doc: Dict, repo_path: str, output_dir: str, pbar: tqdm) -> Dict:
        """Execute a single task"""
        instance_id = doc["instance_id"]
        try:
            log_dir = os.path.join(output_dir, instance_id)
            os.makedirs(log_dir, exist_ok=True)

            problem = doc["problem_statement"]
            focused_visual_graph_summary = self._build_visual_prompt_context(doc)
            repo_dir = resolve_repo_dir(repo_path, doc["repo"])
            allowed_scope_files = self._collect_allowed_scope_files(
                log_dir=log_dir,
                instance_id=instance_id,
            )
            localization_guidance = self._build_localization_prompt_context(
                log_dir=log_dir,
                instance_id=instance_id,
            )
            edit_targets_guidance = self._build_edit_targets_prompt_context(
                log_dir=log_dir,
                instance_id=instance_id,
            )
            previous_failure_reason = self._build_previous_failure_reason(doc)

            patch_list = []
            last_error = ""

            # Clear possible git lock files
            lock_file = os.path.join(repo_dir, '.git', 'index.lock')
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.info(f"Removed git lock file: {lock_file}")

            patch_file = os.path.join(log_dir, f"res_patch_{instance_id}.patch")
            script_path = os.path.join(log_dir, f"{instance_id}_script.sh")
            user_prompt = Template(SubGraphPrompt).render({
                "problem_statement": problem,
                "focused_visual_graph_summary": focused_visual_graph_summary,
                "localization_guidance": localization_guidance,
                "edit_targets_guidance": edit_targets_guidance,
                "previous_failure_reason": previous_failure_reason,
            })

            prompt_file = os.path.join(log_dir, f"user_prompt_{instance_id}.txt")
            with open(prompt_file, "w", encoding="utf-8") as out_file:
                out_file.write(user_prompt)

            code_command = Template(COMMAND).render(
                {
                    "repo_dir": os.path.abspath(repo_dir),
                    "commit_id": doc["base_commit"],
                    "model_name": self.model_name,
                    "prompt_file": os.path.abspath(prompt_file),
                    "log_dir": os.path.abspath(log_dir),
                    "patch_file": os.path.abspath(patch_file),
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "temperature": self.temperature,
                }
            )
            with open(script_path, "w", encoding="utf-8") as outf:
                outf.write(code_command)

            logger.info(f"command: {code_command}")
            result = subprocess.run(
                f"sh {script_path}",
                capture_output=True,
                text=True,
                bufsize=0,
                shell=True,
            )
            if result.returncode != 0:
                last_error = (result.stderr or result.stdout or "").strip()
                logger.error(f"Error processing {instance_id}: exit_code={result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
            else:
                resp = result.stdout.strip()
                stderr_text = (result.stderr or "").strip()
                patch_text = ""
                if os.path.isfile(patch_file):
                    try:
                        with open(patch_file, "r", encoding="utf-8") as infile:
                            patch_text = infile.read().strip()
                    except Exception:
                        patch_text = ""

                error_markers = (
                    "error:",
                    "connection error",
                    "max retries exceeded",
                    "failed to establish a new connection",
                    "connection refused",
                    "api error",
                )
                combined_output = "\n".join([text for text in (resp, stderr_text) if text]).lower()
                has_error_output = any(marker in combined_output for marker in error_markers)
                has_valid_patch = bool(patch_text)

                if has_error_output or not has_valid_patch:
                    last_error = (stderr_text or resp or "empty patch output").strip()
                    logger.error(f"Error processing {instance_id}: invalid patch generation output")
                    if resp:
                        logger.error(f"STDOUT: {resp}")
                    if stderr_text:
                        logger.error(f"STDERR: {stderr_text}")
                else:
                    patch_list.append(resp)
                    logger.info(f"results: {resp}")
                    logger.info(f"Successfully processed {instance_id}")

            doc["fix_patch"] = patch_list
            if allowed_scope_files:
                doc["localization_scope_files"] = allowed_scope_files
            if not patch_list and last_error:
                doc["error"] = last_error

            with open(os.path.join(log_dir, f"resp_{instance_id}.json"), "w", encoding="utf-8") as out_res:
                out_res.write(json.dumps(doc, indent=4, ensure_ascii=False))

            # Update progress bar
            with self.progress_lock:
                pbar.update(1)

            return doc

        except Exception as e:
            logger.error(f"Exception processing {instance_id}: {str(e)}")
            traceback.print_exc()
            doc["error"] = str(e)

            # Update progress bar
            with self.progress_lock:
                pbar.update(1)

            return doc

    def process_repo_group(self, docs: List[Dict], repo_path: str, output_dir: str, pbar: tqdm) -> List[Dict]:
        """Process all tasks for a single repo group"""
        results = []
        for doc in docs:
            try:
                result = self.execute_single_task(doc, repo_path, output_dir, pbar)
                results.append(result)
            except Exception:
                print(f"error when processing {repo_path}")
                traceback.print_exc()
        return results

    def process_batch(
            self,
            data_path: str,
            output_dir: str,
            repo_path: str,
            copy_repo: bool = False
    ) -> None:
        # Copy repo_path to parent directory
        if copy_repo:
            parent_dir = os.path.dirname(os.path.abspath(repo_path))
            timestamp = str(int(time.time()))
            copied_repo_path = os.path.join(parent_dir, f"swebench_m_test_repos_copy_{timestamp}")

            logger.info(f"Copying repo from {repo_path} to {copied_repo_path}")
            shutil.copytree(repo_path, copied_repo_path)
            logger.info(f"Repo copied successfully to {copied_repo_path}")

            # Use copied directory as new repo_path
            repo_path = copied_repo_path

        # Load data
        with open(data_path, "r", encoding="utf-8") as infile:
            data_dict: Dict = json.load(infile)

        # Skip already-processed instances for normal full-run input.
        # For failed-subset reruns (e.g. all_validation_failed_instance.json),
        # keep old behavior and rerun all instances in the subset.
        processed_instances = set(self.get_processed_instances(output_dir))
        print(f"{len(processed_instances)} items have been processed")
        should_skip_processed = os.path.basename(os.path.abspath(data_path)) == "image_ir_data.json"

        if should_skip_processed:
            remaining_data = {k: v for k, v in data_dict.items() if k not in processed_instances}
        else:
            remaining_data = data_dict
        if not remaining_data:
            print("All instances have been processed")
            return

        repo_groups: Dict[str, List[Dict]] = {}
        for _, doc in remaining_data.items():
            repo = doc["repo"]
            if repo not in repo_groups:
                repo_groups[repo] = []
            repo_groups[repo].append(doc)

        print(f"Found {len(repo_groups)} different repositories to process")
        for repo, docs in repo_groups.items():
            print(f"  {repo}: {len(docs)} instances")

        # Create progress bar
        total_tasks = len(remaining_data)
        with tqdm(total=total_tasks, ncols=96, desc="Overall Progress") as pbar:
            # Use thread pool to concurrently execute tasks for different repos
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_repo = {}

                # Submit tasks for each repo group
                for repo_name, docs in repo_groups.items():
                    future = executor.submit(
                        self.process_repo_group,
                        docs,
                        repo_path,
                        output_dir,
                        pbar
                    )
                    future_to_repo[future] = repo_name

                # Wait for all tasks to complete
                all_results = []
                for future in as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        logger.info(f"Completed processing repository: {repo_name}")
                    except Exception as e:
                        logger.error(f"Error processing repository {repo_name}: {str(e)}")
                        traceback.print_exc()

        logger.info(f"Processing completed. Total processed: {len(all_results)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SWE Bench MM Batch Processing Program")
    parser.add_argument("--data_path", type=str, required=True, help="Input JSON data file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output results directory")
    parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", required=True, help="Model name")
    parser.add_argument("--base_url", required=True, help="Base URL for API")
    parser.add_argument("--repo_path", type=str, help="Repository base path")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum concurrent threads")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--copy_repo", action="store_true", help="Whether to copy repo directory before starting")

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save experiment configuration
    config_file = ConfigManager.save_experiment_config(args, output_dir)
    logger.info(f"Experiment configuration saved to: {config_file}")

    data_path = args.data_path

    logger.info(f"Starting processing with data: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Repo path: {args.repo_path}")

    # Create batch processor and run
    processor = PatchGeneration(args.model_name, args.base_url, args.max_workers, args.temperature)
    processor.process_batch(data_path, output_dir, args.repo_path, copy_repo=args.copy_repo)

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
