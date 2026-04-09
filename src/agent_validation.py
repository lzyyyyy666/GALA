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
    repo_path: Repository path (optional)

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
import re

from tqdm import tqdm
from jinja2 import Template

from prompt.test_prompt import TestPrompt
from src.utils.logger import logger
from src.utils.repo_resolver import resolve_repo_dir

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

class AgentBaseValidation:
    """Batch process multiple documents"""

    def __init__(self, repo_path: str, model_name: str, base_url: str,max_workers: int = 4):
        self.model_name = model_name
        self.max_workers = max_workers
        self.progress_lock = threading.Lock()
        self.base_url = base_url
        self.api_key = OPENAI_API_KEY

        self.repo_path = repo_path

    @staticmethod
    def extract_test_result(llm_test_result):
        # Use regular expression to match <result> tag content
        pattern = r'<result>\s*([^<]+)\s*</result>'
        match = re.search(pattern, llm_test_result, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()
        else:
            return None

    @staticmethod
    def extract_failure_reason(llm_test_result):
        pattern = r'<failure_reason>\s*([^<]+)\s*</failure_reason>'
        match = re.search(pattern, llm_test_result, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def get_processed_instances(output_dir: str) -> List[str]:
        if not os.path.exists(output_dir):
            return []

        processed = []
        for file_path in os.listdir(output_dir):
            instance_dir = os.path.join(output_dir, file_path)
            if not os.path.isdir(instance_dir):
                continue
            resp_file = os.path.join(instance_dir, "test_log", f"resp_{file_path}.json")
            if os.path.isfile(resp_file):
                processed.append(file_path)
        return processed

    @staticmethod
    def _build_focused_visual_summary(image_graphs: Any) -> str:
        if not isinstance(image_graphs, list) or not image_graphs:
            return "none"
        graph = next((item for item in image_graphs if isinstance(item, dict) and item), None)
        if not isinstance(graph, dict):
            return "none"
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []

        root_nodes = []
        for node in nodes:
            if isinstance(node, dict) and str(node.get("role") or "").strip().lower() == "root":
                root_nodes.append(node)

        def _node_text(node: Dict[str, Any]) -> str:
            for key in ("type", "text", "id"):
                value = str(node.get(key) or "").strip()
                if value:
                    return value
            return ""

        node_map = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            if node_id:
                node_map[node_id] = _node_text(node)

        lines = ["Focused Visual Graph:"]
        root_text = ", ".join([_node_text(node) for node in root_nodes[:5] if _node_text(node)])
        lines.append(f"- Root objects: {root_text if root_text else 'none'}")

        relation_lines = []
        for edge in edges[:5]:
            if not isinstance(edge, dict):
                continue
            source = node_map.get(str(edge.get("source") or "").strip(), str(edge.get("source") or "").strip())
            target = node_map.get(str(edge.get("target") or "").strip(), str(edge.get("target") or "").strip())
            edge_type = str(edge.get("relation") or edge.get("type") or "").strip()
            if source and target and edge_type:
                relation_lines.append(f"{source} --{edge_type}--> {target}")

        if relation_lines:
            lines.append("- Focused relations:")
            for relation in relation_lines:
                lines.append(f"  - {relation}")
        return "\n".join(lines)

    @staticmethod
    def _build_visual_prompt_context(res_data: Dict[str, Any]) -> str:
        image_graphs = res_data.get("image_graphs")
        if not isinstance(image_graphs, list) or not image_graphs:
            return "none"
        graph = next((item for item in image_graphs if isinstance(item, dict) and item), None)
        if not isinstance(graph, dict):
            return "none"
        focused_visual_graph_summary = AgentBaseValidation._build_focused_visual_summary(image_graphs).strip() or "none"
        return focused_visual_graph_summary

    def run_test_command(self, repo_dir: str, commit_id: str, patch_file: str, prompt_file: str, log_dir: str):
        command = """cd {{repo_dir}} && git reset --hard HEAD && git checkout -f {{commit_id}} && 
           CFUSE_BIN="${CFUSE_BIN:-$(command -v pycfuse || command -v cfuse)}" && [ -n "$CFUSE_BIN" ] &&
           CFUSE_STREAM_FLAG="${CFUSE_STREAM_FLAG:---no-stream}" &&
           "$CFUSE_BIN" --model {{model_name}} --api-key {{api_key}} --base-url {{base_url}} -pp {{prompt_file}} --logs-dir {{log_dir}} --yolo ${CFUSE_STREAM_FLAG} """

        run_command = Template(command).render({
            "repo_dir": repo_dir,
            "commit_id": commit_id,
            "patch_file": patch_file,
            "model_name": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "prompt_file": prompt_file,
            "log_dir": log_dir
        })
        return run_command

    def run_llm_test(self, res_data: Dict, result_path: str, repo_path: str, pbar: tqdm):
        """Execute single task"""
        flag = False
        failure_reason = ""
        instance_dir = os.path.join(result_path, res_data["instance_id"])
        print("instance_dir: ", instance_dir)
        patch_file = glob.glob(os.path.join(instance_dir, "*.patch"))[0]
        print("patch file: ", patch_file)

        log_dir = os.path.join(instance_dir, "test_log")
        os.makedirs(log_dir, exist_ok=True)
        instance_id = os.path.basename(instance_dir)
        logger.info(f"run llm test on {instance_id}")

        problem = res_data["problem_statement"]
        focused_visual_graph_summary = self._build_visual_prompt_context(res_data)
        img_url = res_data.get("image_assets", "")
        repo_dir = resolve_repo_dir(repo_path, res_data["repo"])

        user_prompt = Template(TestPrompt).render({
            "problem_statement": problem,
            "focused_visual_graph_summary": focused_visual_graph_summary,
            "patch_file": os.path.abspath(patch_file),
            "image_file": img_url,
        })

        prompt_file = os.path.join(log_dir, f"user_prompt.txt")
        with open(prompt_file, "w") as out_file:
            out_file.write(user_prompt)

        try:
            # Clear possible git lock files
            lock_file = os.path.join(repo_dir, '.git', 'index.lock')
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.info(f"Removed git lock file: {lock_file}")

            script_path = os.path.join(log_dir, f"run_test_script.sh")
            test_command = self.run_test_command(
                repo_dir,
                commit_id=res_data["base_commit"],
                patch_file=os.path.abspath(patch_file),
                prompt_file=os.path.abspath(prompt_file),
                log_dir=os.path.abspath(log_dir)
            )

            with open(script_path, "w") as outf:
                outf.write(test_command)

            logger.info(f"command: {test_command}")

            test_result = subprocess.run(f"sh {script_path}", capture_output=True, text=True, bufsize=0, shell=True)

            # Update progress bar
            with self.progress_lock:
                pbar.update(1)

            if test_result.returncode != 0:
                logger.error(f"Error processing {instance_id}: exit_code={test_result.returncode}")
                logger.error(f"STDERR: {test_result.stderr}")
                res_data["test_error"] = test_result.stderr
                failure_reason = (test_result.stderr or "").strip() or f"validation command failed with exit code {test_result.returncode}"
            else:
                resp = test_result.stdout.strip()
                # logger.info(f"results: {resp}")
                res_data["llm_test_result"] = resp
                logger.info(f"Successfully processed {instance_id}")
                try:
                    validation_result = self.extract_test_result(resp)
                    if validation_result and "success" in validation_result.lower():
                        flag = True
                    parsed_failure_reason = self.extract_failure_reason(resp)
                    if parsed_failure_reason:
                        failure_reason = parsed_failure_reason
                    elif not flag:
                        failure_reason = (
                            (validation_result or "").strip()
                            or "validation result is failed"
                        )
                except Exception as e:
                    logger.error(f"cannot parse result: {instance_id}, error: {str(e)}")
                    traceback.print_exc()
                    failure_reason = f"cannot parse validation output: {str(e)}"

                # logger.info(f"llm result: {validation_result}")

        except Exception as e:
            logger.error(f"Exception processing {instance_id}: {str(e)}")
            traceback.print_exc()
            failure_reason = f"validation exception: {str(e)}"

            # Update progress bar
            with self.progress_lock:
                pbar.update(1)

        res_data["validation_result"] = flag
        if not flag:
            res_data["validation_failure_reason"] = (failure_reason or "validation failed").strip()
        else:
            res_data["validation_failure_reason"] = "none"
        with open(os.path.join(log_dir, f"resp_{instance_id}.json"), "w") as out_res:
            out_res.write(json.dumps(res_data, indent=4, ensure_ascii=False))

        return res_data

    def process_repo_group(self, docs: List[Dict], repo_path: str, result_path: str, pbar: tqdm) -> List[Dict]:
        """Process all tasks for a single repo group"""
        results = []
        for doc in docs:
            try:
                result = self.run_llm_test(res_data=doc, result_path=result_path, repo_path=repo_path, pbar=pbar)
                results.append(result)
            except:
                print(f"error when processing {repo_path}")
                traceback.print_exc()
        return results

    def process_batch(
            self,
            data_path: str,
            result_path: str,
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
        with open(data_path, "r") as infile:
            data_dict: Dict = json.load(infile)

        # Get processed instances
        processed_instances = set(self.get_processed_instances(result_path))
        print(f"{len(processed_instances)} items have been processed")

        failed_file = os.path.join(result_path, "model_response_validation_failed.json")
        failed_data_dict: Dict = {}
        if os.path.exists(failed_file):
            try:
                with open(failed_file, "r", encoding="utf-8") as infile:
                    loaded_failed = json.load(infile)
                if isinstance(loaded_failed, dict):
                    failed_data_dict = loaded_failed
            except Exception:
                failed_data_dict = {}

        # If current input is already the rule-failed set, do not exclude it again.
        exclude_rule_failed = os.path.abspath(data_path) != os.path.abspath(failed_file)

        if exclude_rule_failed and failed_data_dict:
            remaining_data = {
                k: v for k, v in data_dict.items()
                if k not in processed_instances and k not in failed_data_dict
            }
        else:
            remaining_data = {
                k: v for k, v in data_dict.items()
                if k not in processed_instances
            }

        if not remaining_data:
            print("All instances have been processed")
            return

        # Group by repository
        repo_groups: Dict[str, List[Dict]] = {}
        for instance_id, doc in remaining_data.items():
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

            # Use thread pool to execute tasks for different repositories concurrently
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_repo = {}

                # Submit tasks for each repository group
                for repo_name, docs in repo_groups.items():
                    future = executor.submit(
                        self.process_repo_group,
                        docs,
                        repo_path,
                        result_path,
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
    parser.add_argument("--result_path", type=str, required=True, help="Output results directory")
    parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905")
    parser.add_argument("--base_url", required=True, help="Base URL for API")
    parser.add_argument("--repo_path", type=str, default="data/swe_bench_mm/repos", help="Repository base path")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum concurrent threads")
    parser.add_argument("--copy_repo", action="store_true", help="Whether to copy repo directory before starting")

    args = parser.parse_args()

    data_path = args.data_path

    logger.info(f"Starting processing with data: {data_path}")
    logger.info(f"Output directory: {args.result_path}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Repo path: {args.repo_path}")

    # Create batch processor and run
    processor = AgentBaseValidation(repo_path=args.repo_path, model_name=args.model_name, base_url=args.base_url, max_workers=args.max_workers)
    processor.process_batch(data_path, args.result_path, args.repo_path, copy_repo=args.copy_repo)

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
