import os
import json
import re
import sys
import argparse
from typing import List, Dict, Any

from xml.etree.ElementTree import indent


def extract_test_result(llm_test_result):
    # Use regular expression to match content within <result> tags
    pattern = r'<result>\s*([^<]+)\s*</result>'
    match = re.search(pattern, llm_test_result, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    else:
        return None


def read_resp_json_files(result_path: str) -> tuple[Dict[str, Any], set[str]]:
    """
    Traverse all subdirectories under result_path, read json files starting with resp_ in test_log directory
    
    Args:
        result_path: Root directory path to traverse
        
    Returns:
        List[Dict[str, Any]]: List of all json file contents starting with resp_
    """
    failed_instance = {}
    success_instance_ids = set()

    # Traverse all subdirectories under result_path
    for instance_id in os.listdir(result_path):
        item_path = os.path.join(result_path, instance_id)
        
        # Only process directories
        if os.path.isdir(item_path):
            test_log_path = os.path.join(item_path, "test_log")
            # Check if test_log directory exists
            if os.path.exists(test_log_path) and os.path.isdir(test_log_path):
                # Traverse files in test_log directory
                resp_file = os.path.join(test_log_path, f"resp_{instance_id}.json")
                if not os.path.exists(resp_file):
                    print(f"cannot find resp: {instance_id}")
                    continue
                with open(resp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if "llm_test_result" not in data:
                        continue
                    llm_resp = data["llm_test_result"]
                    validation_result = extract_test_result(llm_resp)
                    compact_reason = str(data.get("validation_failure_reason") or "").strip()
                    if validation_result is None:
                        if not compact_reason:
                            compact_reason = "agent validation output missing <result> tag"
                        data["_agent_validation_failure_reason"] = compact_reason
                        data["_agent_validation_failure_stage"] = "agent_validation"
                        data["_redo_failure_reason"] = compact_reason
                        failed_instance[instance_id] = data
                    elif "success" not in validation_result.strip().lower():
                        if not compact_reason:
                            compact_reason = f"agent validation judged as non-success: {validation_result}"
                        data["_agent_validation_failure_reason"] = compact_reason
                        data["_agent_validation_failure_stage"] = "agent_validation"
                        data["_redo_failure_reason"] = compact_reason
                        failed_instance[instance_id] = data
                    elif not data['validation_result']:
                        if not compact_reason:
                            compact_reason = "agent validation flag is false"
                        data["_agent_validation_failure_reason"] = compact_reason
                        data["_agent_validation_failure_stage"] = "agent_validation"
                        data["_redo_failure_reason"] = compact_reason
                        failed_instance[instance_id] = data
                    else:
                        success_instance_ids.add(instance_id)
                        print(f"{instance_id} validation result: {validation_result}")
                        continue
            else:
                print(f"{instance_id} test log not exist")

    print("total failed instance: ", len(failed_instance))

    return failed_instance, success_instance_ids


def merge_result(agent_failed_result: Dict[str, Any], rule_val_path: str, success_instance_ids: set[str]):
    with open(rule_val_path, "r") as infile:
        rule_val_result = json.load(infile)
    agent_result = dict(agent_failed_result)

    for k in rule_val_result:
        if k in success_instance_ids:
            continue
        if k not in agent_result:
            merged_doc = rule_val_result[k]
            if isinstance(merged_doc, dict):
                merged_doc = dict(merged_doc)
                if not str(merged_doc.get("_agent_validation_failure_reason") or "").strip():
                    merged_doc["_agent_validation_failure_reason"] = "not executed in agent validation stage (filtered by previous rule-based failure)"
                    merged_doc["_agent_validation_failure_stage"] = "agent_validation_skipped"
                if not str(merged_doc.get("_redo_failure_reason") or "").strip():
                    merged_doc["_redo_failure_reason"] = (
                        str(merged_doc.get("_rule_validation_failure_reason") or "").strip()
                        or str(merged_doc.get("_agent_validation_failure_reason") or "").strip()
                        or "validation failed"
                    )
            agent_result[k] = merged_doc

    return agent_result


def process_val(result_path):
    data, success_instance_ids = read_resp_json_files(result_path)

    agent_path = os.path.join(result_path, "agent_validation_failed_instance.json")
    with open(agent_path, "w") as outf:
        outf.write(json.dumps(data, indent=4, ensure_ascii=False))

    rule_path = os.path.join(result_path, "model_response_validation_failed.json")
    all_result = merge_result(data, rule_path, success_instance_ids)
    
    with open(os.path.join(result_path, "all_validation_failed_instance.json"), "w") as outf:
            outf.write(json.dumps(all_result, indent=4, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    process_val(args.result_path)
    

if __name__ == "__main__":
    main()
