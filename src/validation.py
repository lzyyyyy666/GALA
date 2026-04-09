import json
import os
import argparse
import traceback
from typing import *
import glob
from src.utils.logger import logger

class Validation:
    def __init__(self):
        pass

    @staticmethod
    def find_resp_file(instance_dir: str):
        candidates = [
            os.path.join(instance_dir, "resp_*.json"),
            os.path.join(instance_dir, "test_log", "resp_*.json"),
        ]
        for pattern in candidates:
            resp_data = glob.glob(pattern)
            if resp_data:
                return resp_data[0]
        return None

    @staticmethod
    def check_for_empty_result(instance_dir: str):
        resp_data = Validation.find_resp_file(instance_dir)
        patch_data = glob.glob(os.path.join(instance_dir, "*.patch"))

        if not resp_data or not patch_data:
            return None

        with open(patch_data[0], "r") as infile:
            patch = infile.read()

        if not patch:
            return None

        return resp_data, patch_data[0]

    @staticmethod
    def check_for_model_failed(resp_data: str):
        with open(resp_data, "r") as infile:
            res_dict = json.load(infile)

        if "fix_patch" not in res_dict:
            return res_dict

        fix_patch = res_dict.get("fix_patch") or []
        if not isinstance(fix_patch, list) or not fix_patch:
            return None

        res_data = fix_patch[0]
        if not isinstance(res_data, str):
            return None
        if "context_length_exceeded" in res_data.lower():
            return None

        if "Error:" in res_data:
            return None

        return res_dict

    def process_instance(self, instance_dir: str):
        ###### step 1. check for empty result
        empty_patch = self.check_for_empty_result(instance_dir)
        if empty_patch is None:
            return {
                "reason": "empty patch.",
                "instance_dir": instance_dir
            }
        else:
            resp_data, patch_data = empty_patch

        ####### step 2. check for model failed
        res_dict = self.check_for_model_failed(resp_data)
        if res_dict is None:
            return {
                "reason": "model failed. ",
                "instance_dir": instance_dir
            }

        # step 3. check for patch
        patch_size = os.path.getsize(patch_data) / 1024 / 1024
        if patch_size > 15:
            return {
                "reason": "Patch exceed limitation. ",
                "instance_dir": instance_dir
            }

        return None

    def filtering_result(self, data_path: str, result_path: str):
        with open(data_path, "r") as infile:
            instance_dict = json.load(infile)
        logger.info(f"{len(instance_dict)} need to be process")

        error_instance = {}
        crt_instance = {}
        for instance_id in os.listdir(result_path):
            if not os.path.isdir(os.path.join(result_path, instance_id)):
                continue

            if instance_id not in instance_dict:
                continue

            try:
                instance_dir = os.path.join(result_path, instance_id)
                # logger.info("instance dir: ", instance_dir)
                valid_res = self.process_instance(instance_dir)
                # If already processed, proceed to validation step
                if valid_res is not None:
                    failed_doc = instance_dict.pop(instance_id)
                    if isinstance(failed_doc, dict):
                        failed_doc = dict(failed_doc)
                        rule_reason = str(valid_res.get("reason") or "").strip()
                        failed_doc["_rule_validation_failure_reason"] = rule_reason
                        failed_doc["_rule_validation_failure_stage"] = "rule_validation"
                        failed_doc["_redo_failure_reason"] = rule_reason or "rule validation failed"
                    error_instance[instance_id] = failed_doc
                else:
                    crt_instance[instance_id] = instance_dict.pop(instance_id)
            except Exception as e:
                traceback.print_exc()
                # exit()
                failed_doc = instance_dict.pop(instance_id)
                if isinstance(failed_doc, dict):
                    failed_doc = dict(failed_doc)
                    rule_reason = f"rule validation exception: {str(e)}"
                    failed_doc["_rule_validation_failure_reason"] = rule_reason
                    failed_doc["_rule_validation_failure_stage"] = "rule_validation_exception"
                    failed_doc["_redo_failure_reason"] = rule_reason
                error_instance[instance_id] = failed_doc

        for k, v in instance_dict.items():
            pending_doc = v
            if isinstance(pending_doc, dict):
                pending_doc = dict(pending_doc)
                rule_reason = "missing generation result directory or response artifacts"
                pending_doc["_rule_validation_failure_reason"] = rule_reason
                pending_doc["_rule_validation_failure_stage"] = "rule_validation_missing_artifacts"
                pending_doc["_redo_failure_reason"] = rule_reason
            error_instance[k] = pending_doc

        print("checked result: ", len(error_instance))
        failed_path = os.path.join(result_path, "model_response_validation_failed.json")
        with open(failed_path, "w") as outf:
            outf.write(json.dumps(error_instance, indent=4, ensure_ascii=False))

        crt_path = os.path.join(result_path, "model_response_validation_success.json")
        with open(crt_path, "w") as outf:
            outf.write(json.dumps(crt_instance, indent=4, ensure_ascii=False))

        logger.info(f"validation done, {len(error_instance)} cannot pass the validation")
        logger.info(f"{len(crt_instance)} has been validated to be correct.")

        return failed_path, crt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    validation = Validation()
    validation.filtering_result(args.data_path, args.result_path)


if __name__ == "__main__":
    # Main execution code
    main()
