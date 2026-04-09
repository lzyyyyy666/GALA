import argparse
import json
import os
import glob
import traceback


def _write_result_records(records, output_base_path: str) -> None:
    jsonl_path = f"{output_base_path}.jsonl"
    json_path = f"{output_base_path}.json"

    with open(jsonl_path, "w", encoding="utf-8") as outf:
        for record in records:
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as outf:
        json.dump(records, outf, indent=4, ensure_ascii=False)


def convert_list_to_dict(data_path: str, output_dir: str):
    data_dict = {}

    with open(data_path, "r") as infile:
        data_list = json.load(infile)

    for doc in data_list:
        data_dict[doc["instance_id"]] = doc

    with open(os.path.join(output_dir, "swe_bench_mm_prompt_v3_dict.json"), "w") as outf:
        outf.write(json.dumps(data_dict, indent=4, ensure_ascii=False))


def process_git_diff(result_path: str, model_name: str = "GALA", result_tag: str = ""):
    records = []
    for instance_id in os.listdir(result_path):
        res_patch = list(glob.glob(os.path.join(result_path, instance_id, "*.patch")))
        if res_patch:
            res_patch = res_patch[0]
        else:
            continue
        try:
            with open(res_patch, "r") as infile:
                patch = infile.read()
                records.append(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": model_name,
                        "model_patch": patch,
                    }
                )
        except:
            print(f"Error when processing: {instance_id}")
            traceback.print_exc()
            records.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": model_name,
                    "model_patch": "",
                }
            )

    suffix = "_result_path"
    if result_tag:
        output_base_name = f"{model_name}_{result_tag}{suffix}"
    else:
        output_base_name = f"{model_name}{suffix}"
    output_base_path = os.path.join(result_path, output_base_name)
    _write_result_records(records, output_base_path)

    print(f"Total result={len(records)}")


def process_result_with_feedback(repo_data_file: str, result_path: str, model_name: str = "GALA"):
    with open(repo_data_file, "r") as infile:
        repo_data = json.load(infile)

    for instance_id in os.listdir(result_path):
        # if instance_id in repo_data:
        #     continue

        res_patch = list(glob.glob(os.path.join(result_path, instance_id, "*.patch")))
        if res_patch:
            res_patch = res_patch[0]
        else:
            continue
        try:
            with open(res_patch, "r") as infile:
                patch = infile.read()
                repo_data[instance_id] = {
                    "model_name_or_path": model_name,
                    "model_patch": patch
                }
        except:
            print(f"Error when processing: {instance_id}")
            traceback.print_exc()
            repo_data[instance_id] = {
                "model_name_or_path": model_name,
                "model_patch": patch
            }

    records = []
    for instance_id, payload in repo_data.items():
        if not isinstance(payload, dict):
            continue
        records.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": payload.get("model_name_or_path", model_name),
                "model_patch": payload.get("model_patch", ""),
            }
        )

    output_base_path = os.path.join(result_path, f"{model_name}_result")
    _write_result_records(records, output_base_path)

    print(f"total result: {len(repo_data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--repo_file")
    parser.add_argument("--model_name", default="GALA")
    parser.add_argument("--result_tag", default="")
    args = parser.parse_args()

    if args.repo_file is not None:
        process_result_with_feedback(args.repo_file, args.result_path)
    else:
        process_git_diff(args.result_path, model_name=args.model_name, result_tag=args.result_tag)

    print("process result done.")


if __name__ == "__main__":
    # Main execution code
    main()
