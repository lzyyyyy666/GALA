"""
Data Processing Tool: Read parquet file, convert to dictionary format, and optionally download git repositories

Usage:
    python convert_input_to_dict.py --parquet-path <parquet file path> --output-path <output json path> [options]

Parameter Description:
    --parquet-path    Input parquet file path (required)
    --output-path     Output JSON file path (required)
    --repo-dir        Repository download directory (default: /data/repos)
    --skip-download   Skip repository download step

Examples:
    # Basic usage
    python convert_input_to_dict.py --parquet-path data.parquet --output-path result.json
    
    # Specify repository download directory
    python convert_input_to_dict.py --parquet-path data.parquet --output-path result.json --repo-dir ./repos
    
    # Skip repository download
    python convert_input_to_dict.py --parquet-path data.parquet --output-path result.json --skip-download
"""

import json
import argparse
import pandas as pd
import json
import os

repo_urls = {
    "GoogleChrome": "https://github.com/GoogleChrome/lighthouse.git",
    "alibaba-fusion": "https://github.com/alibaba-fusion/next.git",
    "bpmn-io": "https://github.com/bpmn-io/bpmn-js.git",
    "carbon-design-system": "https://github.com/carbon-design-system/carbon.git",
    "eslint": "https://github.com/eslint/eslint.git",
    "grommet": "https://github.com/grommet/grommet.git",
    "highlightjs": "https://github.com/highlightjs/highlight.js.git",
    "openlayers": "https://github.com/openlayers/openlayers.git",
    "prettier": "https://github.com/prettier/prettier.git",
    "PrismJS": "https://github.com/PrismJS/prism.git",
    "quarto-dev": "https://github.com/quarto-dev/quarto-cli.git",
    "scratchfoundation": "https://github.com/scratchfoundation/scratch-gui.git"
}


def process_parquet_to_json(parquet_path):
    """
    Convert parquet file to JSON format (without saving intermediate results)
    
    Args:
        parquet_path: parquet file path
    
    Returns:
        Converted data list
    """
    # Read parquet file
    df = pd.read_parquet(parquet_path)
    
    # Convert each row of DataFrame to dictionary
    data_list = []
    
    for index, row in df.iterrows():
        # Convert each row to dictionary, key as column name, value as corresponding position value
        row_dict = row.to_dict()
        data_list.append(row_dict)
    
    print(f"Conversion completed! Processed {len(data_list)} rows of data")
    
    return data_list

def process_data_to_dict(data_list, output_path):
    """
    Process data list to dictionary format and save
    
    Args:
        data_list: Data list
        output_path: Output file path
    """
    out_dict = {}
    for doc in data_list:
        out_dict[doc["instance_id"]] = doc

    with open(output_path, "w", encoding='utf-8') as outf:
        outf.write(json.dumps(out_dict, ensure_ascii=False, indent=4))
    
    print(f"Dictionary format data saved to: {output_path}")
    return out_dict

def download_repos(repo_urls, base_dir="/data/repos"):
    """
    Download specified git repositories to corresponding folders
    
    Args:
        repo_urls: Repository URL dictionary
        base_dir: Base download directory
    """
    import subprocess
    import os
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    for repo_name, repo_url in repo_urls.items():
        repo_path = os.path.join(base_dir, repo_name)
        
        if os.path.exists(repo_path):
            print(f"Repository {repo_name} already exists, skipping download")
            continue
            
        try:
            print(f"Downloading {repo_name} from {repo_url}...")
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            print(f"✓ {repo_name} download completed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Download {repo_name} failed: {e}")

def main():
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='Process parquet file and download git repositories')
    parser.add_argument('--parquet-path', required=True, 
                       help='Input parquet file path')
    parser.add_argument('--output-path', required=True,
                       help='Output JSON file path')
    parser.add_argument('--repo-dir', default='/data/repos',
                       help='Repository download directory (default: /data/repos)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip repository download step')
    
    args = parser.parse_args()
    
    # Step 1: Read parquet and convert to json list (without saving intermediate results)
    print(f"Step 1: Reading parquet file {args.parquet_path}...")
    data_list = process_parquet_to_json(args.parquet_path)
    
    # Step 2: Process data to dictionary format and save final result
    print(f"Step 2: Processing data to dictionary format, saving to {args.output_path}...")
    process_data_to_dict(data_list, args.output_path)
    
    # Step 3: Download repositories (optional)
    if not args.skip_download:
        print(f"Step 3: Downloading repositories to {args.repo_dir}...")
        download_repos(repo_urls, base_dir=args.repo_dir)
    else:
        print("Step 3: Skipping repository download")


# Usage example
if __name__ == "__main__":
    main()