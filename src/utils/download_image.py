#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os.path
import traceback
import argparse
import requests
from tqdm import tqdm


def download_img(doc: dict, output_dir: str):
    repo_name = doc["repo"].split("/")[-1]
    commit_id = doc["base_commit"]
    instance_id = doc["instance_id"]

    # Create image save directory
    images_dir = output_dir
    os.makedirs(images_dir, exist_ok=True)

    problem = doc["problem_statement"]
    image_list = json.loads(doc["image_assets"])["problem_statement"]
    
    # Multi-threaded image download
    import concurrent.futures
    
    def download_single_image(idx: int, image_url: str):
        """Function to download a single image"""
        try:
            # Extract filename from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"image_{idx}.jpg"
            
            # Create local file path
            local_path = os.path.join(images_dir, f"{instance_id}_{filename}")
            
            # Check if file already exists
            if os.path.exists(local_path):
                print(f"Skipping image {idx}: already exists at {local_path}")
                return idx, local_path
            
            print(f"Downloading image {idx}: {image_url}")
            
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Save image to local
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded image {idx} to {local_path}")
            return idx, local_path
            
        except Exception as e:
            print(f"Error downloading image {idx}: repo={instance_id} image={image_url}")
            print(traceback.format_exc())
            return idx, f"Error: {str(e)}"
    
    # Use multi-threading to download all images
    max_workers = min(8, len(image_list)) if image_list else 1  # Dynamically adjust thread count based on image quantity, maximum 8 threads
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(download_single_image, idx, image_url): idx
            for idx, image_url in enumerate(image_list)
        }

def dl_img(input_data: str, output_dir: str):
    processed_instance = []

    print(f"{len(processed_instance)} has been processed. ")
    os.makedirs(output_dir, exist_ok=True)
    with open(input_data, "r") as infile:
        dats = json.load(infile)

    out_doc_list = []
    bar = tqdm(total=len(dats), ncols=96)
    for key, value in dats.items():
        if key in processed_instance:
            continue
        # print(doc)
        bar.update()
        try:
            download_img(value, output_dir)
        except:
            print(f"Error: {key}")
            print(traceback.format_exc())
            out_doc_list.append(value)

if __name__ == "__main__":
    # Main execution code
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--image_dir", required=True)
    args = parser.parse_args()

    dl_img(args.input_data, args.image_dir)
    
    