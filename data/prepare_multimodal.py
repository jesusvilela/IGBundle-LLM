import os
import json
import logging
import base64
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Epic 18: Dataset Preparation (Gemini)
DATA_DIR = "data/multimodal"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_and_save_llava_instruct():
    logger.info("Downloading LLaVA-Instruct-150K subset...")
    try:
        ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
        output_file = os.path.join(DATA_DIR, "llava_instruct_subset.jsonl")
        count = 0
        limit = 5000
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, total=limit, desc="Processing LLaVA-Instruct"):
                if count >= limit: break
                f.write(json.dumps(item) + "\n")
                count += 1
        logger.info(f"Saved {count} samples to {output_file}")
    except Exception as e:
        logger.error(f"Failed to prepare LLaVA-Instruct: {e}")

def download_and_save_sharegpt4v():
    logger.info("Downloading ShareGPT4V subset...")
    try:
        # Fixed: Using available config 'ShareGPT4V'
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train", streaming=True)
        output_file = os.path.join(DATA_DIR, "sharegpt4v_subset.jsonl")
        count = 0
        limit = 2000
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, total=limit, desc="Processing ShareGPT4V"):
                if count >= limit: break
                # Item might contain PIL Image, need to handle or strip
                serializable_item = {}
                for k, v in item.items():
                    if isinstance(v, Image.Image):
                        continue # Skip raw image objects for now, or save to disk
                    serializable_item[k] = v
                f.write(json.dumps(serializable_item) + "\n")
                count += 1
        logger.info(f"Saved {count} samples to {output_file}")
    except Exception as e:
        logger.error(f"Failed to prepare ShareGPT4V: {e}")

def download_and_save_cauldron_subset():
    logger.info("Downloading The Cauldron subset (ChartQA)...")
    try:
        ds = load_dataset("HuggingFaceM4/the_cauldron", "chartqa", split="train", streaming=True)
        output_file = os.path.join(DATA_DIR, "cauldron_chartqa_subset.jsonl")
        count = 0
        limit = 1000
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, total=limit, desc="Processing The Cauldron"):
                if count >= limit: break
                
                # Handle PIL Images in The Cauldron
                serializable_item = {}
                for k, v in item.items():
                    if k == "images" and isinstance(v, list):
                        img_paths = []
                        for i, img in enumerate(v):
                            if isinstance(img, Image.Image):
                                img_filename = f"cauldron_{count}_{i}.png"
                                img_path = os.path.join(IMAGE_DIR, img_filename)
                                img.save(img_path)
                                img_paths.append(img_path)
                        serializable_item["image_paths"] = img_paths
                    elif isinstance(v, Image.Image):
                        img_filename = f"cauldron_{count}.png"
                        img_path = os.path.join(IMAGE_DIR, img_filename)
                        v.save(img_path)
                        serializable_item["image_path"] = img_path
                    else:
                        serializable_item[k] = v
                
                f.write(json.dumps(serializable_item) + "\n")
                count += 1
        logger.info(f"Saved {count} samples to {output_file}")
    except Exception as e:
        logger.error(f"Failed to prepare The Cauldron: {e}")

if __name__ == "__main__":
    # Skipping LLaVA since it worked, to save time on retry
    # download_and_save_llava_instruct() 
    download_and_save_sharegpt4v()
    download_and_save_cauldron_subset()
    logger.info("Epic 18: Multimodal dataset preparation updated.")
