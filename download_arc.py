import requests
import zipfile
import io
import os
import shutil

URL = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"
TARGET_DIR = "data/arc_agi"

def download_and_extract():
    print(f"Downloading ARC-AGI from {URL}...")
    r = requests.get(URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    print("Extracting...")
    z.extractall("temp_arc")
    
    # Move evaluation folder
    source = os.path.join("temp_arc", "ARC-AGI-2-main", "data", "evaluation")
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    shutil.copytree(source, TARGET_DIR)
    
    print(f"Data ready in {TARGET_DIR}")
    
    # Cleanup
    shutil.rmtree("temp_arc")

if __name__ == "__main__":
    download_and_extract()
