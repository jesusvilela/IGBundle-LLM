
import logging
from datasets import load_dataset
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect(dataset_name):
    logger.info(f"Inspecting {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        sample = next(iter(ds))
        logger.info(f"Keys: {sample.keys()}")
        print(f"--- SAMPLE START ({dataset_name}) ---")
        # Print first few chars of value or structure
        for k, v in sample.items():
            print(f"KEY: {k}")
            print(f"TYPE: {type(v)}")
            str_v = str(v)
            if len(str_v) > 500:
                print(f"VALUE: {str_v[:500]}... [TRUNCATED]")
            else:
                print(f"VALUE: {v}")
        print(f"--- SAMPLE END ---")
    except Exception as e:
        logger.error(f"Failed to load: {e}")

if __name__ == "__main__":
    inspect("Ardea/arc_agi_v1")
    inspect("sahil2801/arc-agi-labelled")
