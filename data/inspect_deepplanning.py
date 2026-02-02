
from datasets import load_dataset

ds = load_dataset("Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b", "stage1", split="train", streaming=True)
print("First example keys:", next(iter(ds)).keys())
print("First example content:", next(iter(ds)))
