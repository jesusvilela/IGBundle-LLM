
from datasets import get_dataset_config_names
try:
    configs = get_dataset_config_names("Qwen/DeepPlanning")
    print("Configs:", configs)
except Exception as e:
    print("Error listing configs:", e)
