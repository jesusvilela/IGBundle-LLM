
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from trainv2 import load_dataset_for_training, Config 

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Mocking Config...")
    class MockConfig:
        class Training:
            dataset_path = "unrestricted_long_context_data.jsonl"
            bf16 = True
        training = Training()
        
    config = MockConfig()
    
    print("Loading dataset...")
    # This calls the actual function from trainv2.py
    data = load_dataset_for_training(tokenizer, config, dataset_size=10)
    
    print(f"Dataset columns: {data.column_names}")
    print(f"Number of examples: {len(data)}")
    
    if len(data) > 0:
        example = data[0]
        print(f"Example 0 keys: {example.keys()}")
        if "input_ids" in example:
            ids = example["input_ids"]
            print(f"Input IDs length: {len(ids)}")
            print(f"Input IDs sample: {ids[:20]}...")
            print(f"Decoded sample: {tokenizer.decode(ids[:50])}")
            
            # Check for masking (if labels present)
            if "labels" in example:
                labels = example["labels"]
                print(f"Labels length: {len(labels)}")
                print(f"Labels sample: {labels[:20]}...")
        else:
             print("ERROR: input_ids not found in dataset example!")

        if "text" in example:
             print(f"Original Text sample: {example['text'][:100]}...")
             
    # Simulate batch creation logic from run_geometric_training
    indices = [0]
    batch_texts = [data[i]["input_ids"] for i in indices]
    max_len = min(512, max(len(text) for text in batch_texts))
    batch = torch.zeros(1, max_len, dtype=torch.long)
    for i, text in enumerate(batch_texts):
        length = min(len(text), max_len)
        batch[i, :length] = torch.tensor(text[:length])
        
    print(f"Constructed Batch Shape: {batch.shape}")
    print(f"Batch sample: {batch[0, :20]}")
    
    # Check targets logic
    outputs = batch # Mock output = input
    targets = batch
    
    shift_outputs = outputs[..., :-1, :].contiguous() if hasattr(outputs, "shape") and len(outputs.shape)>2 else outputs[..., :-1]
    shift_targets = targets[..., 1:].contiguous()
    
    print(f"Shift Targets sample: {shift_targets[0, :20]}")

if __name__ == "__main__":
    main()
