
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading dataset...")
    custom_dataset_path = "unrestricted_long_context_data.jsonl"
    data = load_dataset('json', data_files=custom_dataset_path, split='train')
    
    print(f"Dataset columns: {data.column_names}")
    print(f"Number of examples: {len(data)}")
    
    # Simulate load_dataset_for_training logic from trainv2.py
    if "text" not in data.column_names:
        print("Formatting text...")
        # (Simplified formatting logic from trainv2.py)
        pass 
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    print("Tokenizing...")
    tokenized_data = data.map(tokenize_function, batched=True, remove_columns=data.column_names)
    
    print(f"Tokenized columns: {tokenized_data.column_names}")
    
    if len(tokenized_data) > 0:
        example = tokenized_data[0]
        if "input_ids" in example:
            ids = example["input_ids"]
            print(f"Input IDs length: {len(ids)}")
            print(f"Input IDs sample: {ids[:20]}...")
            
            # Check for non-padding
            non_pad = [x for x in ids if x != tokenizer.pad_token_id]
            print(f"Number of non-pad tokens: {len(non_pad)}")
            
            # Check for labels
            if "labels" in example:
                 print(f"Labels: {example['labels'][:20]}")
            else:
                 print("No labels in tokenized data (Expected for default tokenizer map)")
                 
    # Simulate batch creation
    indices = [0]
    batch_texts = [tokenized_data[i]["input_ids"] for i in indices]
    max_len = min(512, max(len(text) for text in batch_texts))
    
    batch = torch.zeros(1, max_len, dtype=torch.long)
    for i, text in enumerate(batch_texts):
        length = min(len(text), max_len)
        batch[i, :length] = torch.tensor(text[:length])
        
    print(f"Batch sample: {batch[0, :20]}")
    
    # Simulate loss calculation inputs
    logits_mock = torch.randn(1, 512, 100) # Mock
    shift_targets = batch[..., 1:].contiguous()
    
    print(f"Shift Targets sample: {shift_targets[0, :20]}")
    
    # Check if target contains only 0s and if that's valid
    unique_targets = torch.unique(shift_targets)
    print(f"Unique targets in batch: {unique_targets}")
    
    if tokenizer.pad_token_id in unique_targets:
        print(f"Pad token {tokenizer.pad_token_id} is present in targets.")
        
    # Check if 0 is in targets
    if 0 in unique_targets:
        print("Token ID 0 is present in targets.")

if __name__ == "__main__":
    main()
