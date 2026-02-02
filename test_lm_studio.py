
import os
import json
import requests
import random
import glob
import logging

# Config
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
DATA_PATH = os.path.join("data", "ARC-AGI-2", "data", "training")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LMStudioTest")

def chat_complete(messages, temperature=0.7):
    try:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
            "stream": False
        }
        response = requests.post(LM_STUDIO_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None

def qualitative_check():
    logger.info("--- Qualitative Check ---")
    prompt = "Who are you and what is your purpose?"
    logger.info(f"User: {prompt}")
    reply = chat_complete([{"role": "user", "content": prompt}])
    logger.info(f"Model: {reply}")
    
    prompt = "Explain the concept of Hyperbolic Geometry in simple terms."
    logger.info(f"User: {prompt}")
    reply = chat_complete([{"role": "user", "content": prompt}])
    logger.info(f"Model: {reply}\n")

def grid_to_text(grid):
    return str(grid).replace(' ', '')

def quantitative_arc_test(n=5):
    logger.info(f"--- Quantitative ARC Test (n={n}) ---")
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data path not found: {DATA_PATH}")
        return

    json_files = glob.glob(os.path.join(DATA_PATH, "*.json"))
    selected = random.sample(json_files, min(n, len(json_files)))
    
    correct = 0
    total = 0
    
    for jf in selected:
        with open(jf, 'r') as f:
            task = json.load(f)
        
        # Format task
        train_pairs = task['train']
        test_pair = task['test'][0] # Check first test case
        
        prompt_text = "The user will provide input/output grids. Pattern match the rule and solve the final input.\n\n"
        for i, pair in enumerate(train_pairs):
            prompt_text += f"Example {i+1}:\nInput: {grid_to_text(pair['input'])}\nOutput: {grid_to_text(pair['output'])}\n\n"
        
        prompt_text += f"Test:\nInput: {grid_to_text(test_pair['input'])}\nOutput:"
        
        # Send
        reply = chat_complete([
            {"role": "system", "content": "You are an abstract reasoning solver."},
            {"role": "user", "content": prompt_text}
        ], temperature=0.1)
        
        if reply:
            # Simple check: does the reply contain the exact output grid string?
            expected = grid_to_text(test_pair['output'])
            # Normalize whitespace
            reply_clean = reply.replace(" ", "").replace("\n", "")
            expected_clean = expected.replace(" ", "")
            
            is_match = expected_clean in reply_clean
            status = "PASS" if is_match else "FAIL"
            if is_match: correct += 1
            
            logger.info(f"Task {os.path.basename(jf)}: {status}")
            logger.info(f"  Expected: {expected}")
            logger.info(f"  Got (truncated): {reply[:100]}...")
        else:
            logger.info(f"Task {os.path.basename(jf)}: ERROR (No response)")
            
        total += 1
        
    logger.info(f"Results: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    qualitative_check()
    quantitative_arc_test()
