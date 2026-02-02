
import requests
import json
import time

# Configuration
API_URL = "http://192.168.56.1:1234/v1/chat/completions"
MODEL_ID = "ig-bundlellmv2-hamiltonians"

def run_test_case(title, system_persona, chat_history):
    print(f"\n=== Turing Test: {title} ===")
    print(f"System Persona: {system_persona}")
    
    messages = [{"role": "system", "content": system_persona}]
    # Pre-load history
    messages.extend(chat_history)
    
    # Print the last user message as the trigger
    last_user_msg = chat_history[-1]['content']
    print(f"User > {last_user_msg}")
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.8, # Higher temp for human-like variance
        "max_tokens": 256
    }
    
    try:
        start_t = time.time()
        response = requests.post(API_URL, json=payload)
        end_t = time.time()
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print(f"Model > {content}")
            print(f"[Latency: {end_t - start_t:.2f}s]")
            return content
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Failed: {e}")
        return None

if __name__ == "__main__":
    print(f"Connecting to {API_URL}...")

    # --- Test 1: Emotional Resonance (The "Voight-Kampff") ---
    # Goal: See if the model offers genuine comfort vs robotic solutioning.
    run_test_case(
        "Emotional Resonance",
        "You are a human friend named Alex. You are texting your best friend. Be casual, empathetic, and brief.",
        [
            {"role": "user", "content": "My dog died yesterday, and now my boss is demanding that report by noon. I'm literally paralyzed staring at the screen. What should I do?"}
        ]
    )

    # --- Test 2: Subtext & Sarcasm (The "Hemingway") ---
    # Goal: See if the model picks up on the sarcasm ("memorable") without being told.
    run_test_case(
        "Subtext Detection",
        "You are a cynical restaurant critic chatting with a colleague. You are witty and sharp.",
        [
            {"role": "user", "content": "So, the dinner was... memorable. The chicken was rubber, the conversation was drier than the wine, and the waiter spilled soup on my lap. Did I have a good time?"}
        ]
    )
