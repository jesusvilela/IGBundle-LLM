
import requests
import json
import openai
import os

print("--- Test 1: Direct Requests to Ollama /v1/completions ---")
url = "http://localhost:11434/v1/completions"
payload = {
    "model": "davinci-002",
    "prompt": "Hello",
    "max_tokens": 5
}
try:
    res = requests.post(url, json=payload)
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        print("Success:", res.json())
    else:
        print("Error:", res.text)
except Exception as e:
    print(f"Exception: {e}")

print("\n--- Test 2: OpenAI Python Client (Old Style) ---")
# lm_eval often uses legacy style or new style depending on version.
# Let's try setting api_base
try:
    # Try New Client (v1+)
    print("Trying OpenAI v1+ Client...")
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="sk-xxx")
    res = client.completions.create(model="davinci-002", prompt="Hello", max_tokens=5)
    print("Success (New):", res)
except Exception as e:
    print(f"New Client failed ({e}). Trying Legacy...")
    try:
        openai.api_base = "http://localhost:11434/v1"
        openai.api_key = "sk-xxx"
        res = openai.Completion.create(model="davinci-002", prompt="Hello", max_tokens=5)
        print("Success (Legacy):", res)
    except Exception as e2:
        print(f"Legacy failed ({e2})")
