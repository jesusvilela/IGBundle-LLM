import requests
import json
import os

url = "http://localhost:11434/v1/completions"
# Use the alias set in run_benchmarks_server.py or the real model name if known
# Creating a Modelfile usually sets the model name. 
# We'll try 'igbundle-bench' based on previous context, and 'davinci-002'.
# But typically user sends model="davinci-002" in lm_eval which maps to... what?
# lm_eval model_args="model=davinci-002" just passes "model": "davinci-002" in the JSON body.
# We need to make sure Ollama has a model named "davinci-002" OR we rely on Ollama aliasing?
# Actually, run_benchmarks_server.py sets model_args but doesn't seem to CREATE an alias named 'davinci-002'.
# Wait! In Step 3940 log: "--model_path ...igbundle.gguf".
# The server (Ollama) is launched. 
# What model is LOADED?
# If we used 'ollama serve', we need to have done 'ollama create'.
# Let's try sending "davinci-002" and see if Ollama accepts.

payload = {
    "model": "davinci-002", 
    "prompt": "The answer is",
    "max_tokens": 1,
    "logprobs": 5, # lm_eval asks for top logprobs
    "echo": True,   # lm_eval often uses echo for likelihood
    "stream": False
}

print(f"Sending to {url} with payload: {payload}")
try:
    res = requests.post(url, json=payload)
    print(f"Status: {res.status_code}")
    print(f"Body: {res.text[:1000]}") # Print first 1000 chars
except Exception as e:
    print(f"Error: {e}")
