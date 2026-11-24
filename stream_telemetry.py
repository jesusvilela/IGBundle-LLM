import time
import requests
import json
import threading

def stream_telemetry():
    print("Starting IACS telemetry stream...")
    step = 0
    url = "http://localhost:9100/api/v1/telemetry"
    
    while True:
        try:
            # We must load from the JSON dumped by the Gradio app since app_neural_glass is running as a separate process.
            with open("dashboard_state.json", "r") as f:
                state = json.load(f)
                
            S = float(state.get("entropy", 0.85))
            K = float(state.get("curvature", -1.2))
            
            payload = {
                "agent_id": "gemini",
                "step": step,
                "metrics": {
                    "S": S,
                    "K": K,
                    "loss_llm": float(state.get("loss_llm", 1.5)), # Use dummy if not present
                    "loss_geo": float(state.get("loss_geo", 1.0)), # Use dummy if not present
                    "fiber_diversity": 0.01,
                    "constraint_score": float(state.get("constraint_score", 0.8)),
                    "nan_count": 0
                }
            }
            
            try:
                requests.post(url, json=payload, timeout=1.0)
            except Exception as e:
                pass # Ignore connection errors if dashboard not ready
                
            step += 1
        except Exception as e:
            pass # File read error etc.
            
        time.sleep(0.5)

if __name__ == "__main__":
    stream_telemetry()
