import time
import subprocess
import sys

print("Polling for backend installation completion (this may take a few minutes)...")

cmd = [sys.executable, "-c", "import llama_cpp.server; print('Success')"]

while True:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "Success" in result.stdout:
            print("Backend installed successfully!")
            break
    except:
        pass
    time.sleep(5)
    print(".", end="", flush=True)

print("\nLaunching Benchmark Monitor...")
subprocess.run("launch_monitor.bat", shell=True)
