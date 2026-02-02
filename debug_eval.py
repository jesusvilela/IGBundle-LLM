
import sys
import os

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    import evaluate
    print(f"evaluate imported from: {evaluate.__file__}")
    print(f"evaluate version: {evaluate.__version__}")
    
    em = evaluate.load("exact_match")
    print("evaluate.load('exact_match') success")
except Exception as e:
    print(f"evaluate check failed: {e}")

try:
    import lm_eval
    print(f"lm_eval imported from: {lm_eval.__file__}")
    print(f"lm_eval version: {getattr(lm_eval, '__version__', 'unknown')}")
except Exception as e:
    print(f"lm_eval check failed: {e}")
