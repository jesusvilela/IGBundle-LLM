def main():
    with open("available_tasks_list.txt", "r") as f:
        tasks = [line.strip() for line in f]
    
    keywords = ["mmlu", "aime", "gpqa", "arc", "truthfulqa", "math", "gsm8k", "code"]
    
    found = {}
    for k in keywords:
        found[k] = [t for t in tasks if k in t]
        
    for k, v in found.items():
        print(f"--- {k} ({len(v)}) ---")
        # Print top 10
        for i in v[:20]:
            print(i)
        print("...")

if __name__ == "__main__":
    main()
