import lm_eval
from lm_eval.tasks import TaskManager
import json

def main():
    tm = TaskManager()
    tasks = tm.all_tasks
    print(f"Found {len(tasks)} tasks.")
    
    with open("available_tasks_list.txt", "w") as f:
        for t in sorted(tasks):
            f.write(t + "\n")
    
    print("Tasks saved to available_tasks_list.txt")

if __name__ == "__main__":
    main()
