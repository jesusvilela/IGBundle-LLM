import json
import matplotlib.pyplot as plt
import os
import time

def plot_loss(log_dir="output/igbundle_qwen7b"):
    # Locate latest checkpoint or trainer_state
    state_path = os.path.join(log_dir, "trainer_state.json")
    
    # If partial, check checkpoints
    if not os.path.exists(state_path):
        # Look for checkpoints
        checkpoints = [d for d in os.listdir(log_dir) if d.startswith("checkpoint")]
        if not checkpoints:
            print("No checkpoints found yet.")
            return
        
        # Sort by step
        checkpoints.sort(key=lambda x: int(x.split('-')[1]))
        latest = checkpoints[-1]
        state_path = os.path.join(log_dir, latest, "trainer_state.json")
    
    if not os.path.exists(state_path):
        print(f"Could not find trainer_state.json in {state_path}")
        return

    with open(state_path, 'r') as f:
        data = json.load(f)
        
    history = data.get('log_history', [])
    
    steps = []
    losses = []
    
    for entry in history:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            
    if not steps:
        print("No loss data found yet.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('IGBundle Training Convergence')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss.png')
    print(f"Plot saved to training_loss.png. Current Loss at step {steps[-1]}: {losses[-1]}")

if __name__ == "__main__":
    plot_loss()
