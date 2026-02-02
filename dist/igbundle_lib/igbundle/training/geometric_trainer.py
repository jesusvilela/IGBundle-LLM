import torch
from transformers import Trainer
import torch.nn as nn
from typing import Dict, Union, Any, Optional

class GeometricTrainer(Trainer):
    """
    Geometric-aware Trainer for Supervised Fine-Tuning.
    Adds Hamiltonian and Sheaf consistency losses to the standard training loop.
    """
    def __init__(self, geometric_gamma: float = 0.1, sheaf_loss_fn=None, lambda_glue=0.1, state_collector=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometric_gamma = geometric_gamma
        self.sheaf_loss_fn = sheaf_loss_fn
        self.lambda_glue = lambda_glue
        self.state_collector = state_collector
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute total loss = Task Loss + Gamma * Geometric Loss
        """
        # 1. Standard Task Loss (Next Token Prediction)
        # We need to handle num_items_in_batch if using newer transformers
        if num_items_in_batch is not None:
             task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
             task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
             
        # 2. Extract Geometric State
        # The adapter output should be attached to the model output or accessible via state_collector?
        # Our wrap_hf_candidate attaches 'ig_state' to hidden states but standard CausalLMOutput doesn't carry it easily.
        # BUT, the GeometricIGBundleAdapter computes losses internally and adds them to a 'losses' dict if called?
        # Actually, in `test_consensus.py`, we saw `adapter.compute_geometric_losses(state)`.
        
        # We need to access the state.
        # If we use the `StateCollector` callback approach, the state is in `self.state_collector`.
        # Taking the last state?
        geo_loss = 0.0
        
        # Check if model has direct access to the last state (thread unsafe if parallel?)
        # Better: The model return might include it if we modified the headers, but we usually didn't.
        
        # Fallback: We can modify the model to return the auxiliary loss directly in the forward pass
        # and add it to the CrossEntropy loss.
        # If `GeometricIGBundleAdapter` is integrated properly, it modifies the output.
        # But `Trainer` computes loss on `logits` vs `labels`.
        
        # Let's inspect if `outputs` contains our custom extras.
        # If `outputs` is a tuple/CausalLMOutput, we might find `ig_state` if we hacked the model output.
        
        # If we can't find it, we skip for now (Standard SFT) but we really want Epic 7 verification.
        # Wait! In `geometric_adapter.py`, we return `(hidden, geo_state)`.
        # The wrapper in `hf_patch.py`:
        # returns `(h_new,) + outputs[1:]`
        # It also does: `setattr(h_new, 'ig_state', state)`
        
        # The `Trainer` gets `outputs` from `model`.
        # `model` returns `CausalLMOutputWithPast`.
        # `outputs.hidden_states` might contain the hidden states if `output_hidden_states=True`.
        # If so, the last hidden state should have `.ig_state`.
        
        if self.geometric_gamma > 0.0 and outputs is not None:
             # Try to find ig_state
             # Check hidden_states
             hidden_states = outputs.get("hidden_states") if isinstance(outputs, dict) else (outputs.hidden_states if hasattr(outputs, "hidden_states") else None)
             
             state = None
             if hidden_states:
                 last_hidden = hidden_states[-1]
                 if hasattr(last_hidden, "ig_state"):
                     state = last_hidden.ig_state
                     
             # Compute Aux Loss if state found
             if state:
                 # We need access to the adapter to compute losses?
                 # Or we can compute them here if we import the loss functions?
                 # Adapter has `compute_geometric_losses`.
                 # We need to find the adapter instance.
                 adapter = None
                 if hasattr(model, "adapter"): adapter = model.adapter
                 elif hasattr(model, "module") and hasattr(model.module, "adapter"): adapter = model.module.adapter
                 
                 if adapter:
                     aux_losses = adapter.compute_geometric_losses(state)
                     # Sum them up
                     total_aux = sum(aux_losses.values())
                     
                     geo_loss = total_aux * self.geometric_gamma
                     
                     # Log terms for debugging (every 1 step if necessary, or 10)
                     if self.state.global_step % 5 == 0:
                        print(f"\n[Step {self.state.global_step}] Task Loss: {task_loss.item():.4f} | Aux Loss: {total_aux.item():.4f} (Scaled: {geo_loss.item():.4f})")
                        for k, v in aux_losses.items():
                            print(f"   - {k}: {v.item():.4f}")
        
        return task_loss + geo_loss
