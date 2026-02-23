
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

class EquilibriumPropagator:
    """
    Implements the Equilibrium Propagation learning framework for FHN systems.
    
    Principles:
    1. Gradients are defined by the difference between two equilibrium states:
       - Free State (Prediction): Settling under input drive.
       - Clamped State (Observation): Settling under input + output nudge.
    
    2. Update Rule (Contrastive Hebbian):
       Delta_J ~ (1/beta) * (Out_Clamped * Out_Clamped^T - Out_Free * Out_Free^T)
    """
    def __init__(self, model: nn.Module, beta: float = 0.5, lr: float = 0.01):
        """
        Args:
            model: The dynamic system (e.g., FHNIntegrator). Must have .step() and .coupling parameter.
            beta: Nudging strength (the 'pressure' to match target).
            lr: Learning rate for the plasticity update.
        """
        self.model = model
        self.beta = beta
        self.lr = lr
        
        # We assume the model has a coupling parameter that we want to learn
        # If the model has other optimizable params, we could use a proper optimizer.
        # For strict Hebbian learning, we might do manual updates.
        # Let's use SGD for flexibility.
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def run_phase(self, initial_state, input_drive, target=None, steps=50, return_trajectory=False):
        """
        Runs the dynamics for a fixed number of steps.
        
        Args:
            initial_state: Tuple (q, p)
            input_drive: Tensor [B, N]
            target: Optional Tensor [B, N]. If provided, applies nudge.
        """
        q, p = initial_state
        trajectory = []
        
        # We run in no_grad mode because EqProp replaces backprop.
        # The 'gradient' is computed physically.
        with torch.no_grad():
            for _ in range(steps):
                # Calculate Nudge (Clamped Phase Force)
                # F_nudge = - beta * d(Energy_nudge)/dq
                # Energy_nudge = 0.5 * ||q - target||^2
                # dE/dq = q - target
                # F_nudge = - beta * (q - target) = beta * (target - q)
                nudge = 0.0
                if target is not None:
                    nudge = self.beta * (target - q)
                
                total_drive = input_drive + nudge
                
                q, p = self.model.step((q, p), total_drive)
                
                if return_trajectory:
                    trajectory.append(q)
                    
        final_state = (q, p)
        if return_trajectory:
            return torch.stack(trajectory), final_state
        return None, final_state

    def learn_step(self, input_pattern, target_pattern, free_steps=60, clamped_steps=30):
        """
        Performs one full learning cycle:
        1. Free Phase -> q_free
        2. Clamped Phase (starting from q_free) -> q_clamped
        3. Weight Update based on q_clamped vs q_free
        
        Returns:
            loss: Squared error between Free Prediction and Target.
        """
        batch_size = input_pattern.shape[0]
        
        # Initial State (Rest)
        q0 = torch.zeros_like(input_pattern)
        p0 = torch.zeros_like(input_pattern)
        
        # --- 1. Free Phase ---
        _, (q_free, p_free) = self.run_phase((q0, p0), input_pattern, target=None, steps=free_steps)
        
        # --- 2. Clamped (Nudged) Phase ---
        # Continue from free equilibrium
        # This is strictly correct for EqProp (Infinitesimal Nudge off equilibrium)
        _, (q_clamped, p_clamped) = self.run_phase((q_free, p_free), input_pattern, target=target_pattern, steps=clamped_steps)
        
        # --- 3. Compute Update ---
        # Update rule derived from Scellier & Bengio (2017):
        # Grad_Theta ~ (1/beta) * ( dE(clamped)/dTheta - dE(free)/dTheta )
        # For Hebbian coupling J: Energy term is -0.5 * q^T J q
        # dE/dJ = -0.5 * q^T q (outer product)
        # Delta J ~ (1/beta) * [ (-0.5 * q_c * q_c^T) - (-0.5 * q_f * q_f^T) ]
        # Delta J ~ (0.5/beta) * [ q_f * q_f^T - q_c * q_c^T ]
        # Wait, usually we want to DECREASE Energy of Clamped state.
        # So we move J in direction of NEGATIVE gradient of Clamped energy?
        # Standard Hebbian: "Fire together, wire together".
        # We want J to support the Clamped state (Truth).
        # So J should increase for correlated components in Clamped phase.
        # Update: J += lr * (q_clamped * q_clamped^T - q_free * q_free^T)
        
        factor = 1.0 / self.beta
        
        # Compute correlations
        # [B, N] -> [B, N, N]
        corr_clamped = torch.bmm(q_clamped.unsqueeze(2), q_clamped.unsqueeze(1))
        corr_free = torch.bmm(q_free.unsqueeze(2), q_free.unsqueeze(1))
        
        # Average over batch
        # We want (Clamped - Free)
        grad_estimate = factor * (corr_clamped - corr_free).mean(dim=0)
        
        # Apply Update
        # Note: This is simplified manual update. 
        # For deep networks, we'd accumulate .grad into parameters and step optimizer.
        # Here we directly modify the coupling matrix for the prototype.
        
        with torch.no_grad():
            if hasattr(self.model, 'coupling'):
                self.model.coupling.data += self.lr * grad_estimate
                # Enforce zero self-coupling to prevent runaway
                self.model.coupling.data.fill_diagonal_(0)
                
        # Metric: how far was free state from target?
        error = torch.mean((q_free - target_pattern)**2).item()
        return error
