
import torch
import torch.nn as nn
import torch.optim as optim

class EqPropTrainer:
    """
    Trainer for FHNIntegrator using Equilibrium Propagation (EqProp).
    
    Training Phases:
    1. Free Phase: Run dynamics with input x. Reaches fixed point q_free.
    2. Nudged Phase: Run dynamics with input x + beta*(y - q). Reaches q_nudged.
    3. Update: Delta_W ~ -(1/beta) * ( dE(q_nudged)/dW - dE(q_free)/dW )
       For Hebbian energy E ~ -qWq, this becomes (q_nudged * q_nudged^T - q_free * q_free^T).
    """
    def __init__(self, model, beta=0.5, lr=0.01):
        """
        Args:
            model: FHNIntegrator instance.
            beta: Nudging strength.
            lr: Learning rate for coupling weights.
        """
        self.model = model
        self.beta = beta
        self.lr = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def simulate_phase(self, initial_state, input_drive, target=None, steps=50):
        """
        Runs simulation. If target is provided, applies nudge force.
        
        Nudge Force: beta * (target - q)
        """
        q, p = initial_state
        trajectory = []
        
        with torch.no_grad(): # EqProp calculates gradients manually/physically
            for _ in range(steps):
                # Calculate Nudge (if clamped phase)
                nudge = 0.0
                if target is not None:
                    nudge = self.beta * (target - q)
                
                # Total Drive
                total_drive = input_drive + nudge
                
                # Step
                q, p = self.model.step((q, p), total_drive)
                trajectory.append(q)
                
        return torch.stack(trajectory), (q, p)

    def train_step(self, input_pattern, target_pattern, steps=100):
        """
        Performs one EqProp update step.
        """
        batch_size = input_pattern.shape[0]
        
        # Initial State (Rest)
        q0 = torch.zeros_like(input_pattern)
        p0 = torch.zeros_like(input_pattern)
        
        # 1. Free Phase
        # Run until equilibrium (approx)
        _, (q_free, p_free) = self.simulate_phase((q0, p0), input_pattern, target=None, steps=steps)
        
        # 2. Nudged (Clamped) Phase
        # Continue from free state, seeing where the "truth" pulls us
        _, (q_nudged, p_nudged) = self.simulate_phase((q_free, p_free), input_pattern, target=target_pattern, steps=steps//2)
        
        # 3. Compute Gradient (Contrastive Hebbian)
        # We want to lower energy of q_nudged and raise energy of q_free.
        # E_interaction = -0.5 * q^T J q
        # dE/dJ = -0.5 * q^T q (outer product)
        # Update rule: J += lr * (1/beta) * (q_nudged * q_nudged^T - q_free * q_free^T)
        # Scale by 1/beta per Scellier & Bengio
        
        factor = 1.0 / self.beta
        
        # Compute outer products for batch
        # shape: [B, N] -> [B, N, N]
        out_nudged = torch.bmm(q_nudged.unsqueeze(2), q_nudged.unsqueeze(1))
        out_free = torch.bmm(q_free.unsqueeze(2), q_free.unsqueeze(1))
        
        grad_estimate = factor * (out_nudged - out_free).mean(dim=0) # Average over batch
        
        # Apply Update (Manual Gradient Ascent on J for Hebbian learning)
        # Or specifically, we want to MINIMIZE energy of nudged?
        # Energy E ~ -qJq. Minimizing E means Increasing qJq correlation.
        # So we ADD to J.
        
        with torch.no_grad():
            self.model.coupling.data += self.lr * grad_estimate
            
            # Optional: Enforce constraints (e.g. zero diagonal output)
            self.model.coupling.data.fill_diagonal_(0)

        loss = torch.mean((q_free - target_pattern)**2).item()
        return loss

