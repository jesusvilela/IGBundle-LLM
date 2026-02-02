import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GeometricPhaseMemory(nn.Module):
    """
    Implements Path-Dependent Context via Geometric (Berry) Phase.
    
    Tracks the accumulation of 'Phase' (Holonomy) as the thought vector
    traverses the hyperbolic manifold.
    
    The phase is computed as the symplectic area swept by the path.
    Phase = Integral of Connection 1-form A.
    On Poincaré disk D: A = (2 * Im(z_bar * dz)) / (1 - |z|^2)
    
    This phase acts as a 'Context' that modulates the memory state.
    """
    def __init__(self, dim: int, max_history: int = 128):
        super().__init__()
        self.dim = dim
        self.max_history = max_history
        
        # Memory Vector (learnable or persistent state)
        # Represents the "Self" concept that gets rotated by experience
        self.memory_vector = nn.Parameter(torch.randn(1, dim) * 0.1)
        
        # Gating mechanism to decide how much Phase influences output
        self.phase_gate = nn.Linear(dim, dim)
        
    def compute_symplectic_area_step(self, prev_z: torch.Tensor, curr_z: torch.Tensor) -> torch.Tensor:
        """
        Computes the symplectic area element for a step z_{t-1} -> z_t.
        Approximates the connection 1-form integral.
        
        z is treated as complex numbers (pairs of dimensions).
        We reshape (..., D) -> (..., D/2, 2).
        """
        # 1. Reshape to complex-like pairs
        # (B, D) -> (B, D/2, 2)
        # We assume D is even. If odd, ignore last dim or pad.
        B, D = prev_z.shape
        if D % 2 != 0:
            # Pad with 0
            prev_z = F.pad(prev_z, (0, 1))
            curr_z = F.pad(curr_z, (0, 1))
            D += 1
            
        p_c = prev_z.view(B, D // 2, 2)
        c_c = curr_z.view(B, D // 2, 2)
        
        # 2. Compute Im(z_bar * dz)
        # dz = c - p
        # z_bar = (x, -y)
        # Im((x - iy)(dx + idy)) = x*dy - y*dx
        # This is essentially the 2D cross product z x dz
        
        x = p_c[..., 0]
        y = p_c[..., 1]
        
        dx = c_c[..., 0] - x
        dy = c_c[..., 1] - y
        
        cross_prod = x * dy - y * dx
        
        # 3. Metric Factor: 2 / (1 - |z|^2)
        # Use midpoint for stability
        mid_z = (prev_z + curr_z) / 2
        mid_norm_sq = (mid_z ** 2).sum(dim=-1, keepdim=True).clamp(max=0.99)
        # Broadcast norm to (B, D/2) needed? No, norm is global for the ball usually...
        # Wait, if we have D dimensions, is it one ball or Product of disks?
        # Our adapter treats it variously. 
        # If 'Product of Disks' (P), we sum phases.
        # If one big ball, this formula applies to the 2D projection or we need High-D Kahler form.
        # Let's assume Product of D/2 Disks for phase accumulation.
        # So we use component-wise norms.
        
        mid_c = mid_z.view(B, D // 2, 2)
        mid_norm_sq_c = (mid_c ** 2).sum(dim=-1) # (B, D/2)
        
        metric_factor = 2.0 / (1.0 - mid_norm_sq_c + 1e-6)
        
        # 4. Phase Contribution
        d_phase = metric_factor * cross_prod # (B, D/2)
        
        return d_phase

    def forward(self, coords_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords_sequence: (B, T, P, D)
            
        Returns:
            contextual_memory: (B, T, D) - Memory modulated by phase
            total_phase: (B, T, P, D/2) - Accumulated phase
        """
        B, T, P, D = coords_sequence.shape
        
        # Flatten P into Batch for parallel calc
        coords = coords_sequence.view(B * P, T, D)
        
        # Calculate diffs
        prev_coords = coords[:, :-1, :]
        curr_coords = coords[:, 1:, :]
        
        # Initial phase is 0
        phases = [torch.zeros(B * P, D // 2, device=coords.device)]
        
        # Scan loop (could be optimized with cumsum if metric factor was constant, but it's not)
        # For efficiency in Python, we might just compute all steps parallel?
        # Yes! 
        # But metric factor depends on position.
        # We can compute all (t, t-1) pairs in parallel.
        
        # Vectorized Phase Calc
        # prev: (BP, T-1, D)
        # curr: (BP, T-1, D)
        
        # Reshape to complex
        if D % 2 != 0:
            prev_complex = F.pad(prev_coords, (0, 1)).view(B*P, T-1, (D+1)//2, 2)
            curr_complex = F.pad(curr_coords, (0, 1)).view(B*P, T-1, (D+1)//2, 2)
            eff_D = D + 1
        else:
            prev_complex = prev_coords.view(B*P, T-1, D//2, 2)
            curr_complex = curr_coords.view(B*P, T-1, D//2, 2)
            eff_D = D
            
        x = prev_complex[..., 0]
        y = prev_complex[..., 1]
        dx = curr_complex[..., 0] - x
        dy = curr_complex[..., 1] - y
        
        cross_prod = x * dy - y * dx
        
        # Metric
        mid_complex = (prev_complex + curr_complex) / 2
        mid_norm_sq = (mid_complex ** 2).sum(dim=-1) # (BP, T-1, D/2)
        metric_factor = 2.0 / (1.0 - mid_norm_sq.clamp(max=0.99) + 1e-6)
        
        step_phases = metric_factor * cross_prod # (BP, T-1, D/2)
        
        # Cumulative Sum
        cum_phases = torch.cumsum(step_phases, dim=1) # (BP, T-1, D/2)
        
        # Prepend 0 for t=0
        zeros = torch.zeros(B*P, 1, eff_D//2, device=coords.device)
        all_phases = torch.cat([zeros, cum_phases], dim=1) # (BP, T, D/2)
        
        # Apply Phase Rotation to Memory Vector
        # Memory is (1, D). We treat it as pairs of complex numbers too.
        mem_complex = self.memory_vector.view(1, 1, eff_D//2, 2)
        if eff_D > D: # Handle padding for memory
             mem_complex = F.pad(self.memory_vector, (0, 1)).view(1, 1, eff_D//2, 2)
        
        # Rotate: z' = z * e^(i*theta)
        # x' = x cos(theta) - y sin(theta)
        # y' = x sin(theta) + y cos(theta)
        
        cos_theta = torch.cos(all_phases) # (BP, T, D/2)
        sin_theta = torch.sin(all_phases)
        
        mx = mem_complex[..., 0] # (1, 1, D/2) broadcast to (BP, T, D/2)
        my = mem_complex[..., 1]
        
        mx_rot = mx * cos_theta - my * sin_theta
        my_rot = mx * sin_theta + my * cos_theta
        
        mem_rotated = torch.stack([mx_rot, my_rot], dim=-1).view(B*P, T, eff_D)
        
        if eff_D > D:
            mem_rotated = mem_rotated[..., :D]
            
        # Reshape back to (B, T, P, D)
        mem_rotated = mem_rotated.view(B, T, P, D)
        
        # Aggregate across P (Components) - Maybe sum?
        # "Context" usually should be single vector per T.
        # Let's sum across P components.
        context_out = mem_rotated.mean(dim=2) # (B, T, D)
        
        # Gate
        context_out = self.phase_gate(context_out)
        
        return context_out, all_phases.view(B, T, P, eff_D//2)
