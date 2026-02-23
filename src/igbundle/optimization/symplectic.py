import torch
from .spider import FiberBundleSPIDER

class SymplecticSPIDER(FiberBundleSPIDER):
    """
    SymplecticSPIDER: Fiber Bundle Optimizer with Relativistic Adaptive Descent (RAD) mechanics.
    Interprets optimization as the flow of a Conformal Hamiltonian System with 
    Relativistic Kinetic Energy K(p) = c^2 * sqrt(1 + ||p||^2/c^2).
    
    This imposes a "cosmic speed limit" c on the parameter updates, preventing
    instability near hyperbolic boundaries where gradients explode.
    """
    def __init__(self, params, c=1.0, **kwargs):
        """
        Args:
            c (float): Speed of light limit. 
                       If c -> infinity, degrades to standard SPIDER/SGD.
                       If c ~ 1.0, acts as robust gradient clipper/stabilizer.
        """
        super().__init__(params, **kwargs)
        self.c = c

    def _update_parameter(self, p, m, lr, is_base, manifold):
        """
        Relativistic Parameter Update.
        Velocity v = dH/dp = p / sqrt(1 + ||p||^2/c^2).
        """
        # Calculate Relativistic Velocity
        # p_norm_sq = ||m||^2
        m_norm_sq = torch.sum(m * m, dim=-1, keepdim=True)
        
        # Lorentz Factor gamma = sqrt(1 + m^2/c^2)
        # v = m / gamma
        gamma = torch.sqrt(1.0 + m_norm_sq / (self.c**2))
        velocity = m / gamma
        
        # Apply Update using Relativistic Velocity
        if is_base:
            if hasattr(manifold, 'exp_map'):
                 # Exp_p(-lr * v)
                 direction = velocity.mul(-lr)
                 new_p = manifold.exp_map(p, direction)
                 p.copy_(new_p)
            else:
                 p.add_(velocity, alpha=-lr)
        else:
            p.add_(velocity, alpha=-lr)
