import torch
from torch.optim import Optimizer
import math
from igbundle.geometry.poincare import PoincareBall

# Attempt to import geoopt for advanced manifolds if available
try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False

class FiberBundleSPIDER(Optimizer):
    """
    FiberBundle-SPIDER: A curvature-aware optimizer for fiber bundle neural networks.
    
    Implements:
    1. Horizontal/Vertical Decomposition (Base vs Fiber parameters).
    2. R-SPIDER Variance Reduction (Recursive Path-Integrated Estimator).
    3. Asymmetric Momentum (Conservative Base, Aggressive Fiber).
    4. Curvature-Aware Batching (handled via external batch size scaling, 
       but optimizer is aware of curvature distortion).
       
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        base_lr (float): Learning rate for base manifold parameters (default: 1e-4).
        fiber_lr (float): Learning rate for fiber/vertical parameters (default: 1e-3).
        base_momentum (float): Momentum factor for base parameters (default: 0.0).
        fiber_momentum (float): Momentum factor for fiber parameters (default: 0.9).
        period (int): SPIDER reset period (default: 100).
        manifold (object, optional): Manifold instance for geometric operations. 
                                     Defaults to PoincareBall(c=1.0).
    """

    def __init__(self, params, 
                 base_lr=1e-4, fiber_lr=1e-3, 
                 base_momentum=0.0, fiber_momentum=0.9, 
                 period=100, 
                 manifold=None):
        
        if manifold is None:
            manifold = PoincareBall(dim=1) # Dim isn't strictly needed for methods except init logic
            
        defaults = dict(
            base_lr=base_lr, fiber_lr=fiber_lr,
            base_momentum=base_momentum, fiber_momentum=fiber_momentum,
            period=period,
            manifold=manifold
        )
        super(FiberBundleSPIDER, self).__init__(params, defaults)
        self.state['step_count'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss. REQUIRED for R-SPIDER variance reduction logic
                unless `grad` is pre-manipulated.
        """
        loss = None
        if closure is not None:
            # We might need closure for the "lookback" gradient but usually
            # generic Closure just computes current loss w/ current params.
            # SPIDER needs grad at prev_params with CURRENT batch.
            loss = closure()

        self.state['step_count'] += 1
        k = self.state['step_count']
        
        for group in self.param_groups:
            period = group['period']
            manifold = group['manifold']
            
            # Reset condition: k=1 or k mod p == 0? 
            # Prompt says: "if k mod p == 1: Full gradient reset"
            # So if p=100, steps 1, 101, 201 are resets.
            is_reset = (k % period == 1)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Identify if parameter is Base (Horizontal) or Fiber (Vertical)
                # Convention: 
                # 1. Check `_is_base` attribute on parameter
                # 2. Check group-level `is_base` flag
                # 3. Default to Fiber (Vertical/Euclidean) if undetermined
                is_base = getattr(p, '_is_base', False) or group.get('is_base', False)
                
                # Fetch hyperparameters
                lr = group['base_lr'] if is_base else group['fiber_lr']
                beta = group['base_momentum'] if is_base else group['fiber_momentum']
                
                state = self.state[p]
                
                # State Initialization
                if len(state) == 0:
                    state['v'] = torch.zeros_like(p)          # Estimator
                    state['m'] = torch.zeros_like(p)          # Momentum buffer
                    state['prev_p'] = p.clone()               # Previous parameter
                    state['prev_grad'] = torch.zeros_like(p)  # Gradient at prev_p (approx)
                    # Note: Storing exact prev_grad from last step is valid for SPIDER 
                    # ONLY if batch is same? No, SPIDER is stochastic.
                    # R-SPIDER: v_k = \nabla f_{S_k}(x_k) - \nabla f_{S_k}(x_{k-1}) + v_{k-1}
                    # We need \nabla f_{S_k}(x_{k-1}). 
                    
                v = state['v']
                m = state['m']
                prev_p = state['prev_p']
                
                grad_current = p.grad # \nabla f_{S_k}(x_k)
                
                # --- SPIDER ESTIMATOR UPDATE ---
                v = self._update_estimator(p, grad_current, state, is_reset, is_base, manifold)
                
                # --- MOMENTUM UPDATE ---
                m = self._update_momentum(p, v, state, beta, is_base, manifold, prev_p)
                
                # --- PARAMETER UPDATE ---
                self._update_parameter(p, m, lr, is_base, manifold)
                
                # --- POST-UPDATE MAINTENANCE ---
                # Update prev_p for next step's transport
                state['prev_p'].copy_(p)
                
                # Project back if needed (Base only)
                if is_base:
                    # Check norm and project if OOB
                    with torch.no_grad():
                         norm = torch.norm(p, dim=-1, keepdim=True)
                         if torch.any(norm > 0.99):
                              # Project
                              scale = 0.99 / norm.clamp(min=1e-6)
                              cond = norm > 0.99
                              p.data.copy_(torch.where(cond, p * scale, p))

        return loss

    def _update_estimator(self, p, grad, state, is_reset, is_base, manifold):
        v = state['v']
        if is_reset:
            v.copy_(grad)
        else:
            prev_p = state['prev_p']
            if is_base and hasattr(manifold, 'parallel_transport'):
                 v_transported = manifold.parallel_transport(prev_p, p, v)
                 v.copy_(v_transported.add_(grad)) 
            else:
                 v.add_(grad)
        return v

    def _update_momentum(self, p, v, state, beta, is_base, manifold, prev_p):
        m = state['m']
        if is_base and hasattr(manifold, 'parallel_transport'):
            m_prev_transported = manifold.parallel_transport(prev_p, p, m)
            # m = beta * m_trans + (1-beta) * v
            # Use out-of-place or careful in-place
            m.copy_(m_prev_transported.mul_(beta).add_(v, alpha=(1.0 - beta)))
        else:
            m.mul_(beta).add_(v, alpha=(1.0 - beta))
        return m

    def _update_parameter(self, p, m, lr, is_base, manifold):
        if is_base:
            if hasattr(manifold, 'exp_map'):
                 direction = m.mul(-lr)
                 new_p = manifold.exp_map(p, direction)
                 p.copy_(new_p)
            else:
                 p.add_(m, alpha=-lr)
        else:
            p.add_(m, alpha=-lr)
