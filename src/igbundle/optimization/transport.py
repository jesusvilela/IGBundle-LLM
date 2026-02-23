import torch

def unified_transport(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, manifold=None, is_flat: bool=False) -> torch.Tensor:
    """
    Unified Vector Transport for Fiber Bundle Optimization.
    
    Args:
        v (Tensor): Vector to transport.
        x (Tensor): Source point.
        y (Tensor): Target point.
        manifold (Object): Manifold instance (must implement parallel_transport).
        is_flat (bool): If True, assumes flat bundle (Euclidean/Trivial Transport).
    
    Returns:
        Tensor: Transported vector at y.
    """
    if is_flat:
        # For flat bundles (Euclidean fibers), transport is identity 
        # (assuming trivial global frame).
        # In BuNN, it's O_uv, but if params are in Euclidean representation, it's Identity.
        return v
    
    if manifold is not None and hasattr(manifold, 'parallel_transport'):
        return manifold.parallel_transport(x, y, v)
        
    # Default fallback: Euclidean Identity
    return v
