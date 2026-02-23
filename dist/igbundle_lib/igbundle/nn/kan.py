
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KanLinear(nn.Module):
    """
    Fast Kolmogorov-Arnold Network (KAN) Linear Layer.
    Uses Radial Basis Functions (RBF) for efficient spline approximation.
    
    Fixed Dimension Logic:
    grid_size = number of RBF centers.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=8, # Default number of basis functions
        spline_order=3, # Unused for RBF, kept for API compat
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KanLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # 1. Base network (SiLU) - Residual connection
        self.base_activation = base_activation()
        self.scale_base = nn.Parameter(torch.ones(in_features, out_features) * scale_base)

        # 2. Spline network (RBF)
        # uniform grid
        h = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        grid = torch.linspace(grid_range[0], grid_range[1], steps=grid_size)
        
        # (in, grid_size)
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1).contiguous())
        
        # approximate sigma for RBF width
        self.register_buffer("sigma", torch.tensor(h * 1.5)) # overlap factor

        # Spline weights: (out, in, grid_size)
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size)
        )
        
        # Scaling for spline contribution
        self.scale_spline = nn.Parameter(torch.ones(in_features, out_features) * scale_spline)

        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise=0.1):
        # Initialize spline weights with random noise
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * scale_noise)

    def b_splines(self, x: torch.Tensor):
        """
        Compute RBF bases for input x.
        """
        # Unwrap arbitrary batch dims
        x_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # x_flat: (N, in)
        # grid: (in, grid_size)
        
        # Expand for broadcasting
        # x_expanded: (N, in, 1)
        x_exp = x_flat.unsqueeze(-1)
        
        # grid_expanded: (1, in, grid_size)
        grid_exp = self.grid.unsqueeze(0)
        
        # RBF: exp( - ((x - c) / sigma)^2 )
        # (N, in, grid_size)
        bases = torch.exp( - ((x_exp - grid_exp) / self.sigma).pow(2) )
        
        return bases

    def forward(self, x: torch.Tensor):
        # x: (..., in)
        input_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # 1. Base Branch
        # x_silu = silu(x)
        x_acts = self.base_activation(x_flat) 
        # (N, in) * (in, out) -> (N, out)
        base_output = torch.einsum('bi,io->bo', x_acts, self.scale_base)
        
        # 2. Spline Branch
        # bases: (N, in, grid_size)
        bases = self.b_splines(x_flat) 
        
        # weights: (out, in, grid_size)
        # spline_term: sum over grid(g). Output shape (N, in, out)
        # 'big,oig->bio'
        spline_term = torch.einsum('big,oig->bio', bases, self.spline_weight)
        
        # Scale and sum over input dimension (i)
        # spline_output: (N, out)
        # 'bio,io->bo'
        spline_output = torch.einsum('bio,io->bo', spline_term, self.scale_spline)
        
        output_flat = base_output + spline_output
        
        # Reshape back based on input batch dimensions
        output_shape = input_shape[:-1] + (self.out_features,)
        return output_flat.view(output_shape)
