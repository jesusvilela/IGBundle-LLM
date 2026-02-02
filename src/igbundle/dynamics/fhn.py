
import torch
import torch.nn as nn
import torch.nn.functional as F

class FHNIntegrator(nn.Module):
    """
    Implements a system of coupled FitzHugh-Nagumo oscillators.
    
    State: (q, p) where:
        q = membrane potential (fiber activation)
        p = recovery variable (semantic inertia)
        
    Dynamics:
        dq/dt = q - q^3/3 - p + I_ext + J * h(neighbors)
        dp/dt = epsilon * (q + alpha - beta * p)
    """
    def __init__(self, num_fibers, dt=0.01, epsilon=0.08, alpha=0.7, beta=0.8):
        super().__init__()
        self.num_fibers = num_fibers
        self.dt = dt
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        # Learnable Coupling Matrix J
        # Initialize as sparse or small random values
        self.coupling = nn.Parameter(torch.randn(num_fibers, num_fibers) * 0.01)
        
        # Self-excitation bias (optional intrinsic excitability)
        self.bias = nn.Parameter(torch.zeros(num_fibers))

    def coupling_current(self, q):
        """
        Computes the lateral input from other fibers.
        Current = J @ tanh(q)
        """
        # We use tanh as the activation function h(.)
        return torch.matmul(self.coupling, torch.tanh(q))

    def derivative(self, state, input_drive):
        """
        Computes dq/dt and dp/dt.
        
        Args:
            state: Tuple (q, p)
            input_drive: External input I(t) [Batch, NumFibers]
        """
        q, p = state
        
        # 1. Compute Coupling
        i_couple = self.coupling_current(q.T).T # Handle batch dim if present
        
        # 2. Fast Variable (q) Dynamics
        # dq/dt = q - q^3/3 - p + I_ext + I_couple
        dq_dt = (q - (q ** 3) / 3.0) - p + input_drive + i_couple + self.bias
        
        # 3. Slow Variable (p) Dynamics
        # dp/dt = eps * (q + alpha - beta * p)
        dp_dt = self.epsilon * (q + self.alpha - self.beta * p)
        
        return dq_dt, dp_dt

    def step(self, state, input_drive):
        """
        Performs one RK4 integration step.
        """
        q, p = state
        
        # k1
        dq1, dp1 = self.derivative((q, p), input_drive)
        
        # k2
        dq2, dp2 = self.derivative((q + 0.5 * self.dt * dq1, p + 0.5 * self.dt * dp1), input_drive)
        
        # k3
        dq3, dp3 = self.derivative((q + 0.5 * self.dt * dq2, p + 0.5 * self.dt * dp2), input_drive)
        
        # k4
        dq4, dp4 = self.derivative((q + self.dt * dq3, p + self.dt * dp3), input_drive)
        
        # Update
        q_next = q + (self.dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
        p_next = p + (self.dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        
        q_next = q + (self.dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
        p_next = p + (self.dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        
        # Stability Clamp (Prevent Explosion)
        q_next = torch.clamp(q_next, -20.0, 20.0)
        p_next = torch.clamp(p_next, -20.0, 20.0)
        
        return q_next, p_next

    def simulate(self, initial_state, input_drive_seq):
        """
        Simulates trajectory over time steps.
        
        Args:
            initial_state: (q0, p0)
            input_drive_seq: Tensor [Batch, Time, NumFibers] OR [Batch, NumFibers] (constant drive)
        
        Returns:
            q_trace: [Batch, Time, NumFibers]
        """
        q, p = initial_state
        trajectory = []
        
        # If input is static, repeat it; else iterate
        is_seq = (input_drive_seq.dim() == 3)
        steps = input_drive_seq.shape[1] if is_seq else 100 # default steps for static
        
        for t in range(steps):
            drive = input_drive_seq[:, t, :] if is_seq else input_drive_seq
            q, p = self.step((q, p), drive)
            trajectory.append(q)
            
        return torch.stack(trajectory, dim=1), (q, p)
