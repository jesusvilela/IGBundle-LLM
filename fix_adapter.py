
import os

file_path = "src/igbundle/modules/geometric_adapter.py"

with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
in_function = False
skip = False

# The corrected function body
correct_body = """    def _symplectic_integrate(self, coords: torch.Tensor, sections: torch.Tensor, metric: RiemannianMetric) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        Perform Symplectic Leapfrog Integration to evolve state (z) and momentum (p).
        \"\"\"
        steps = 4 
        dt = 0.01 # Smaller step for stability
        damping = 0.1 # Friction to prevent explosion
        
        # Initialize Momentum (p)
        p_coords = torch.zeros_like(coords)
        p_sections = torch.zeros_like(sections)
        
        z_coords = coords.clone()
        z_sections = sections.clone()
        
        # Enable grad for Potential calculation
        with torch.enable_grad():
             if not z_coords.requires_grad and torch.is_grad_enabled():
                 z_coords.requires_grad_(True)
             if not z_sections.requires_grad and torch.is_grad_enabled():
                 z_sections.requires_grad_(True)
                 
             try:
                 for step in range(steps):
                     # 1. Compute Forces F = -grad(U)
                     potential = self.compute_hamiltonian_potential(z_coords, z_sections, metric)
                     
                     grads = torch.autograd.grad(potential, [z_coords, z_sections], create_graph=False, retain_graph=False, allow_unused=True)
                     
                     force_coords = -grads[0] if grads[0] is not None else torch.zeros_like(z_coords)
                     force_sections = -grads[1] if grads[1] is not None else torch.zeros_like(z_sections)
                     
                     if torch.isnan(force_coords).any() or torch.isnan(force_sections).any():
                         raise ValueError(f"NaNs in Hamiltonian Forces (Step {step})")

                     # 2. Momentum Update (Half Step)
                     p_coords = p_coords + 0.5 * force_coords * dt
                     p_sections = p_sections + 0.5 * force_sections * dt

                     # 2b. Typed Routing Mask
                     confidence = torch.max(z_sections, dim=-1)[0]
                     mask = (confidence > 0.1).float().unsqueeze(-1)
                     p_coords = p_coords * mask
                     p_sections = p_sections * mask
                     
                     # 2c. Damping
                     p_coords = p_coords * (1.0 - damping)
                     p_sections = p_sections * (1.0 - damping)

                     # 3. Position Update
                     metric_diag = torch.diagonal(metric.metric_tensor, dim1=-2, dim2=-1)
                     metric_diag = torch.clamp(metric_diag, min=0.1)
                     metric_inv_diag = 1.0 / metric_diag
                     
                     p_coords = torch.clamp(p_coords, -1.0, 1.0)
                     p_sections = torch.clamp(p_sections, -1.0, 1.0)

                     z_coords = z_coords + (p_coords * metric_inv_diag) * dt 
                     z_sections = z_sections + p_sections * dt
                     
                     z_coords = torch.clamp(z_coords, -10.0, 10.0)
                     
                     # 4. Momentum Update (Full Step)
                     potential_new = self.compute_hamiltonian_potential(z_coords, z_sections, metric)
                     grads_new = torch.autograd.grad(potential_new, [z_coords, z_sections], create_graph=False, retain_graph=False, allow_unused=True)
                     force_coords_new = -grads_new[0] if grads_new[0] is not None else torch.zeros_like(z_coords)
                     force_sections_new = -grads_new[1] if grads_new[1] is not None else torch.zeros_like(z_sections)
                     
                     if torch.isnan(force_coords_new).any() or torch.isnan(force_sections_new).any():
                          raise ValueError(f"NaNs in Hamiltonian Forces (New) at Step {step}")

                     p_coords = p_coords + 0.5 * force_coords_new * dt
                     p_sections = p_sections + 0.5 * force_sections_new * dt

             except Exception as e:
                 with open("hamiltonian_error.log", "a") as f:
                     f.write(f"Symplectic Integrate Error: {e}\\n")
                 return coords, sections

        # Ensure Probability Simplex for Sections
        z_sections = torch.nn.functional.softmax(z_sections, dim=-1)
        
        return z_coords, z_sections
"""

for line in lines:
    if "def _symplectic_integrate" in line:
        in_function = True
        new_lines.append(correct_body + "\n")
        continue
    
    if in_function:
        if "def create_geometric_adapter" in line:
            in_function = False
            new_lines.append(line)
        # Skip all lines inside the function
        continue
    
    new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)

print("Successfully repaired _symplectic_integrate.")
