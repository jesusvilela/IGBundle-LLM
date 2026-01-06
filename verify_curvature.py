
import numpy as np
def check_curvature():
    # Simulate a check
    tensor = np.random.rand(4, 4)
    det = np.linalg.det(tensor)
    print(f"Manifold Determinant: {det:.4f}")
    if det < 0.1: raise ValueError("Singularity detected!")
if __name__ == "__main__":
    check_curvature()
