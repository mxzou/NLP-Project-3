import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print("PyTorch version:", torch.__version__)
print("\nDevice Information:")
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

# Import config directly
from src.model.config import ProjectConfig
config = ProjectConfig()
print("\nSelected device:", config.device)

# Test tensor operations
try:
    print("\nTesting tensor operations...")
    # Create tensor
    x = torch.randn(2, 3).to(config.device)
    y = torch.randn(2, 3).to(config.device)
    # Perform operation
    z = x + y
    print("Tensor operation successful")
    print("Tensor device:", z.device)
except Exception as e:
    print(f"Error during tensor operation: {str(e)}")

print("\nTest completed!")