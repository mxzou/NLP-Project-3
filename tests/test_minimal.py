import sys
from pathlib import Path
print("Starting minimal test...")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
print(f"Added {project_root} to Python path")

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name())
    
    from src.model.config import ProjectConfig
    print("\nSuccessfully imported ProjectConfig")
    
    config = ProjectConfig()
    print("Successfully created config instance")
    
    print("\nTest completed successfully!")
except Exception as e:
    print(f"\nError occurred: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    raise