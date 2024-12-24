from pathlib import Path
import importlib.util
import sys

def check_file(file_path: Path, required: bool = True) -> bool:
    """Check if file exists and is importable."""
    exists = file_path.exists()
    print(f"{'[REQUIRED]' if required else '[OPTIONAL]'} {file_path}: {'✅' if exists else '❌'}")
    
    if exists and file_path.suffix == '.py':
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"  └─ Import test: ✅")
            return True
        except Exception as e:
            print(f"  └─ Import test: ❌ ({str(e)})")
            return False
    return exists

def main():
    project_root = Path.cwd()
    print(f"\nChecking project structure in: {project_root}\n")
    
    # Required files
    required_files = [
        "src/__init__.py",
        "src/model/config.py",
        "src/model/model.py",
        "src/data/loader.py",
        "requirements.txt"
    ]
    
    # Optional files
    optional_files = [
        "src/training/utils.py",
        "src/evaluation/evaluator.py",
        "tests/test_setup.py",
        "notebooks/experiments.ipynb"
    ]
    
    all_good = True
    
    # Check required files
    print("Checking required files:")
    for file_path in required_files:
        if not check_file(project_root / file_path, required=True):
            all_good = False
    
    # Check optional files
    print("\nChecking optional files:")
    for file_path in optional_files:
        check_file(project_root / file_path, required=False)
    
    print("\nChecking Python environment:")
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch not installed")
        all_good = False
    
    if all_good:
        print("\n✅ Project structure looks good!")
    else:
        print("\n❌ Some required files are missing or have issues!")

if __name__ == "__main__":
    main()