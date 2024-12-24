import subprocess
import sys
from pathlib import Path

def run_command(command):
    """Run a command and print output."""
    print(f"\nRunning: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def main():
    """Setup the project environment."""
    print("Setting up MIDI Caption project environment...")
    
    # Create necessary directories
    dirs = [
        "src/data",
        "src/model",
        "src/training",
        "src/evaluation",
        "tests",
        "notebooks"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Install requirements
    commands = [
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio",  # M1 Mac optimized
        "pip install -r requirements.txt",
        "pip install nltk",
        "python -c \"import nltk; nltk.download('punkt')\"",
    ]
    
    success = all(run_command(cmd) for cmd in commands)
    
    if success:
        print("\n✅ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python check_structure.py")
        print("2. Run: python tests/test_setup.py")
    else:
        print("\n❌ Some setup steps failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()