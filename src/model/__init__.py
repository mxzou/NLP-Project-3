from pathlib import Path

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Import key components
# from .model.model import MIDICaptionModel
from src.data.loader import MIDICAPSDataset
from .config import ProjectConfig

# Instead, use a simpler import structure
from . import model

# Export the class with the new name
__all__ = ['MIDICAPSDataset']