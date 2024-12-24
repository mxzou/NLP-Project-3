import os
import sys
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
logger.info(f"Added {project_root} to Python path")

from src.model.config import ProjectConfig
from src.data.loader import MIDICapsLoader

def test_data_loading():
    """Test data loading functionality."""
    logger.info("\nTesting data loading...")
    
    try:
        config = ProjectConfig()
        logger.info("Created project config")

        # Create data directory if it doesn't exist
        data_path = project_root / "data" / "midi_captions"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loaders
        train_loader = MIDICapsLoader(
            data_path=str(data_path),
            max_length=config.data.max_length,
            model_name=config.model.base_model,
            split="train"
        )
        
        val_loader = MIDICapsLoader(
            data_path=str(data_path),
            max_length=config.data.max_length,
            model_name=config.model.base_model,
            split="val"
        )
        
        logger.info("Successfully created data loaders")
        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Error in data loading:")
        logger.error(str(e))
        raise

def main():
    """Main test sequence."""
    logger.info("Starting test setup...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
    try:
        logger.info("Importing required modules...")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        
        if torch.__version__:
            logger.info(f"PyTorch version {torch.__version__} available.")
        
        logger.info("Successfully imported project modules")
        logger.info("Starting main test sequence...")
        
        train_loader, val_loader = test_data_loading()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error("Error in main test sequence:")
        logger.error(str(e))
        raise

if __name__ == "__main__":
    main()

