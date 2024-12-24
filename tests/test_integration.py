import unittest
import torch
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import MIDICAPSDataset
from src.model.config import ProjectConfig

class TestMusicCaptionIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.config = ProjectConfig()
        
        # Create data directory if it doesn't exist
        data_path = project_root / "data" / "midi_captions"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize datasets
        cls.train_dataset = MIDICAPSDataset(
            data_path=str(data_path),
            max_length=cls.config.data.max_length,
            model_name=cls.config.model.base_model,
            split="train"
        )
        
        cls.val_dataset = MIDICAPSDataset(
            data_path=str(data_path),
            max_length=cls.config.data.max_length,
            model_name=cls.config.model.base_model,
            split="val"
        )

    def test_dataset_initialization(self):
        """Test if datasets are properly initialized."""
        self.assertIsNotNone(self.train_dataset)
        self.assertIsNotNone(self.val_dataset)
        self.assertIsInstance(self.train_dataset, MIDICAPSDataset)
        self.assertIsInstance(self.val_dataset, MIDICAPSDataset)

if __name__ == '__main__':
    unittest.main()