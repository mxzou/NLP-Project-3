# Data loading and processing

import os
from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MIDICAPSDataset(Dataset):
    def __init__(self, split='train', max_samples=None):
        """
        Initialize MIDICAPS dataset from Hugging Face
        """
        print(f"Loading {split} split from MIDICAPS dataset...")
        self.dataset = load_dataset("AMAAI-Lab/MIDICAPS", split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        print(f"Loaded {len(self.dataset)} samples from {split} set")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
