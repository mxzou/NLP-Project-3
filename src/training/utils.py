# Training utilities

import torch
import random
import numpy as np
from pathlib import Path
from typing import Optional

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_dir: Path,
    name: str = "checkpoint"
):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, save_dir / f"{name}_epoch_{epoch}.pt")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Path
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

def get_device() -> torch.device:
    """Get available device (GPU/CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")