import sys
from pathlib import Path
import wandb
import argparse
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.trainer import MusicT5Trainer
from src.model.config import ProjectConfig
from src.data.loader import MIDICAPSDataset
from src.utils.logger import WandBLogger

def train_with_config(sweep_config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Train model with given hyperparameter configuration."""
    # Initialize wandb run first
    run = wandb.init(project="midi-caption")
    
    try:
        # Now we can safely access the config
        config = ProjectConfig()
        
        # Update config with sweep parameters
        config.training.batch_size = run.config.batch_size
        config.training.learning_rate = run.config.learning_rate
        config.model.lora_r = run.config.lora_r
        config.model.lora_alpha = run.config.lora_alpha
        config.model.lora_dropout = run.config.lora_dropout
        
        # Rest of your training setup...
        print(f"Training with batch_size={config.training.batch_size}, "
              f"lr={config.training.learning_rate}")
              
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise
    finally:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, required=True,
                       help='W&B sweep ID to run')
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of trials to run')
    args = parser.parse_args()
    
    wandb.agent(
        sweep_id=args.sweep_id,
        function=lambda: train_with_config(wandb.config, args),
        project="midi-caption",
        count=args.num_trials
    )

if __name__ == "__main__":
    main()