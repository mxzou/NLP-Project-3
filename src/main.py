# MAIN Training script

import argparse
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
import wandb
from src.utils.manage_utils import WandBProcessManager

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now we can import from src
from src.data.loader import MIDICAPSDataset
from src.model.config import ProjectConfig
from src.training.trainer import MusicT5Trainer
from src.utils.logger import WandBLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train MIDI caption model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='data/midi_captions',
                       help='Path to the dataset directory')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    # Add sweep parameters
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (for sweep compatibility)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout rate')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA r parameter')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (for sweep compatibility)')
    
    args = parser.parse_args()
    
    # Use sweep parameters if provided
    if args.learning_rate is not None:
        args.lr = args.learning_rate
    if args.num_epochs is not None:
        args.epochs = args.num_epochs
    
    return args

def create_wandb_config(args: argparse.Namespace, config: ProjectConfig, device: str) -> Dict[str, Any]:
    """Create W&B configuration dictionary.
    
    Args:
        args: Command line arguments
        config: Project configuration
        device: Device being used for training
    
    Returns:
        Dictionary of configuration parameters for W&B
    """
    return {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "learning_rate": args.lr,
        "model": config.model.base_model,
        "max_length": config.data.max_length,
        "device": device
    }

def main():
    # Initialize process manager
    wandb_manager = WandBProcessManager()
    
    args = parse_args()
    config = ProjectConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize WandB logger
    if args.wandb_project:
        logger = WandBLogger(
            project_name=args.wandb_project,
            run_name=f"run_{timestamp}",
            config=create_wandb_config(args, config, str(device))  # Pass all required parameters
        )
    
    try:
        wandb_manager.cleanup()  # Clean existing processes first
        # ... rest of your wandb initialization code ...
        wandb.init(project="your-project")
        wandb_manager.initialized = True
        
        # Create data directory if it doesn't exist
        data_path = Path(args.data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("Loading datasets...")
        train_dataset = MIDICAPSDataset(
            data_path=str(data_path),
            max_length=config.data.max_length,
            model_name=config.model.base_model,
            split="train"
        )
        
        val_dataset = MIDICAPSDataset(
            data_path=str(data_path),
            max_length=config.data.max_length,
            model_name=config.model.base_model,
            split="val"
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )

        # Initialize model using the correct model class for T5
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model.base_model)
        
        trainer = MusicT5Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            wandb_config=None  # We already initialized wandb
        )

        # Start training
        print("Starting training...")
        for epoch in range(args.epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch_idx, batch in enumerate(train_pbar):
                # Forward and backward passes
                loss = trainer.training_step(batch)
                total_train_loss += loss
                
                # Log batch metrics
                if args.wandb_project and batch_idx % 10 == 0:
                    logger.log_training_metrics(
                        epoch=epoch,
                        train_loss=loss,
                        val_loss=0.0,
                        learning_rate=args.lr,
                        batch_size=args.batch_size,
                        batch_idx=batch_idx
                    )
            
            # ... (rest of your training loop)
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        wandb_manager.cleanup()
        raise
    finally:
        if args.wandb_project:
            logger.finish()

if __name__ == "__main__":
    main()