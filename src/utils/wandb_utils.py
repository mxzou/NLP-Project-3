import wandb
import torch
from typing import Dict, Optional
import psutil
import GPUtil

class WandBLogger:
    """Enhanced WandB logging utilities for MIDI caption model."""
    
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """Initialize WandB logger with custom metrics."""
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            reinit=True
        )
        
        # Initialize system monitoring
        self.gpu_available = torch.cuda.is_available()
        
    def log_system_metrics(self):
        """Log system resource usage."""
        metrics = {
            'system/cpu_percent': psutil.cpu_percent(),
            'system/memory_percent': psutil.virtual_memory().percent,
        }
        
        if self.gpu_available:
            gpu = GPUtil.getGPUs()[0]  # Get first GPU
            metrics.update({
                'system/gpu_utilization': gpu.load * 100,
                'system/gpu_memory_percent': gpu.memoryUtil * 100
            })
            
        wandb.log(metrics)
    
    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        batch_size: int,
        **kwargs
    ):
        """Log training metrics with enhanced tracking."""
        metrics = {
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/learning_rate': learning_rate,
            'train/epoch': epoch,
            'train/batch_size': batch_size
        }
        
        # Add additional metrics
        metrics.update(**kwargs)
        
        # Log system metrics alongside training metrics
        self.log_system_metrics()
        
        wandb.log(metrics)
    
    def log_evaluation_metrics(
        self,
        bleu_score: float,
        feature_coverage: Dict[str, float],
        num_examples: int
    ):
        """Log evaluation metrics."""
        metrics = {
            'eval/bleu_score': bleu_score,
            'eval/num_examples': num_examples
        }
        
        # Log feature coverage metrics
        for feature, coverage in feature_coverage.items():
            metrics[f'eval/coverage/{feature}'] = coverage
            
        wandb.log(metrics)
    
    def log_generation_examples(
        self,
        input_features: str,
        generated_caption: str,
        reference_caption: str,
        step: int
    ):
        """Log generation examples to WandB."""
        wandb.log({
            'examples/input_features': wandb.Html(input_features),
            'examples/generated': wandb.Html(generated_caption),
            'examples/reference': wandb.Html(reference_caption),
            'examples/step': step
        })
    
    def finish(self):
        """Properly close WandB run."""
        wandb.finish()

def create_wandb_config(args) -> Dict:
    """Create a standardized WandB config dictionary."""
    return {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'max_samples': args.max_samples,
        'model': args.model_name,
        'learning_rate': args.lr,
        'system': {
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'python_version': psutil.python_version(),
            'total_memory': psutil.virtual_memory().total / (1024 ** 3)  # GB
        }
    }