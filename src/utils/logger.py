import wandb
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class WandBLogger:
    """Wrapper for Weights & Biases logging."""
    
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize WandB logger.
        
        Args:
            project_name: Name of the W&B project
            run_name: Optional name for this run
            config: Optional configuration dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{timestamp}"
        
        try:
            wandb.init(
                project=project_name,
                name=self.run_name,
                config=config,
                reinit=True
            )
            logger.info(f"Initialized W&B run: {self.run_name}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {str(e)}")
            raise

    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        batch_size: int,
        batch_idx: Optional[int] = None
    ) -> None:
        """Log training metrics to W&B."""
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/learning_rate": learning_rate,
            "train/batch_size": batch_size
        }
        
        if batch_idx is not None:
            metrics["train/batch_idx"] = batch_idx
        
        try:
            wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {str(e)}")

    def log_generation_examples(
        self,
        input_features: str,
        generated_caption: str,
        reference_caption: str,
        step: int
    ) -> None:
        """Log generation examples to W&B."""
        try:
            wandb.log({
                "examples/input_features": input_features,
                "examples/generated_caption": generated_caption,
                "examples/reference_caption": reference_caption,
                "examples/step": step
            })
        except Exception as e:
            logger.warning(f"Failed to log examples: {str(e)}")

    def log_evaluation_metrics(
        self,
        bleu_score: float,
        feature_coverage: Dict[str, float],
        num_examples: int
    ) -> None:
        """Log evaluation metrics to W&B."""
        try:
            metrics = {
                "eval/bleu": bleu_score,
                "eval/num_examples": num_examples,
                **{f"eval/{k}": v for k, v in feature_coverage.items()}
            }
            wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics: {str(e)}")

    def finish(self) -> None:
        """Clean up W&B run."""
        try:
            wandb.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {str(e)}") 