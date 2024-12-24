import os
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import Optional, Dict, Any, Union, List
import logging
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from src.data.loader import MIDICAPSDataset
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    learning_rate: float
    epoch: int
    global_step: int

class MusicT5Trainer:  # This is the class we're importing
    """Trainer class for Music T5 model with comprehensive logging and error handling."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize trainer for Music T5 model."""
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=5e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
            
        self.lr_scheduler = lr_scheduler

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step.
        
        Args:
            batch: Dictionary containing input_ids, attention_mask, and labels
            
        Returns:
            Loss value for this step
        """
        try:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise

    def validation_step(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        self.model.train()
        return avg_val_loss

    def train(
        self,
        num_epochs: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 100
    ) -> List[float]:
        """Train the model."""
        self.model.train()
        global_step = 0
        epoch_losses = []
        
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
                
                with tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}") as pbar:
                    for step, batch in enumerate(pbar):
                        try:
                            # Move batch to device
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            
                            # Forward pass
                            outputs = self.model(**batch)
                            loss = outputs.loss / gradient_accumulation_steps
                            
                            # Backward pass
                            loss.backward()
                            
                            if (step + 1) % gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    max_grad_norm
                                )
                                
                                self.optimizer.step()
                                if self.lr_scheduler is not None:
                                    self.lr_scheduler.step()
                                self.optimizer.zero_grad()
                                
                                global_step += 1
                            
                            # Update metrics
                            epoch_loss += loss.item() * gradient_accumulation_steps
                            current_loss = epoch_loss / (step + 1)
                            pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                            
                        except Exception as e:
                            logger.error(f"Error in training step: {str(e)}")
                            raise RuntimeError(f"Training failed at step {step}: {str(e)}")
                
                epoch_losses.append(current_loss)
                logger.info(f"Epoch {epoch+1} completed with loss: {current_loss:.4f}")
                
            return epoch_losses
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def prepare_prompt(sample):
    """
    Prepare prompt from MIDI metadata with safety checks
    """
    def safe_join(field, default=""):
        """Safely join field values or return default"""
        if isinstance(field, (list, tuple)):
            return ", ".join(str(x) for x in field)
        elif isinstance(field, str):
            return field
        return default

    prompt = f"""Key: {sample.get('key', 'Unknown')}
Time Signature: {sample.get('time_signature', '4/4')}
Tempo: {sample.get('tempo', '')} ({sample.get('tempo_word', 'Unknown')})
Duration: {sample.get('duration_word', 'Unknown')}
Genres: {safe_join(sample.get('genre', []))}
Moods: {safe_join(sample.get('mood', []))}
Instruments: {safe_join(sample.get('instrument_summary', []))}
Main Chords: {safe_join(sample.get('chord_summary', []))}
Description: """
    
    return prompt

def train_model(batch_size, learning_rate, lora_r, lora_alpha, lora_dropout, num_epochs, max_samples):
    """
    Training function for MIDI caption generation using LoRA
    """
    try:
        print("\n=== Initialization Phase ===")
        print("1. Loading model and tokenizer...")
        model_name = "facebook/opt-350m"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        print("✓ Model loaded successfully")

        print("\n2. Configuring LoRA...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("✓ LoRA configured")

        def collate_fn(batch):
            """Custom collate function to properly handle input-target pairs"""
            prompts = [prepare_prompt(x) for x in batch]
            captions = [x['caption'] for x in batch]
            
            # Combine prompt and caption for input
            combined_inputs = [f"{prompt}{caption}" for prompt, caption in zip(prompts, captions)]
            
            # Tokenize combined inputs
            encodings = tokenizer(
                combined_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            
            # Create labels by shifting input_ids right
            labels = input_ids.clone()
            
            # Mask prompt tokens in labels with -100
            for idx, prompt in enumerate(prompts):
                prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
                labels[idx, :prompt_tokens] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        print("\n3. Loading dataset...")
        train_dataset = MIDICAPSDataset(split='train', max_samples=max_samples)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        print(f"✓ Dataset loaded with {len(train_dataset)} samples")

        print("\n=== Training Phase ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # AXIOM 1: Optimizer Configuration
        print("Configuring optimizer...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # Standard Adam betas
            eps=1e-8,           # Stability constant
            weight_decay=0.01   # L2 regularization
        )
        
        # AXIOM 2: Learning Rate Schedule
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = num_training_steps // 10  # 10% warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # AXIOM 3: Training Loop with Gradient Management
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, 
                              desc=f"Epoch {epoch+1}/{num_epochs}",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}')
            
            for batch_idx, batch in enumerate(progress_bar):
                # AXIOM 4: Forward Pass
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                except RuntimeError as e:
                    print(f"Error in forward pass: {e}")
                    continue
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # AXIOM 5: Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress - Fixed postfix format
                if batch_idx % 5 == 0:
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                    wandb.log({
                        'batch_loss': loss.item(),
                        'epoch': epoch,
                        'learning_rate': scheduler.get_last_lr()[0],
                        'progress': batch_idx / len(train_loader)
                    })
            
            # AXIOM 6: Epoch Summary
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)
            
            # Log epoch metrics
            wandb.log({
                'epoch_loss': avg_loss,
                'epoch': epoch
            })

        print("\n=== Training Complete ===")
        
        # Create checkpoints directory using pathlib
        save_directory = Path("checkpoints/best_model")
        save_directory.parent.mkdir(parents=True, exist_ok=True)
        save_directory.mkdir(exist_ok=True)
        
        print(f"Saving model to {save_directory}...")
        
        # 1. Save the base model and tokenizer
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        
        # 2. Explicitly save LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        lora_config.save_pretrained(save_directory)
        
        print(f"✓ Model, tokenizer, and LoRA config saved to {save_directory}")
        
        return model

    except Exception as e:
        print(f"\n❌ Error in training: {e}")
        raise e

# Make sure the class is available for import
__all__ = ['MusicT5Trainer']