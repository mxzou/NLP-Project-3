import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Optional, Union, List

class MIDICaptionModel:
    """T5 model with LoRA adaptation for MIDI caption generation."""
    
    def __init__(
        self,
        model_name: str = "t5-base",
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """Initialize model with LoRA configuration."""
        # Set device (MPS for M1 Mac)
        self.device = (device if device is not None 
                      else torch.device("mps" if torch.backends.mps.is_available()
                                     else "cpu"))
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        
        # Initialize model
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Configure LoRA
        if target_modules is None:
            target_modules = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
            
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.to(self.device)
        
        # Print trainable parameters info
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        """Calculate and print trainable parameters statistics."""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with loss calculation if labels provided."""
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.2,
        early_stopping: bool = True
    ) -> torch.Tensor:
        """Generate captions for given inputs."""
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping
        )
    
    def save_pretrained(self, path: str):
        """Save model and configuration."""
        self.model.save_pretrained(path)
    
    def load_pretrained(self, path: str):
        """Load saved model."""
        self.model = self.model.from_pretrained(
            self.base_model,
            path,
            config=self.peft_config
        )
        self.model.to(self.device)