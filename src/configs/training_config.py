"""Training configuration for MIDI caption model."""

config = {
    # Model parameters
    'model_name': 't5-small',
    'max_source_length': 512,
    'max_target_length': 128,
    
    # Training parameters
    'batch_size': 8,  # Increased from test run
    'num_epochs': 20,  # Full training run
    'learning_rate': 2e-5,  # Based on successful test
    'max_samples': 10000,  # Increased dataset size
    
    # LoRA parameters
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    
    # System parameters
    'num_workers': 2,
    'pin_memory': True,
    'gradient_accumulation_steps': 2,
    
    # Logging parameters
    'log_every_n_steps': 10,
    'eval_every_n_epochs': 1,
    'save_every_n_epochs': 2,
    
    # Paths
    'checkpoint_dir': 'checkpoints/full_training',
    'wandb_project': 'midi-caption-full',
    
    # Early stopping
    'patience': 3,
    'min_delta': 0.001
}