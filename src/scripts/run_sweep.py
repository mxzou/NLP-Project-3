import wandb
from src.training.trainer import train_model

def main():
    # Initialize wandb with default config
    default_config = {
        "batch_size": 8,
        "learning_rate": 0.0001,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "num_epochs": 5,
        "max_samples": 2000
    }
    
    wandb.init(
        project="midi_caption_project-src_scripts",
        entity="mx-zou-cooper-union-for-advancement-of-science-and-arts",
        config=default_config
    )
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    print("Starting training run with config:")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"LoRA r: {config.lora_r}")
    
    try:
        model = train_model(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            num_epochs=config.num_epochs,
            max_samples=config.max_samples
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise e

if __name__ == "__main__":
    main()  