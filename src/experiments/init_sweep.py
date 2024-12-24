import wandb
from sweep_config import sweep_configuration

def init_sweep():
    """Initialize a new W&B sweep."""
    try:
        # Initialize the sweep
        sweep_id = wandb.sweep(
            sweep_configuration,
            project="midi-caption"
        )
        print(f"\nSweep initialized successfully!")
        print(f"Sweep ID: {sweep_id}")
        print(f"\nTo start the sweep, run:")
        print(f"python src/experiments/hyperparameter_search.py --sweep_id {sweep_id} --num_trials 5")
        
        return sweep_id
        
    except Exception as e:
        print(f"Error initializing sweep: {str(e)}")
        raise

if __name__ == "__main__":
    init_sweep() 