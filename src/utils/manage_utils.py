import psutil
import wandb

class WandBProcessManager:
    def __init__(self):
        self.initialized = False
        
    def cleanup(self):
        """Initialize wandb with proper cleanup"""
        try:
            # Get all running wandb processes
            current_process = psutil.Process()
            for proc in current_process.children(recursive=True):
                if "wandb" in proc.name().lower():
                    proc.kill()
            
            if wandb.run is not None:
                wandb.finish()
            self.initialized = False
        except Exception as e:
            print(f"Cleanup error: {e}") 