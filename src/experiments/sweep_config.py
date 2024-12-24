sweep_configuration = {
    'method': 'random',  # Grid, random, or bayesian
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [4, 8, 16]
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-5,
            'max': 1e-3
        },
        'lora_r': {
            'values': [4, 8, 16, 32]
        },
        'lora_alpha': {
            'values': [16, 32, 64]
        },
        'lora_dropout': {
            'values': [0.1, 0.2, 0.3]
        }
    }
} 