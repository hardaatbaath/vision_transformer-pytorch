def get_config():
    return {
        "chw": (1, 28, 28),
        "num_epochs": 5, 
        "lr": 0.005,
        "n_patches": 7,
        "n_blocks": 2,
        "hidden_d": 8,
        "n_heads": 2,
        "out_d": 10
    }