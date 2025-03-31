import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "img_size": 16,
    "img_channels": 4,
    "n_classes": -1,

    "batch_size": 64,
    "n_epochs": 1000,
    "learning_rate": 3e-4,

    "T": 200,

    "patch_size": 2,
    "n_blocks": 12,
    "n_heads": 12,
    "embd_dim": 12 * 32,
    "dropout": 0.0
}