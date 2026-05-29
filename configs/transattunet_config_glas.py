"""
transattunet_config_glas.py — TransAttUNet config for GlaS dataset.

Same as HTAN config but without CLIP_GRAD.
Use this to fairly compare TransAttUNet vs HTAN on GlaS under identical optimizer conditions.
"""

import torch

CONFIG = {
    # Model
    "MODEL":          "transattunet",

    # Training
    "EPOCHS":         100,
    "BATCH_SIZE":     4,
    "SEED":           123,

    # Optimizer — AdamW same as HTAN
    "OPTIMIZER":      "adamw",
    "LR":             1e-4,
    "WEIGHT_DECAY":   0.01,
    "BETAS":          (0.9, 0.999),

    # Scheduler — cosine same as HTAN
    "LR_SCHEDULER":   "cosine",
    "LR_MIN":         1e-6,

    # No gradient clipping — TransAttUNet doesn't need Sinkhorn protection
    "CLIP_GRAD":      None,

    # Hardware
    "DEVICE":         "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS":    4,

    # Paths
    "SAVES_ROOT":     "/opt/dlami/nvme/HTAN/saves",
    "RESULTS_ROOT":   "/opt/dlami/nvme/HTAN/results",
}