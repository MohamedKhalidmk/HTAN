from configs.base_config import BASE, set_seed
import copy

# ---------------------------------------------------------------------------
# TransAttUNet — paper faithful
# SGD, momentum 0.9, weight decay 1e-4, LR decay x10 every 40 epochs
# Same config used for U-Net and DoubleU-Net baselines
# ---------------------------------------------------------------------------

CONFIG = copy.deepcopy(BASE)

CONFIG.update({
    "MODEL":        "transattunet",
    "EXP_NAME":     "transattunet_bs4_ep100",

    # Attention
    "NUM_HEADS":    8,

    # Optimizer
    "OPTIMIZER":    "sgd",
    "LR":           0.0001,
    "MOMENTUM":     0.9,
    "WEIGHT_DECAY": 0.0001,

    # Scheduler — step decay
    "LR_SCHEDULER": "step",
    "LR_STEP":      40,       # decay every 40 epochs
    "LR_GAMMA":     0.1,      # decay factor

    # Gradient clipping — CRITICAL: stabilizes SAA lambda parameters
    "CLIP_GRAD":    1.0,
})

CONFIG["CHECKPOINT_DIR"] = f"{CONFIG['SAVES_ROOT']}/{CONFIG['MODEL']}"


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    print(CONFIG)