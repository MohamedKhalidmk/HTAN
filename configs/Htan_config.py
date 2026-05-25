from configs.base_config import BASE, set_seed
import copy

# ---------------------------------------------------------------------------
# HTAN — AdamW + cosine annealing
# Used for all HTAN variants: htan_1_n2, htan_1_n4, htan_2_n2, htan_2_n4,
# htan_1_hres_only
# EXPANSION_N and MODEL are overridden per variant in train.py
# ---------------------------------------------------------------------------

CONFIG = copy.deepcopy(BASE)

CONFIG.update({
    "MODEL":        "htan",
    "EXP_NAME":     "htan_bs4_ep100",

    # Attention
    "NUM_HEADS":    8,

    # Optimizer
    "OPTIMIZER":    "adamw",
    "LR":           0.0001,
    "BETAS":        (0.9, 0.999),
    "WEIGHT_DECAY": 0.01,

    # Scheduler — cosine annealing
    "LR_SCHEDULER": "cosine",
    "LR_MIN":       1e-6,

    # mHC specific
    "EXPANSION_N":  4,         # default, overridden per experiment
    "MHC_BLOCKS":   1,         # 1 = HTAN_1, 2 = HTAN_2
    "HRES_ONLY":    False,     # True = only constrain H_res, fix H_pre and H_post

    # Gradient clipping — CRITICAL: stabilizes SAA lambda parameters
    "CLIP_GRAD":    1.0,
})

CONFIG["CHECKPOINT_DIR"] = f"{CONFIG['SAVES_ROOT']}/{CONFIG['MODEL']}"


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    print(CONFIG)