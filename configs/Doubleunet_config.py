from configs.transattunet_config import CONFIG as _BASE
import copy

# ---------------------------------------------------------------------------
# DoubleU-Net — same training config as TransAttUNet (SGD, step decay)
# Only MODEL and EXP_NAME differ
# ---------------------------------------------------------------------------

CONFIG = copy.deepcopy(_BASE)

CONFIG.update({
    "MODEL":    "doubleunet",
    "EXP_NAME": "doubleunet_bs4_ep100",
})

CONFIG["CHECKPOINT_DIR"] = f"{CONFIG['SAVES_ROOT']}/{CONFIG['MODEL']}"