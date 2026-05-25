from configs.transattunet_config import CONFIG as _BASE
import copy

# ---------------------------------------------------------------------------
# U-Net — same training config as TransAttUNet (SGD, step decay)
# Only MODEL and EXP_NAME differ
# ---------------------------------------------------------------------------

CONFIG = copy.deepcopy(_BASE)

CONFIG.update({
    "MODEL":    "unet",
    "EXP_NAME": "unet_bs4_ep100",
})

CONFIG["CHECKPOINT_DIR"] = f"{CONFIG['SAVES_ROOT']}/{CONFIG['MODEL']}"