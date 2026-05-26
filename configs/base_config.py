import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared across ALL models and datasets
# ---------------------------------------------------------------------------

BASE = {
    # Image
    "IMG_SIZE":    256,
    "BATCH_SIZE":  4,
    "EPOCHS":      100,
    "SEED":        123,
    "NUM_WORKERS": 4,


    # Paths
    "DATA_ROOT":  "/opt/dlami/nvme/HTAN/data",
    "SAVES_ROOT": "/opt/dlami/nvme/HTAN/saves",
    "RESULTS_ROOT": "/opt/dlami/nvme/HTAN/results",

    # Device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # Datasets available
    "DATASETS": ["isic", "lung", "covid", "bowl", "glas"],


}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True
    print(f"Seed set to {seed}")