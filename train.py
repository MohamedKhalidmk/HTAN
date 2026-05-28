"""
train.py — Run training for any model on any dataset.

Usage:
    python3 train.py --model transattunet      --dataset isic
    python3 train.py --model htan_2_n2         --dataset isic
    python3 train.py --model transattunet      --dataset glas
    python3 train.py --model htan_2_n2         --dataset glas
    python3 train.py --model transattunet      --dataset covid
    python3 train.py --model transattunet      --dataset lung
    python3 train.py --model transattunet      --dataset bowl
"""

import argparse
import os
import sys
import torch


# ---------------------------------------------------------------------------
# Per-dataset image size — matches paper exactly
# ---------------------------------------------------------------------------
DATASET_IMG_SIZE = {
    "isic":  256,
    "glas":  128,
    "covid": 512,
    "lung":  512,
    "bowl":  256,
}


def get_model(name, img_size=256):
    from models.transattunet.TransAttUnet import TransAttUNet_R
    from models.htan.htan import HTAN_1, HTAN_2, HTAN_1_Hres_only
    from models.baselines.unet import UNet
    from models.baselines.doubleunet import DoubleUNet

    registry = {
        "transattunet":     lambda: TransAttUNet_R(),
        "unet":             lambda: UNet(),
        "doubleunet":       lambda: DoubleUNet(),
        "htan_1_n2":        lambda: HTAN_1(expansion_n=2, img_size=img_size),
        "htan_1_n4":        lambda: HTAN_1(expansion_n=4, img_size=img_size),
        "htan_2_n2":        lambda: HTAN_2(expansion_n=2, img_size=img_size),
        "htan_2_n4":        lambda: HTAN_2(expansion_n=4, img_size=img_size),
        "htan_1_hres_only": lambda: HTAN_1_Hres_only(expansion_n=4, img_size=img_size),
    }

    if name not in registry:
        print(f"Unknown model: {name}")
        print(f"Available: {list(registry.keys())}")
        sys.exit(1)

    return registry[name]()


def get_config(model_name, dataset_name):
    if model_name in ("transattunet", "unet", "doubleunet"):
        from configs.transattunet_config import CONFIG
        cfg = dict(CONFIG)
    else:
        from configs.htan_config import CONFIG
        cfg = dict(CONFIG)

    # Override IMG_SIZE based on dataset
    cfg["IMG_SIZE"]       = DATASET_IMG_SIZE.get(dataset_name, 256)
    cfg["MODEL"]          = model_name
    cfg["DATASET"]        = dataset_name
    cfg["CHECKPOINT_DIR"] = os.path.join(cfg["SAVES_ROOT"], f"{model_name}_{dataset_name}")
    os.makedirs(cfg["CHECKPOINT_DIR"], exist_ok=True)
    return cfg


def get_loaders(dataset_name, config):
    if dataset_name == "isic":
        from datasets.isic_dataset import get_loaders as _get
        return _get(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
    elif dataset_name == "glas":
        from datasets.glas_dataset import get_loaders as _get
        return _get(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"],
        )
    elif dataset_name == "covid":
        from datasets.covid_dataset import get_loaders as _get
        return _get(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
    elif dataset_name == "lung":
        from datasets.lung_dataset import get_loaders as _get
        return _get(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
    elif dataset_name == "bowl":
        from datasets.bowl_dataset import get_loaders as _get
        return _get(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
    raise NotImplementedError(f"Dataset '{dataset_name}' not yet implemented.")


def get_optimizer(model, config):
    opt = config.get("OPTIMIZER", "sgd").lower()
    if opt == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config["LR"],
            betas=config.get("BETAS", (0.9, 0.999)),
            weight_decay=config["WEIGHT_DECAY"],
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=config["LR"],
        momentum=config.get("MOMENTUM", 0.9),
        weight_decay=config["WEIGHT_DECAY"],
    )


def get_scheduler(optimizer, config):
    sched = config.get("LR_SCHEDULER", "step").lower()
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["EPOCHS"],
            eta_min=config.get("LR_MIN", 1e-6)
        )
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("LR_STEP", 40),
        gamma=config.get("LR_GAMMA", 0.1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    config = get_config(args.model, args.dataset)

    from configs.base_config import set_seed
    set_seed(config["SEED"])

    print(f"\nModel:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"IMG_SIZE:   {config['IMG_SIZE']}")
    print(f"Device:     {config['DEVICE']}")
    print(f"Epochs:     {config['EPOCHS']}")
    print(f"Optimizer:  {config.get('OPTIMIZER', 'sgd').upper()}")
    print(f"Checkpoint: {config['CHECKPOINT_DIR']}\n")

    train_loader, val_loader = get_loaders(args.dataset, config)

    model    = get_model(args.model, img_size=config["IMG_SIZE"]).to(config["DEVICE"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    from utils.losses import PaperLoss
    from utils.trainer import train_model

    train_model(
        model, train_loader, val_loader,
        optimizer, PaperLoss(), config, scheduler
    )


if __name__ == "__main__":
    main()