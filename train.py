"""
train.py — Run training for any model on any dataset.

Usage:
    python train.py --model transattunet --dataset isic
    python train.py --model htan_1_n4    --dataset isic
    python train.py --model htan_2_n2    --dataset isic
    python train.py --model unet         --dataset isic
    python train.py --model doubleunet   --dataset isic

    # Ablation
    python train.py --model htan_1_hres_only --dataset isic
"""

import argparse
import os
import sys
import torch

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def get_model(name):
    from models.transattunet.TransAttUnet import UNet_Attention_Transformer_Multiscale
    from models.htan.htan import HTAN_MODELS
    from models.baselines.unet import UNet
    from models.baselines.doubleunet import DoubleUNet

    registry = {
        "transattunet": lambda: UNet_Attention_Transformer_Multiscale(
            n_channels=3, n_classes=1
        ),
        "unet":         lambda: UNet(n_channels=3, n_classes=1),
        "doubleunet":   lambda: DoubleUNet(n_channels=3, n_classes=1),
        **HTAN_MODELS,
    }

    if name not in registry:
        print(f"Unknown model: {name}")
        print(f"Available: {list(registry.keys())}")
        sys.exit(1)

    return registry[name]()


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------
def get_config(model_name):
    if model_name in ("transattunet", "unet", "doubleunet"):
        from configs.transattunet_config import CONFIG
        cfg = dict(CONFIG)
    else:
        from configs.htan_config import CONFIG
        cfg = dict(CONFIG)

    cfg["MODEL"] = model_name
    cfg["CHECKPOINT_DIR"] = os.path.join(cfg["SAVES_ROOT"], model_name)
    os.makedirs(cfg["CHECKPOINT_DIR"], exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
def get_loaders(dataset_name, config):
    if dataset_name == "isic":
        from datasets.isic_dataset import get_loaders as _get
        return _get(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' loader not yet implemented.")


# ---------------------------------------------------------------------------
# Optimizer + scheduler factory
# ---------------------------------------------------------------------------
def get_optimizer(model, config):
    opt_name = config.get("OPTIMIZER", "sgd").lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config["LR"],
            betas=config.get("BETAS", (0.9, 0.999)),
            weight_decay=config["WEIGHT_DECAY"],
        )
    else:  # sgd
        return torch.optim.SGD(
            model.parameters(),
            lr=config["LR"],
            momentum=config.get("MOMENTUM", 0.9),
            weight_decay=config["WEIGHT_DECAY"],
        )


def get_scheduler(optimizer, config):
    sched_name = config.get("LR_SCHEDULER", "step").lower()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["EPOCHS"],
            eta_min=config.get("LR_MIN", 1e-6),
        )
    else:  # step
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("LR_STEP", 40),
            gamma=config.get("LR_GAMMA", 0.1),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Model name")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    args = parser.parse_args()

    config = get_config(args.model)

    from configs.base_config import set_seed
    set_seed(config["SEED"])

    print(f"\nModel:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device:  {config['DEVICE']}")
    print(f"Epochs:  {config['EPOCHS']}")
    print(f"Optimizer: {config.get('OPTIMIZER', 'sgd').upper()}")
    print(f"Checkpoint: {config['CHECKPOINT_DIR']}\n")

    # Data
    train_loader, val_loader = get_loaders(args.dataset, config)

    # Model
    model = get_model(args.model).to(config["DEVICE"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    # Optimizer + scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Loss
    from utils.losses import PaperLoss
    loss_fn = PaperLoss()

    # Train
    from utils.trainer import train_model
    history = train_model(
        model, train_loader, val_loader,
        optimizer, loss_fn, config, scheduler
    )

    print("Done.")


if __name__ == "__main__":
    main()