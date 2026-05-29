"""
evaluate.py — Load best checkpoint and evaluate a model or all models.

Usage:
    python3 evaluate.py --model transattunet --dataset isic
    python3 evaluate.py --model all          --dataset isic
    python3 evaluate.py --model transattunet --dataset glas
    python3 evaluate.py --model all          --dataset glas
"""

import argparse
import os
import json
import torch
import torchvision.transforms.functional as TF

from utils.metrics import segmentation_metrics


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

ALL_MODELS = [
    "unet",
    "doubleunet",
    "transattunet",
    "htan_1_n2",
    "htan_1_n4",
    "htan_2_n2",
    "htan_2_n4",
    "htan_1_hres_only",
]


def get_model(name, img_size=256):
    from models.transattunet.TransAttUnet import TransAttUNet_R
    from models.htan.htan import HTAN_1, HTAN_2, HTAN_1_Hres_only
    from models.baselines.unet import UNet
    from models.baselines.doubleunet import DoubleUNet

    registry = {
        "transattunet":     lambda: TransAttUNet_R(),
        "unet":             lambda: UNet(n_channels=3, n_classes=1),
        "doubleunet":       lambda: DoubleUNet(n_channels=3, n_classes=1),
        "htan_1_n2":        lambda: HTAN_1(expansion_n=2, img_size=img_size),
        "htan_1_n4":        lambda: HTAN_1(expansion_n=4, img_size=img_size),
        "htan_2_n2":        lambda: HTAN_2(expansion_n=2, img_size=img_size),
        "htan_2_n4":        lambda: HTAN_2(expansion_n=4, img_size=img_size),
        "htan_1_hres_only": lambda: HTAN_1_Hres_only(expansion_n=4, img_size=img_size),
    }
    return registry[name]()


def get_config(model_name, dataset_name):
    if model_name in ("transattunet", "unet", "doubleunet"):
        from configs.transattunet_config import CONFIG
        cfg = dict(CONFIG)
    else:
        from configs.htan_config import CONFIG
        cfg = dict(CONFIG)

    cfg["IMG_SIZE"]       = DATASET_IMG_SIZE.get(dataset_name, 256)
    cfg["MODEL"]          = model_name
    cfg["DATASET"]        = dataset_name
    cfg["CHECKPOINT_DIR"] = os.path.join(
        cfg["SAVES_ROOT"],
        f"{model_name}_{dataset_name}" if dataset_name != "isic" else model_name
    )
    return cfg


def get_val_loader(dataset_name, config):
    if dataset_name == "isic":
        from datasets.isic_dataset import get_loaders
        _, val_loader, _ = get_loaders(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
        return val_loader
    elif dataset_name == "glas":
        from datasets.glas_dataset import get_loaders
        _, val_loader, _ = get_loaders(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"],
        )
        return val_loader
    elif dataset_name == "covid":
        from datasets.covid_dataset import get_loaders
        _, val_loader, _ = get_loaders(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
        return val_loader
    elif dataset_name == "lung":
        from datasets.lung_dataset import get_loaders
        _, val_loader, _ = get_loaders(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
        return val_loader
    elif dataset_name == "bowl":
        from datasets.bowl_dataset import get_loaders
        _, val_loader, _ = get_loaders(
            img_size=config["IMG_SIZE"],
            batch_size=config["BATCH_SIZE"],
            seed=config["SEED"],
            num_workers=config["NUM_WORKERS"],
        )
        return val_loader
    raise NotImplementedError(f"Dataset '{dataset_name}' not yet implemented.")


def predict_with_tta(model, imgs):
    with torch.no_grad():
        pred_orig = model(imgs)
        pred_h    = TF.hflip(model(TF.hflip(imgs)))
        pred_v    = TF.vflip(model(TF.vflip(imgs)))
    return (pred_orig + pred_h + pred_v) / 3.0


def evaluate_model(model_name, dataset_name):
    config    = get_config(model_name, dataset_name)
    device    = config["DEVICE"]
    img_size  = config["IMG_SIZE"]
    best_ckpt = os.path.join(config["CHECKPOINT_DIR"], "best_model.pth")

    if not os.path.exists(best_ckpt):
        print(f"  [{model_name}] No checkpoint at {best_ckpt}, skipping.")
        return None

    model = get_model(model_name, img_size=img_size).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    val_loader  = get_val_loader(dataset_name, config)
    metrics_sum = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "rec": 0.0, "pre": 0.0}

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = predict_with_tta(model, imgs)
            m = segmentation_metrics(preds, masks)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

    n = len(val_loader)
    for k in metrics_sum:
        metrics_sum[k] = round(metrics_sum[k] / n * 100, 2)

    return metrics_sum


def print_table(results):
    header = f"{'Model':<22} {'Dice':>6} {'IoU':>6} {'ACC':>6} {'REC':>6} {'PRE':>6}"
    print("\n" + "=" * 57)
    print(header)
    print("=" * 57)
    for model_name, m in results.items():
        if m is None:
            print(f"{model_name:<22} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>6}")
        else:
            print(
                f"{model_name:<22} "
                f"{m['dice']:>6.2f} {m['iou']:>6.2f} "
                f"{m['acc']:>6.2f} {m['rec']:>6.2f} {m['pre']:>6.2f}"
            )
    print("=" * 57)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Model name or 'all'")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    args = parser.parse_args()

    models_to_eval = ALL_MODELS if args.model == "all" else [args.model]

    results = {}
    for m in models_to_eval:
        print(f"Evaluating {m}...")
        results[m] = evaluate_model(m, args.dataset)

    print_table(results)

    # Save results
    from configs.base_config import BASE
    results_dir = os.path.join(BASE["RESULTS_ROOT"], args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    for model_name, metrics in results.items():
        if metrics:
            out_path = os.path.join(results_dir, f"{model_name}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "model":   model_name,
                    "dataset": args.dataset,
                    **metrics
                }, f, indent=4)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()