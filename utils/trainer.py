import os
import torch
import torchvision.transforms.functional as TF
from utils.metrics import segmentation_metrics


# ---------------------------------------------------------------------------
# TTA — Test Time Augmentation (original + hflip + vflip)
# ---------------------------------------------------------------------------
def predict_with_tta(model, imgs, epoch=None):
    """Pass epoch for SAA warmup lambda during validation."""
    pred_orig = model(imgs, epoch=epoch)
    pred_h    = TF.hflip(model(TF.hflip(imgs), epoch=epoch))
    pred_v    = TF.vflip(model(TF.vflip(imgs), epoch=epoch))
    return (pred_orig + pred_h + pred_v) / 3.0


# ---------------------------------------------------------------------------
# One training epoch — fp32, passes epoch to model
# ---------------------------------------------------------------------------
def train_one_epoch(loader, model, optimizer, loss_fn, config, epoch):
    model.train()
    total_loss = 0.0
    device     = config["DEVICE"]

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        preds = model(imgs, epoch=epoch)
        loss  = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Validation with TTA — fp32, passes epoch for SAA warmup
# ---------------------------------------------------------------------------
def validate(loader, model, loss_fn, config, epoch):
    model.eval()
    total_loss  = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "rec": 0.0, "pre": 0.0}
    device      = config["DEVICE"]

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds       = predict_with_tta(model, imgs, epoch=epoch)
            loss        = loss_fn(preds, masks)
            total_loss += loss.item()

            batch_metrics = segmentation_metrics(preds, masks)
            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]

    n = len(loader)
    for k in metrics_sum:
        metrics_sum[k] /= n

    return total_loss / n, metrics_sum


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer,
                loss_fn, config, scheduler=None):
    epochs         = config["EPOCHS"]
    checkpoint_dir = config["CHECKPOINT_DIR"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_ckpt_path   = os.path.join(checkpoint_dir, "best_model.pth")
    resume_ckpt_path = os.path.join(checkpoint_dir, "resume_checkpoint.pth")

    history = {
        "train_loss": [], "val_loss": [],
        "dice": [], "iou": [], "acc": [], "rec": [], "pre": []
    }

    best_dice   = 0.0
    start_epoch = 1

    # --- Resume if checkpoint exists ---
    if os.path.exists(resume_ckpt_path):
        print(f"Resuming from: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=config["DEVICE"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history     = ckpt["history"]
        best_dice   = ckpt["best_dice"]
        start_epoch = ckpt["epoch"] + 1
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resumed from epoch {start_epoch - 1}, best Dice: {best_dice:.4f}")

    model.to(config["DEVICE"])

    for epoch in range(start_epoch, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            train_loader, model, optimizer, loss_fn, config, epoch
        )
        val_loss, val_metrics = validate(
            val_loader, model, loss_fn, config, epoch
        )

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in ["dice", "iou", "acc", "rec", "pre"]:
            history[k].append(val_metrics[k])

        print(
            f"Epoch {epoch:>3}/{epochs} | LR: {current_lr:.2e} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Dice: {val_metrics['dice']*100:.2f}% | "
            f"IoU: {val_metrics['iou']*100:.2f}%"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  New best Dice: {best_dice*100:.2f}% — saved.")

        if epoch % 5 == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "history":   history,
                "best_dice": best_dice,
            }, resume_ckpt_path)

    print(f"\nTraining complete. Best Validation Dice: {best_dice*100:.2f}%")
    return history