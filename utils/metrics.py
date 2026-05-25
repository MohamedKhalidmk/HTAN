import torch


def segmentation_metrics(preds, targets, eps=1e-6):
    """
    Computes standard segmentation metrics from raw logits.

    Args:
        preds:   Raw logits from model (B, 1, H, W)
        targets: Binary ground truth masks (B, 1, H, W)

    Returns:
        dict with keys: dice, iou, acc, rec, pre
    """
    preds   = preds.detach()
    targets = targets.detach()

    probs    = torch.sigmoid(preds)
    preds_bin = (probs > 0.5).float()

    # Flatten entire batch for global calculation
    preds_bin = preds_bin.view(-1)
    targets   = targets.view(-1)

    TP = (preds_bin * targets).sum()
    TN = ((1 - preds_bin) * (1 - targets)).sum()
    FP = (preds_bin * (1 - targets)).sum()
    FN = ((1 - preds_bin) * targets).sum()

    return {
        "dice": ((2 * TP)      / (2 * TP + FP + FN + eps)).item(),
        "iou":  (TP            / (TP + FP + FN + eps)).item(),
        "acc":  ((TP + TN)     / (TP + TN + FP + FN + eps)).item(),
        "rec":  (TP            / (TP + FN + eps)).item(),
        "pre":  (TP            / (TP + FP + eps)).item(),
    }