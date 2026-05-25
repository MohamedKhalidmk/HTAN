import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperLoss(nn.Module):
    """
    Equation (9) from TransAttUNet paper:
        L = 0.5 * BCE + 0.5 * Dice

    BCE uses logits directly (sigmoid applied internally).
    Dice uses sigmoid probabilities.
    """

    def forward(self, logits, targets, smooth=1e-6):
        # BCE — sigmoid applied internally
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # Dice — apply sigmoid manually
        probs    = torch.sigmoid(logits)
        probs_f  = probs.view(-1)
        targets_f = targets.view(-1)

        intersection = (probs_f * targets_f).sum()
        dice_score   = (2. * intersection + smooth) / \
                       (probs_f.sum() + targets_f.sum() + smooth)
        dice_loss    = 1 - dice_score

        return 0.5 * bce_loss + 0.5 * dice_loss