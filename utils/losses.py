import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.contiguous().view(-1)

        # Compute Dice coefficient
        intersection = (outputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)

        # Convert Dice coefficient to loss
        return 1.0 - dice