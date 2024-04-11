import torch

def dice_coefficient(outputs, targets, threshold=0.5, smooth=1e-6):
    # Apply threshold to outputs
    outputs = (outputs > threshold).float()

    # Flatten the tensors
    outputs = outputs.view(-1)
    targets = targets.contiguous().view(-1)

    # Compute Dice coefficient
    intersection = (outputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)

    return dice.item()

def iou_score(outputs, targets, threshold=0.5, smooth=1e-6):
    # Apply threshold to outputs
    outputs = (outputs > threshold).float()

    # Flatten the tensors
    outputs = outputs.view(-1)
    targets = targets.contiguous().view(-1)

    # Compute intersection and union
    intersection = (outputs * targets).sum()
    union = (outputs + targets).sum()

    # Compute IoU score
    iou = (intersection + smooth) / (union - intersection + smooth)

    return iou.item()