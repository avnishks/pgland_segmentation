import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import interpol

from scipy.ndimage import distance_transform_edt


def cce_loss(logits, targets, weights=1, reduction='mean', **kwargs):
    weights = torch.as_tensor(weights, device=logits.device)
    outputs = torch.log_softmax(logits, dim=1)
    numer = torch.sum(weights * -(targets * outputs), keepdim=True, axis=1)
    denom = torch.sum(weights * (targets), keepdim=True, axis=1)

    if reduction == 'none':
        return numer
    
    return torch.sum(numer) / torch.sum(denom)


def dice_cce_loss(logits, targets, softmax=True, **kwargs):
    return dice_loss_safe(logits, targets, start_idx=1) + cce_loss(logits, targets)


def dice_loss(logits, targets, weights=1, start_idx=0, softmax=True, **kwargs):
    weights = torch.as_tensor(weights, device=logits.device)
    outputs = F.softmax(logits, dim=1) if softmax else logits
    axes = [0] + list(range(2, outputs.ndim))
    numer = torch.sum(weights * (outputs * targets * 2), axis=axes) + 1e-5
    denom = torch.sum(weights * (outputs * outputs + targets * targets), axis=axes) + 1e-5

    return torch.mean(1 - numer[start_idx:] / (denom[start_idx:] + 1e-8))


def dice_loss_safe(logits, targets, weights=1, start_idx=1, softmax=True, **kwargs):
    weights = torch.as_tensor(weights, device=logits.device)
    outputs = F.softmax(logits, dim=1) if softmax else logits
    axes = [0] + list(range(2, outputs.ndim))
    numer = 2 * torch.sum(weights * (outputs * targets), axis=axes) + 1e-5
    denom = 1 * torch.sum(weights * (outputs + targets), axis=axes) + 1e-5

    return torch.mean(1 - numer[start_idx:] / (denom[start_idx:] + 1e-8))



def surf_dist(logits, targets, p=2, reduction='sum', **kwargs):
    count = torch.zeros(outputs.shape[1], device=logits.device)
    sumsq = torch.zeros(outputs.shape[1], device=logits.device)

    outputs = logits.argmax(dim=1).long().cpu()
    targets = targets.argmax(dim=1).long().cpu()

    for i in range(count.shape[0]):
        h_target = torch.as_tensor(distance_transform_edt(targets != i) - 0) \
                 - torch.as_tensor(distance_transform_edt(targets == i) - 1)

        h_output = torch.as_tensor(distance_transform_edt(outputs != i) - 0) \
                 - torch.as_tensor(distance_transform_edt(outputs == i) - 1)

        count[i] = count[i] + (h_output == 0).sum()
        sumsq[i] = sumsq[i] + (h_target[h_output == 0].abs() ** p).sum()

        count[i] = count[i] + (h_target == 0).sum()
        sumsq[i] = sumsq[i] + (h_output[h_target == 0].abs() ** p).sum()

    if reduction=='none':
        return (sumsq / (count + 1e-10)) ** (1/p)

    if reduction=='mean':
        return (sumsq / (count + 1e-10)).mean() ** (1/p)

    if reduction=='sum':
        return (sumsq.sum() / count.sum()) ** (1/p)

def surf_dist_class(logits, targets, p=2, **kwargs):
    return surf_dist(logits, targets, p, reduction='none', **kwargs)

def haus_dist(logits, targets, p=8, **kwargs):
    return surf_dist(logits, targets, p, **kwargs)
