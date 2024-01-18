import sys, math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as edt


def MeanDice(y_pred, y, exclude_zero=False):
    if exclude_zero:
        foreground_mask = (y_pred!=0) | (y!=0)
        num = (y_pred[foreground_mask]==y[foreground_mask]).sum()
        denom = 2 * foreground_mask.sum()
    else:
        num = (y_pred==y).sum()
        denom = y.numel() + y_pred.numel()
        
    dice = 2 * num/denom
    return dice.item()


def MeanDice2(y_pred, y, exclude_zero=True):
    if exclude_zero:
        foreground_mask = (y_pred!=0) | (y!=0)
        num = (y_pred[foreground_mask]==y[foreground_mask]).sum()
        denom = 2 * foreground_mask.sum()
    else:
        num = (y_pred==y).sum()
        denom = y.numel() + y_pred.numel()

    dice = 2 * num/denom
    return dice.item()



def LabelDice(y_pred, y, label_list:list[int]=None):
    labels = y.unique.numpy() if label_list is None else label_list
    dice = np.zeros(labels.size, 1)
    
    for i in range(labels):
        num = ((y_pred==labels[i]) * (y==labels[i])).sum()
        denom = (y_pred==labels[i]).sum() + (y==labels[i]).sum()
        dice[i] = 2 * num / denom

    return dice.item()



def HausDist(y_pred, y, p=2, reduction='sum', **kwargs):
    count = torch.zeros(y_pred.shape[1], device=y_pred.device)
    sumsq = torch.zeros(y_pred.shape[1], device=y_pred.device)

    for i in range(count.shape[0]):
        h_y = torch.from_numpy(edt(y.cpu()!=i) - edt(y.cpu()==i) + 1).to(device=y_pred.device)
        h_y_pred = torch.from_numpy(edt(y_pred.cpu()!=i) - edt(y_pred.cpu()==i) + 1).to(device=y_pred.device)

        count[i] = count[i] + (h_y_pred==0).sum()
        sumsq[i] = sumsq[i] + (h_y[h_y_pred==0].abs() ** p).sum()

        count[i] = count[i] + (h_y==0).sum()
        sumsq[i] = sumsq[i] + (h_y_pred[h_y==0].abs() ** p).sum()

    if reduction=='none':
        return (sumsq / (count + 1e-10)) ** (1/p)

    if reduction=='mean':
        return (sumsq / (count + 1e-10)).mean() ** (1/p)

    if reduction=='sum':
        return (sumsq.sum() / count.sum()) ** (1/p)
