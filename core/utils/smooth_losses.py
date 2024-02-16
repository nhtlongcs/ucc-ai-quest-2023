import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from segmentation_models_pytorch.losses import (JaccardLoss, DiceLoss, FocalLoss, LovaszLoss,
                                                SoftBCEWithLogitsLoss, SoftCrossEntropyLoss)


def _convert_to_onehot_labels(seg_label, num_classes):
    """Convert segmentation label to onehot.

    Args:
        seg_label (Tensor): Segmentation label of shape (N, H, W).
        num_classes (int): Number of classes.

    Returns:
        Tensor: Onehot labels of shape (N, num_classes).
    """

    batch_size = seg_label.size(0)
    onehot_labels = seg_label.new_zeros((batch_size, num_classes))
    for i in range(batch_size):
        hist = seg_label[i].float().histc(
            bins=num_classes, min=0, max=num_classes - 1)
        onehot_labels[i] = hist > 0
    return onehot_labels


def mask_to_onehot(mask, num_classes):
    """
    Convert a segmentation mask (B, H, W) to (B, C, H, W) one-hot mask.
    """
    one_hot = torch.zeros((mask.shape[0], num_classes, *mask.shape[1:]), dtype=torch.float, device=mask.device)
    for i in range(num_classes):
        one_hot[:, i, :, :] = (mask == i)
    return one_hot


def onehot_to_mask(one_hot):
    """
    Convert a one-hot mask (B, C, H, W) to a segmentation mask (B, H, W).
    """
    mask = torch.argmax(one_hot, dim=1)
    return mask


class MultiLoss(nn.Module):
    def __init__(self, weights,
                 convert_to_onehot=False,
                 mode='binary',
                 smooth_factor=0.0,
                 **kwargs):
        super().__init__()

        self.weights = weights
        self.ce = SoftCrossEntropyLoss(smooth_factor=smooth_factor)
        self.bce = SoftBCEWithLogitsLoss(smooth_factor=smooth_factor)
        self.dice = DiceLoss(mode=mode)
        self.focal = FocalLoss(mode=mode)
        self.jaccard = JaccardLoss(mode=mode)
        self.lovasz = LovaszLoss(mode=mode)

        self.mapping = {'bce': self.bce,
                        'ce': self.ce,
                        'dice': self.dice,
                        'focal': self.focal,
                        'jaccard': self.jaccard,
                        'lovasz': self.lovasz,
                        }
        self.mode = mode
        self.convert_to_onehot = convert_to_onehot
        self.smooth_factor = smooth_factor

    def forward(self, outputs, targets, return_dict=False):
        if self.convert_to_onehot:
            targets = mask_to_onehot(targets, outputs.size(1))
        loss = 0
        loss_dict = {}
        for name, weight in self.weights.items():
            # loss += weight * self.mapping[name](outputs, targets)
            loss_dict['loss_' + name] = self.mapping[name](outputs, targets)
            loss += weight * loss_dict['loss_' + name]

        loss = loss.clamp(min=1e-5)
        if return_dict:
            return loss, loss_dict
        else:
            return loss
