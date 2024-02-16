import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):

    def __init__(self, num_classes, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        if inputs.shape != targets.shape:
            # B W H
            targets = nn.functional.one_hot(targets,
                                            num_classes=self.num_classes)
            # B W H NUM_CLS 
            targets = targets.view(inputs.shape)
            # B NUM_CLS W H
        targets = targets.float()
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma,
                                  self.reduction)



def dice_loss(pred, ref, nCls, average, eps: float = 1e-8):
    # Input tensors will have shape (Batch, Class)
    # Dimension 0 = batch
    # Dimension 1 = class code or predicted logit
    
    # compute softmax over the classes axis to convert logits to probabilities
    pred_soft = torch.softmax(pred, dim=1)

    # create reference one hot tensors
    ref_one_hot = F.one_hot(ref, num_classes = nCls)
    ref_one_hot = ref_one_hot.view(pred_soft.shape)

    #Calculate the dice loss
    if average == "micro":
        #Use dim=1 to aggregate results across all classes
        intersection = torch.sum(pred_soft * ref_one_hot, dim=1)
        cardinality = torch.sum(pred_soft + ref_one_hot, dim=1)
    else:
        #With no dim argument, will be calculated separately for each class
        intersection = torch.sum(pred_soft * ref_one_hot)
        cardinality = torch.sum(pred_soft + ref_one_hot)

    dice_score = 2.0 * intersection / (cardinality + eps)
    dice_loss = -dice_score + 1.0

    # reduce the loss across samples (and classes in case of `macro` averaging)
    dice_loss = torch.mean(dice_loss)

    return dice_loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes, average="micro", eps: float = 1e-8) -> None:
        super().__init__()
        self.nCls = num_classes
        self.average = average
        self.eps = eps

    def forward(self, pred, ref):
        return dice_loss(pred, ref, self.nCls, self.average, self.eps)