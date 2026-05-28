import torch
import torch.nn as nn
import torch.nn.functional as F


# Número de clases BraTS
NUM_CLASSES = 4

# Smooth term
SMOOTH = 1e-5

def dice_loss(pred, target, smooth=SMOOTH):
    # SOFTMAX
    # Shape:
    # [B, C, D, H, W]

    pred = F.softmax(
        pred,
        dim=1
    )

    # ONE HOT ENCODING
    # Shape:
    # [B, D, H, W] -> [B, C, D, H, W]

    target_onehot = F.one_hot(
        target,
        num_classes=NUM_CLASSES
    )

    target_onehot = target_onehot.permute(
        0,
        4,
        1,
        2,
        3
    ).float()

    intersection = (
        pred * target_onehot
    ).sum()

    union = (
        pred.sum() +
        target_onehot.sum()
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        union + smooth
    )

    return 1.0 - dice


def dice_score(pred, target, smooth=SMOOTH):

    pred = torch.argmax(
        pred,
        dim=1
    )

    pred = pred > 0
    target = target > 0
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).float().sum()

    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()

def iou_score(pred,target, smooth=SMOOTH):

    pred = torch.argmax(
        pred,
        dim=1
    )
    pred = pred > 0

    target = target > 0

    pred = pred.contiguous().view(-1)

    target = target.contiguous().view(-1)

    intersection = (
        pred * target
    ).float().sum()

    union = (
        pred.sum() +
        target.sum() -
        intersection
    )

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()

class SegmentationLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        pred,
        target
    ):

        ce_loss = self.ce(
            pred,
            target
        )

        d_loss = dice_loss(
            pred,
            target
        )

        total = ce_loss + d_loss

        return total

if __name__ == "__main__":

    pred = torch.randn(
        1,
        NUM_CLASSES,
        64,
        64,
        64
    )

    target = torch.randint(
        0,
        NUM_CLASSES,
        (1, 64, 64, 64)
    )

    loss_fn = SegmentationLoss()

    loss = loss_fn(
        pred,
        target
    )

    dice = dice_score(
        pred,
        target
    )

    iou = iou_score(
        pred,
        target
    )

    print("Loss :", loss.item())

    print("Dice:", dice)

    print("IoU  :", iou)