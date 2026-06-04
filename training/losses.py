import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4
SMOOTH = 1e-5

def dice_loss(pred, target, smooth=SMOOTH):
    pred = F.softmax(pred, dim=1)

    target_onehot = F.one_hot(target, num_classes=NUM_CLASSES) 
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

    intersection = (pred * target_onehot).sum(dim=(0, 2, 3, 4))
    union = pred.sum(dim=(0, 2, 3, 4)) + target_onehot.sum(dim=(0, 2, 3, 4))

    dice_per_class = (2.0 * intersection + smooth) / (union + smooth) 
    return 1.0 - dice_per_class.mean()

def dice_score(pred, target, smooth=SMOOTH):
    pred_labels = torch.argmax(pred, dim=1) 

    def _region_dice(pred_bin, gt_bin):
        p = pred_bin.contiguous().view(-1).float()
        g = gt_bin.contiguous().view(-1).float()
        inter = (p * g).sum()
        return ((2.0 * inter + smooth) / (p.sum() + g.sum() + smooth)).item()

    wt_pred = pred_labels > 0
    wt_gt = target > 0
    dice_wt = _region_dice(wt_pred, wt_gt)

    tc_pred = (pred_labels == 1) | (pred_labels == 3)
    tc_gt = (target == 1) | (target == 3)
    dice_tc = _region_dice(tc_pred, tc_gt)

    et_pred = pred_labels == 3
    et_gt = target == 3
    dice_et = _region_dice(et_pred, et_gt)

    return (dice_wt + dice_tc + dice_et) / 3.0


def dice_score_regions(pred, target, smooth=SMOOTH):
    pred_labels = torch.argmax(pred, dim=1)

    def _region_dice(pred_bin, gt_bin):
        p = pred_bin.contiguous().view(-1).float()
        g = gt_bin.contiguous().view(-1).float()
        inter = (p * g).sum()
        return ((2.0 * inter + smooth) / (p.sum() + g.sum() + smooth)).item()

    return {
        "WT": _region_dice(pred_labels > 0, target > 0),
        "TC": _region_dice((pred_labels==1)|(pred_labels==3), (target==1)|(target==3)),
        "ET": _region_dice(pred_labels == 3, target == 3),
    }

def iou_score(pred, target, smooth=SMOOTH):
    pred_labels = torch.argmax(pred, dim=1)

    def _region_iou(pred_bin, gt_bin):
        p = pred_bin.contiguous().view(-1).float()
        g = gt_bin.contiguous().view(-1).float()
        inter = (p * g).sum()
        union = p.sum() + g.sum() - inter
        return ((inter + smooth) / (union + smooth)).item()

    iou_wt = _region_iou(pred_labels > 0, target > 0)
    iou_tc = _region_iou((pred_labels==1)|(pred_labels==3), (target==1)|(target==3))
    iou_et = _region_iou(pred_labels == 3, target == 3)

    return (iou_wt + iou_tc + iou_et) / 3.0

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        d_loss = dice_loss(pred, target)
        return ce_loss + d_loss

if __name__ == "__main__":

    pred = torch.randn(1, NUM_CLASSES, 64, 64, 64)
    target = torch.randint(0, NUM_CLASSES, (1, 64, 64, 64))

    loss_fn = SegmentationLoss()
    loss = loss_fn(pred, target)
    dice = dice_score(pred, target)
    iou = iou_score(pred, target)
    regions = dice_score_regions(pred, target)

    print(f"Loss : {loss.item():.4f}")
    print(f"Dice (mean WT/TC/ET): {dice:.4f}")
    print(f"IoU  (mean WT/TC/ET): {iou:.4f}")
    print(f"Dice por región: WT={regions['WT']:.4f}  TC={regions['TC']:.4f}  ET={regions['ET']:.4f}")
