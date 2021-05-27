import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        '''
        args:
            pred:(B,C,Z,X,Y) or (B,C,H,W)
            target:(B,Z,X,Y) or (B,H,W)
        '''
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        intersection = pred * target
        dice_coef = (2.0 * intersection.sum(1)+self.smooth) / \
            (pred.sum(1) + target.sum(1) + self.smooth)

        return 1 - dice_coef.mean()


def Dice(pred: torch.Tensor, target: torch.Tensor, smooth=1) -> float:
    batch_size = pred.size(0)
    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)

    intersection = pred*target
    dice_coef = 2.0 * (intersection.sum(1)+smooth) / \
        (pred.sum(1)+target.sum(1)+smooth)
    return dice_coef.mean()


def Jaccard(pred: torch.Tensor, target: torch.Tensor, smooth=1) -> float:
    batch_size = pred.size(0)
    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)

    intersection = pred*target
    jaccard_coef = (intersection.sum(1)+smooth)/(pred.sum(1) +
                                                 target.sum(1)-intersection.sum(1)+smooth)
    return jaccard_coef.mean()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.view(-1)
    target = target.view(-1)
    TP = pred*target
    acc = TP.sum()/target.sum()
    return acc
