import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import cv2


# from utils.metrics import calculate_area, mean_dice, mean_iou, accuracy


def ce_loss(pred: Tensor, label: Tensor):
    """
    使用 nn.CrossEntropyLoss 计算交叉熵
    :param pred: N * C * H * W
    :param label: N * H * W
    :return: 交叉熵损失值
    """
    return nn.CrossEntropyLoss()(pred, label) * 2


def calculate_area(pred: Tensor, label: Tensor, num_classes: int = 2):
    """
    计算 preds 和 labels 的各类的公共区域，以及各类的区域
    :param pred: N C H W
    :param label: N H W
    :param num_classes: 2
    :return:
    """
    # convert the label to onehot
    label = F.one_hot(label, 2).permute(0, 3, 1, 2).float()  # N * C * H * W ,
    pred = F.softmax(pred, dim=1).float()  # N * C * H * W
    inter = label * pred

    label_area = torch.sum(label, dim=2)
    label_area = torch.sum(label_area, dim=2)  # N * C
    pred_area = torch.sum(pred, dim=2)
    pred_area = torch.sum(pred_area, dim=2)  # N * C
    intersect_area = torch.sum(inter, dim=2)
    intersect_area = torch.sum(intersect_area, dim=2)  # N * C

    return intersect_area, pred_area, label_area


def dice_loss(pred: Tensor, label: Tensor, eps=1e-5):
    """
    计算dice loss
    :param pred:
    :param label:
    :param eps:
    :return:
    """
    intersect_area, pred_area, label_area = calculate_area(pred, label)
    diceloss = intersect_area * 2 / (label_area + pred_area + eps)
    diceloss = torch.mean(diceloss, dim=1)  # 对类别dice取平均
    diceloss = torch.mean(diceloss, dim=0)  # 对Batch dice取平均
    # diceloss = 1 - diceloss
    diceloss = -torch.log(diceloss)

    return diceloss


def iou_loss(pred: Tensor, label: Tensor, eps=1e-5):
    """
    计算iou loss
    :param pred:
    :param label:
    :param eps:
    :return:
    """
    intersect_area, pred_area, label_area = calculate_area(pred, label)
    iouloss = intersect_area / (label_area + pred_area - intersect_area + eps)
    iouloss = torch.mean(iouloss, dim=1)  # 对类别dice取平均
    iouloss = torch.mean(iouloss, dim=0)  # 对Batch dice取平均
    # iouloss = 1 - iouloss
    iouloss = -torch.log(iouloss)

    return iouloss


def ce_dice_loss(pred: Tensor, label: Tensor):
    """
    ce + dice 损失
    :param pred:
    :param label:
    :return:
    """
    diceloss = dice_loss(pred, label)
    celoss = ce_loss(pred, label)
    return celoss + diceloss


def ce_dice_iou_loss(pred: Tensor, label: Tensor):
    """
    ce + dice + iou 损失
    :param pred:
    :param label:
    :return:
    """
    diceloss = dice_loss(pred, label)
    celoss = ce_loss(pred, label)
    iouloss = iou_loss(pred, label)
    return celoss + diceloss + iouloss


if __name__ == "__main__":
    preds = cv2.imread("../preout/pre_6.png", cv2.IMREAD_GRAYSCALE)
    preds[preds > 1] = 1
    labels = cv2.imread("../preout/label_6.png", cv2.IMREAD_GRAYSCALE)
    labels[labels > 1] = 1
    preds = torch.tensor(preds).unsqueeze(0).long()
    preds = F.one_hot(preds, 2).float().permute(0, 3, 1, 2)
    labels = torch.tensor(labels).unsqueeze(0).long()

    dicelossv = ce_dice_iou_loss(preds, labels)

    print(dicelossv)
