from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','FocalTverskyLoss','CombinedLoss', 'DiceFocalLoss','CombinedLoss123','DiceLoss','FocalLoss','HBD_Loss']


def hausdorff_distance_loss(preds, targets, scale_factor=1e-3):
    # 计算预测和目标之间的距离
    preds_dist = torch.cdist(preds, targets, p=2)
    targets_dist = torch.cdist(targets, preds, p=2)

    # 计算距离的平均值，而不是平方，以提高稳定性
    mean_preds_dist = torch.mean(preds_dist)
    mean_targets_dist = torch.mean(targets_dist)

    # 应用缩放因子
    scaled_loss = scale_factor * (mean_preds_dist + mean_targets_dist)

    return scaled_loss


# 定义一个BCEDiceLoss类
class BCEDiceLoss(nn.Module):
    # 初始化方法继承父类
    def __init__(self):
        super().__init__()

    # 定义前向传播方法，输入（对象，输入，目标）
    def forward(self, input, target):
        # 使用逻辑回归的二值交叉熵计算 输入，目标的bce
        bce = F.binary_cross_entropy_with_logits(input, target)
        #print("bce=",bce)
        # 平滑因子，防止0除
        smooth = 1e-5
        # 对输入使用sigmoid激活函数
        input = torch.sigmoid(input)
        # 获取num
        num = target.size(0)
        # view函数将一个多行的Tensor拼接成一行
        input = input.view(num, -1)
        target = target.view(num, -1)
        # 计算
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice_loss = 1 - dice.sum() / num
        #print("dice_loss=,",dice_loss)
        #return  bce
        return  0.9*bce+0.1*dice_loss 
        #return  bce + 2*dice_loss  



# 定义组合损失类
class HBD_Loss(nn.Module):
    def __init__(self, alpha=0.9, beta=0.1):
        super().__init__()
        self.bce_dice_loss = BCEDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target, preds_boundary, targets_boundary):
        loss_bce_dice = self.bce_dice_loss(input, target)
        loss_hausdorff = hausdorff_distance_loss(preds_boundary, targets_boundary)
        return self.alpha * loss_bce_dice + self.beta * loss_hausdorff



class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, normalizer):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        preds = torch.sigmoid(logits)
        pt = (1 - preds) * targets + preds * (1 - targets)
        F_loss = alpha * (1 - pt).pow(gamma) * BCE_loss / normalizer

        return F_loss.sum()

    def forward(self, input, target):
        smooth = 1e-5
        input = input.view(-1)
        target = target.view(-1)
        
        # Dice Loss
        input_soft = torch.sigmoid(input)
        intersection = (input_soft * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (input_soft.sum() + target.sum() + smooth)

        # Focal Loss
        normalizer = target.sum()
        if normalizer == 0:
            normalizer = input.size()[0]
        
        focal_loss = self.focal_loss_with_logits(input, target, self.alpha, self.gamma, normalizer)
        
        # Combined Loss
        combined_loss = 0.2 * dice_loss + 0.8 * focal_loss  # Adjust the weights as per your requirement

        return combined_loss


# Define the custom loss function class with Tversky Loss and Focal Loss
class TverskyFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, beta=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, normalizer):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        preds = torch.sigmoid(logits)
        pt = (1 - preds) * targets + preds * (1 - targets)
        F_loss = alpha * (1 - pt).pow(gamma) * BCE_loss / normalizer

        return F_loss.sum()

    def tversky_loss(self, input, target, alpha, beta):
        smooth = 1e-5
        input_soft = torch.sigmoid(input)
        intersection = (input_soft * target).sum()
        fp = (input_soft * (1 - target)).sum()
        fn = ((1 - input_soft) * target).sum()
        return 1 - (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        
        # Tversky Loss
        tversky_loss = self.tversky_loss(input, target, self.alpha, self.beta)
        
        # Focal Loss
        normalizer = target.sum()
        if normalizer == 0:
            normalizer = input.size()[0]
        
        focal_loss = self.focal_loss_with_logits(input, target, self.alpha, self.gamma, normalizer)
        
        # Combined Loss
        combined_loss = 0.2 * tversky_loss + 0.8 * focal_loss  # Adjust the weights as per your requirement

        return combined_loss





"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
 

 
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
 
 
#函数输入是一个排过序的 标签组  越靠近前面的标签 表示这个像素点与真值的误差越大
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    # print("p = ", p)
    # print("gt_sorted = ", gt_sorted)
    gts = gt_sorted.sum()#求个和
    #gt_sorted.float().cumsum(0) 1维的 计算的是累加和 例如 【1 2 3 4 5】 做完后就是【1 3 6 10 15】
    #这个intersection是用累加和的值按维度减 累加数组的值，目的是做啥呢  看字面是取交集
    intersection = gts - gt_sorted.float().cumsum(0) #对应论文Algorithm 1的第3行
    union = gts + (1 - gt_sorted).float().cumsum(0) #对应论文Algorithm 1的第4行
    jaccard = 1. - intersection / union #对应论文Algorithm 1的第5行
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]#对应论文Algorithm 1的第7行
    return jaccard
 
 
def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou
 
 
def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)
 
 
# --------------------------- BINARY LOSSES ---------------------------
 
 
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss
 
 
def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss
 
 
def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore == None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
 
 
class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()
 
 
def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss
 
 
# --------------------------- MULTICLASS LOSSES ---------------------------
 
 
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    print("probas.shape = ", probas.shape)
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
    	  #lovasz_softmax_flat的输入就是probas 【262144 2】  labels【262144】
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss
 
 
#这个函数是计算损失函数的部位
def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    #预测像素点个数，一张512*512的图
    if probas.numel() == 0:#返回数组中元素的个数
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)#获得通道数呗  就是预测几类
    losses = []
    #class_to_sum = [0 1]  类的种类总数 用list存储
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c  如果语义标注数据与符合第c类，fg中存储1.0样数据
        if (classes == 'present') and (fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]#取出第c类预测值 是介于 0~1之间的float数
        #errors 是预测结果与标签结果差的绝对值
        errors = (Variable(fg) - class_pred).abs()
        #对误差排序 从大到小排   perm是下标值 errors_sorted 是排序后的预测值
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
 
        perm = perm.data
        #排序后的标签值
        fg_sorted = fg[perm]
        
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)
 
 
def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    #在这维度为probas 【1 2 512 512】  labels维度为【1 1 512 512】
    if probas.dim() == 3:#dim()数组维度
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()#数组维度
    #维度交换并变形   将probas.permute(0, 2, 3, 1)变换后的前3维合并成1维，通道不变
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    #
    labels = labels.view(-1)
    #我的代码是用默认值 直接返回了 probas  labels  两个压缩完事的东西  
    #在这维度为probas 【262144 2】  labels维度为【262144】
    
    if ignore == None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels
 
def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)
 
 
# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n



# 定义一个LovaszHingeLoss类
class LovaszHingeLoss(nn.Module):
    # 继承父类的初始化方法
    def __init__(self):
        super().__init__()

    # 定义前向传播方法，输入（对象，输入，目标）
    def forward(self, input, target):
        # squeeze（a，N）对数据的维度进行压缩，把a中指定的N维维数为1的维度去掉
        input = input.squeeze(1)
        target = target.squeeze(1)
        # 调用loss.py里的lovasz_hinge函数计算loss
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
    


class fun_loss(nn.Module):
    def __init__(self):
        super(fun_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.flatten = nn.Flatten()
    def forward(self, true, pred):
        avg_criterion = torch.mean(self.criterion(true, pred))
        y_true_pos = self.flatten(true)
        y_pred_pos = self.flatten(pred)
        true_pos = torch.sum(y_true_pos*y_pred_pos)
        flase_neg = torch.sum(y_true_pos*(1-y_pred_pos))
        false_pos = torch.sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.4
        smooth = 1.0
        de_tversky = (true_pos+smooth)/(true_pos+alpha*flase_neg+(1-alpha)*false_pos+smooth)
        tversky = 1-de_tversky
        return avg_criterion+tversky
    


import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入所需的库
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        """
        初始化 DiceLoss 类
        参数:
            eps (float): 用于保证数值稳定性的小常数
        """
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        前向传播计算 Dice Loss
        参数:
            pred (Tensor): 模型的预测输出，形状为 (batch_size, num_classes, height, width)
            target (Tensor): 真实标签，形状为 (batch_size, num_classes, height, width)
        返回:
            dice_loss (Tensor): Dice Loss 的值
        """
        # 计算交集和并集
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)

        # 计算 Dice Loss
        dice_loss = 1 - (2 * intersection + self.eps) / (union + self.eps)
        
        return dice_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=5.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = (1 - p_t)**self.gamma * bce_loss
        return focal_loss.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

    def forward(self, pred, target):
        tversky = TverskyLoss(alpha=self.tversky_alpha, beta=self.tversky_beta)(pred, target)
        focal = FocalLoss(gamma=self.gamma)(pred, target)
        return self.alpha * tversky + self.beta * focal




# 专门针对Mult_SEA_SOT网络的loss
class CombinedLoss(nn.Module):
    def __init__(self, lambda_1=0.1, lambda_2=0.9):
        super(CombinedLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.seg_criterion = BCEDiceLoss() 
        self.small_object_criterion = nn.CrossEntropyLoss()

    def forward(self, pred_seg, target_seg, pred_small_object, target_small_object):
        seg_loss = self.seg_criterion(pred_seg, target_seg)
        #print("seg_loss=" ,seg_loss)
        small_object_loss = self.small_object_criterion(pred_small_object, target_small_object)
        #print("small_object_loss=" ,small_object_loss)
        return self.lambda_1 * seg_loss + self.lambda_2 * small_object_loss


# nn.BCEWithLogitsLoss()，BCEDiceLoss()
# 专门针对Mult_SEA_SOT123网络的loss 深度监督
class CombinedLoss123(nn.Module):
    def __init__(self, lambda_1=0.5, lambda_2=0.3, lambda_3=0.1, lambda_4=0.1):
        super(CombinedLoss123, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.seg_criterion = nn.BCEWithLogitsLoss()
        self.small_object_criterion = nn.CrossEntropyLoss()

    def forward(self, pred_seg, target_seg, pred_small_object1, target_small_object1, pred_small_object2, target_small_object2, pred_small_object3, target_small_object3):
        seg_loss = self.seg_criterion(pred_seg, target_seg)
        small_object_loss1 = self.small_object_criterion(pred_small_object1, target_small_object1)
        small_object_loss2 = self.small_object_criterion(pred_small_object2, target_small_object2)
        small_object_loss3 = self.small_object_criterion(pred_small_object3, target_small_object3)
        return self.lambda_1 * seg_loss + self.lambda_2 * small_object_loss1+ self.lambda_3 * small_object_loss2+ self.lambda_4 * small_object_loss3
