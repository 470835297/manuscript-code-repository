import numpy as np
import torch
import torch.nn.functional as F

# 定义一个iou_score函数，输入（输出，目标）
# def iou_score(output, target):
#     # 平滑因子，防止0除
#     smooth = 1e-5

#     # 如果output是一个pytorch张量，执行
#     if torch.is_tensor(output):  
#         # 对输出使用sigmoid激活函数，并把tensor转换为numpy格式，output.shape:(2, 1, 256, 256)
#         output = torch.sigmoid(output).data.cpu().numpy()
#     # 如果target是一个pytorch张量，执行
#     if torch.is_tensor(target):  
#         # target.shape:(2, 1, 256, 256)
#         target = target.data.cpu().numpy()
#     # 将输出和目标二值化，阈值为0.5
#     output_ = output > 0.2
#     target_ = target > 0.2
#     # 计算
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()

#     return (intersection + smooth) / (union + smooth)


def iou_score(output, target):
    # 平滑因子，防止0除
    smooth = 1e-5

    # 如果output是一个pytorch张量，执行
    if torch.is_tensor(output):  
        # 对输出使用sigmoid激活函数，并把tensor转换为numpy格式，output.shape:(2, 1, 256, 256)
        output = torch.sigmoid(output).data.cpu().numpy()
    # 如果target是一个pytorch张量，执行
    if torch.is_tensor(target):  
        # target.shape:(2, 1, 256, 256)
        target = target.data.cpu().numpy()
    
    # 将输出和目标二值化，阈值为0.2
    output_ = (output > 0.2).astype(np.int64)
    target_ = (target > 0.2).astype(np.int64)
    
    # 计算
    intersection = (output_ * target_).sum()
    union = output_.sum() + target_.sum() - intersection

    return (intersection + smooth) / (union + smooth)

# 定义一个dice系数函数，输入（输出，目标）
def dice_coef(output, target):
    # 平滑因子，防止0除
    smooth = 1e-5
    
    # 对输出使用sigmoid激活函数
    output = torch.sigmoid(output)
    
    # 二值化处理
    threshold = 0.65  # 设置阈值，您可以根据需要调整这个值
    output = (output > threshold).float()
    target = (target > threshold).float()  # 对target也进行二值化
    # 将tensor转换为numpy格式，view函数将一个多行的Tensor拼接成一行
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    # 计算
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def acc_coef(output, target):
    # 平滑因子，防止0除
    smooth = 1e-5
    # # 二值化处理
    # threshold = 0.2  # 设置阈值，您可以根据需要调整这个值
    # output = (output > threshold).float()
    # # 对输出使用sigmoid激活函数，并把tensor转换为numpy格式 view函数将一个多行的Tensor拼接成一行
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    # 计算
    tp = (output * target).sum()
    tn = ((1-output) * (1-target)).sum()
    fp = (output * (1-target)).sum()
    fn = ((1-output) * target).sum()

    return (tp+tn + smooth) / \
        (fp+fn+tp+tn + smooth)

def sensi_coef(output, target):
    # 平滑因子，防止0除
    smooth = 1e-5
    # # 二值化处理
    # threshold = 0.2  # 设置阈值，您可以根据需要调整这个值
    # output = (output > threshold).float()
    # # 对输出使用sigmoid激活函数，并把tensor转换为numpy格式 view函数将一个多行的Tensor拼接成一行
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    # 计算
    tp = (output * target).sum()
    tn = ((1-output) * (1-target)).sum()
    fp = (output * (1-target)).sum()
    fn = ((1-output) * target).sum()

    return (tp + smooth) / \
        (tp+fn + smooth)

def spec_coef(output, target):
    # 平滑因子，防止0除
    smooth = 1e-5
    # # 二值化处理
    # threshold = 0.2  # 设置阈值，您可以根据需要调整这个值
    # output = (output > threshold).float()
    # # 对输出使用sigmoid激活函数，并把tensor转换为numpy格式 view函数将一个多行的Tensor拼接成一行
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    # 计算
    tp = (output * target).sum()
    tn = ((1-output) * (1-target)).sum()
    fp = (output * (1-target)).sum()
    fn = ((1-output) * target).sum()

    return (tn + smooth) / \
        (tn+fp + smooth)