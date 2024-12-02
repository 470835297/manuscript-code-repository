import torch
import os
import argparse
import yaml
import tqdm
import dataset
import cv2

import numpy as np
import archs
import losses
from medpy import metric
from dataset import Dataset

import pandas as pd
import torch.backends.cudnn as cudnn
import albumentations as albu

from glob import glob
from tqdm import tqdm
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from collections import OrderedDict

from torch.optim import lr_scheduler
from utils import AverageMeter, str2bool
from metrics import *
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf



from data import *

MODEL_FILE = archs.__all__
LOSS_NAMES = losses.__all__

torch.cuda.current_device()
torch.cuda._initialized = True
#MODEL_FILE = archs.__all__




def predict():

    out_dir = 'test_256_adaptive_CLAHE/'
    model = archs.__dict__['SEA_Unet_7'](num_classes=1)   #修改模型
    model = model.cuda()
    model.eval()  # 设置模型为评估模式
    model.load_state_dict(torch.load('D:\\RAU-Net\\model\\CLNHE_SEA_Unet_7_CT\\model.pth')) #修改模型路径

    val_img_ids = open('inputs/CT/test_256_adaptive_edges.txt').read().strip().split()                       #修改验证集路径
    val_transform = Compose([
        albu.Resize(256, 256),
        albu.Normalize(),
    ])
    val_dataset = dataset.Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'CT', 'test_256_adaptive_CLAHE'),
        mask_dir=os.path.join('inputs', 'CT', 'test_GT_256_adaptive'),
        img_ext='.png',# 这里需要根据图像格式更改，jpg，png
        mask_ext='.png',# 这里需要根据图像格式更改，jpg，png
        transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        # shuffle=True,
        drop_last=False)
    
    with torch.no_grad():
        # 初始化累积变量
        total_dice = 0.0
        total_spec = 0.0
        total_sensi = 0.0
        total_acc = 0.0
        total_iou = 0.0
        num_samples = 0  # 计数器，记录处理过的样本数量
  
        a=[]
        
        pbar = tqdm(total=len(val_loader))

        for input, target, img_id in val_loader:
            input = input.cuda()
            target = target.cuda()
            #output,small_target = model(input) # 用Mult_SEA_SOT的时候用
            output = model(input)
            # 将输出和目标转移到CPU
            # output = output.cpu()
            # target = target.cpu()
            # 累加每个指标
            
            total_dice += dice_coef(output, target)
            total_spec += spec_coef(output, target)
            total_sensi += sensi_coef(output, target)
            total_acc += acc_coef(output, target)
            total_iou += iou_score(output, target)

            #深监督用这个
            # total_dice += dice_coef(output[-1], target)
            # total_spec += spec_coef(output[-1], target)
            # total_sensi += sensi_coef(output[-1], target)
            # total_acc += acc_coef(output[-1], target)
            # total_iou += iou_score(output[-1], target)

            num_samples += 1  # 更新样本数量

            # 假设 out_img 是一个PyTorch Tensor
            out_img = output.cpu().numpy().squeeze(0)  # 确保将其移至CPU并移除批处理维度

            #深监督用这个
            # out_img = output[-1].cpu().numpy().squeeze(0)  # 确保将其移至CPU并移除批处理维度
            out_img = torch.sigmoid(torch.tensor(out_img))  # 应用sigmoid激活函数
            
            out_img = (out_img > 0.65)  # 应用阈值

            # 将布尔Tensor转换为uint8类型的二值图像
            out_img = out_img.to(torch.uint8) * 255

            # 如果你需要将结果保存为图像或进行其他处理，可能需要将其转换回NumPy数组
            out_img = out_img.numpy()

            # print(out_img.shape)  # 查看形状
            # print(out_img.dtype)  # 查看数据类型

            if not os.path.exists(out_dir): #如果保存测试结果的文件夹不存在则创建
                os.makedirs(out_dir)
            out_img = out_img.squeeze(0)  # 去除单一维度
            cv2.imwrite(os.path.join(out_dir, img_id['img_id'][0] + '.png'), out_img) 
            #out_img.save()
            pbar.update(1)
        pbar.close()
        # 计算平均指标值
        avg_dice = total_dice / num_samples
        avg_spec = total_spec / num_samples
        avg_sensi = total_sensi / num_samples
        avg_acc = total_acc / num_samples
        avg_iou = total_iou / num_samples

        # 打印平均指标值
        print(f"Average Dice Coefficient: {avg_dice}")
        print(f"Average Specificity Coefficient: {avg_spec}")
        print(f"Average Sensitivity Coefficient: {avg_sensi}")
        print(f"Average Accuracy Coefficient: {avg_acc}")
        print(f"Average IOU Score: {avg_iou}")

if __name__ == '__main__':
    predict()