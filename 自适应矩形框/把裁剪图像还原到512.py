import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# 原图像的尺寸
original_size = (512, 512)

# # 裁剪区域的坐标 224*224
# global_min_x, global_min_y = 165 ,142     
# global_max_x, global_max_y = 389 ,366
#裁剪区域的坐标 160*224
global_min_x, global_min_y = 197, 142
global_max_x, global_max_y = 357, 366

# 输入和输出文件夹路径
input_folder = r'D:\RAU-Net\save_second_SEA_Unet_5'
output_folder = r'D:\RAU-Net\save_second_SEA_Unet_5_512'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有图像路径
image_paths = glob.glob(os.path.join(input_folder, '*.png'))

for img_path in tqdm(image_paths, desc="Processing images"):
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保持原始通道

    # 检查图像通道数
    if len(img.shape) == 2:
        # 单通道图像
        restored_img = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        restored_img[global_min_y:global_max_y, global_min_x:global_max_x] = img
    else:
        # 多通道图像
        restored_img = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
        restored_img[global_min_y:global_max_y, global_min_x:global_max_x] = img

    # 保存还原后的图像
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, restored_img)
