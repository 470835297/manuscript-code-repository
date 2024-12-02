import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 全局最小矩形框的坐标
global_min_x, global_min_y = 197, 142
global_max_x, global_max_y = 357, 366
GT_dir="D:\\RAU-Net\\inputs\\CT\\train_GT"
# 创建输出目录如果不存在
output_dir = "D:\\RAU-Net\\inputs\\CT\\train_min_rect"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取图像文件名列表，并初始化进度条
image_files = [filename for filename in os.listdir(GT_dir) if filename.endswith(".png")]
pbar = tqdm(image_files)

# 遍历图像目录
for filename in pbar:
    pbar.set_description(f"Processing {filename}")
    img_path = os.path.join(GT_dir, filename)
    
    # 使用PIL读取图像并转换为彩色图（RGB模式）
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    
    # 使用OpenCV绘制矩形框
    color = (0, 255, 0)  # 绿色
    thickness = 2
    img = cv2.rectangle(img, (global_min_x, global_min_y), (global_max_x, global_max_y), color, thickness)
    
    # 保存绘制了矩形框的图像
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print(f"All images have been processed and saved in {output_dir}.")
