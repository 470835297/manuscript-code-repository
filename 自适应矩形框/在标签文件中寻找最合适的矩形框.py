import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 初始化变量用于存储全局最小矩形框的坐标
global_min_x = float('inf')
global_min_y = float('inf')
global_max_x = 0
global_max_y = 0
img_dir="D:\\RAU-Net\\inputs\\CT\\test_GT"
# 创建输出目录如果不存在
output_dir = "D:\\RAU-Net\\inputs\\CT\\test_GT_min_rect"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开输出txt文件
with open(os.path.join(output_dir, 'rect_test.txt'), 'w') as f:
    # 获取图像文件名列表，并初始化进度条
    image_files = [filename for filename in os.listdir(img_dir) if filename.endswith(".png")]
    pbar = tqdm(image_files)
    
    # 遍历图像目录
    for filename in pbar:
        pbar.set_description(f"Processing {filename}")
        img_path = os.path.join(img_dir, filename)
        
        # 使用PIL读取图像并转换为灰度图
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        
        # 使用OpenCV找到白色区域的轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 初始化图像内最小矩形框的坐标
        img_min_x = float('inf')
        img_min_y = float('inf')
        img_max_x = 0
        img_max_y = 0
        
        # 计算图像内的最小矩形框
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            img_min_x = min(img_min_x, x)
            img_min_y = min(img_min_y, y)
            img_max_x = max(img_max_x, x + w)
            img_max_y = max(img_max_y, y + h)
        
        # 写入到输出txt文件
        f.write(f"{filename} 1 {img_min_x} {img_min_y} {img_max_x} {img_max_y}\n")
        
        # 更新全局最小矩形框的坐标
        global_min_x = min(global_min_x, img_min_x)
        global_min_y = min(global_min_y, img_min_y)
        global_max_x = max(global_max_x, img_max_x)
        global_max_y = max(global_max_y, img_max_y)




# 如果您知道具体值，可以直接替换
image_max_x = 512  # 请替换为您的图像的实际最大宽度
image_max_y = 512  # 请替换为您的图像的实际最大高度
# 输出全局最小矩形框的坐标，并外扩10
expand_value = 10
global_min_x = max(global_min_x - expand_value, 0)
global_min_y = max(global_min_y - expand_value, 0)
global_max_x = min(global_max_x + expand_value, image_max_x)
global_max_y = min(global_max_y + expand_value, image_max_y)


# 计算当前矩形框的尺寸
current_width = global_max_x - global_min_x
current_height = global_max_y - global_min_y

# 找到一个最接近但大于或等于这个尺寸的、同时又是64的整数倍的数
factor = 64  # 2的6次方
new_dimension = max(current_width, current_height)
new_dimension = ((new_dimension + factor - 1) // factor) * factor

# 使用这个新的尺寸作为矩形框的长和宽
global_max_x = global_min_x + new_dimension
global_max_y = global_min_y + new_dimension

# 确保这些值不超过图像的最大尺寸
global_max_x = min(global_max_x, image_max_x)
global_max_y = min(global_max_y, image_max_y)
# 这里结束

# 输出调整后的全局最小矩形框的坐标
print(f"Adjusted global minimum rectangle coordinates: ({global_min_x}, {global_min_y}) to ({global_max_x}, {global_max_y})")
