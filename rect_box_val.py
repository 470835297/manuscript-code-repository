import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm  

# 计算IoU的函数
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


class RectBoxDataset(Dataset):
    def __init__(self, img_folder, txt_file):
        self.img_folder = img_folder
        self.coordinates = {}
        self._load_coordinates(txt_file)
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]
        
    def _load_coordinates(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                img_file_name = parts[0]
                # 忽略类别标签（parts[1]），仅使用坐标信息
                coordinates = list(map(int, parts[2:]))
                self.coordinates[img_file_name] = coordinates
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = Image.open(os.path.join(self.img_folder, img_file)).convert('RGB')
        img = np.array(img)
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # 归一化到 [0, 1]

        box = self.coordinates.get(img_file, [0, 0, 0, 0])
        box = torch.tensor(box).float() / 512.0  # 归一化到 [0, 1]，假设图像尺寸为 512x512

        return img, box


# 简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 125 * 125, 4)  # Assuming the input image size is 128x128

    def forward(self, x):
        x = self.features(x)
        #print(x.size())  # 打印特征图的大小
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x







# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load('D:\\RAU-Net\\model\\SCNN_rect_model.pth'))
model.eval()

# 初始化验证集
val_dataset = RectBoxDataset('D:\\RAU-Net\\inputs\\CT\\test', 'D:\\RAU-Net\\inputs\\CT\\test_GT_min_rect\\rect_test.txt')   #验证集的图像和对应的矩形框
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# [省略其他代码，直接从验证部分开始]

# 创建保存所有预测结果的单个txt文件
pred_filename = 'D:\\RAU-Net\\rect_pred.txt'   #预测结果的总文件路径和名称
with open(pred_filename, 'w') as f:
    pass  # 创建文件并清空任何既有内容

# 验证
ious = []
for batch_idx, (imgs, true_boxes) in enumerate(tqdm(val_dataloader, desc="Validating")):
    with torch.no_grad():
        pred_boxes = model(imgs)

    # 从数据集获取当前批次的图像文件名
    start_idx = batch_idx * val_dataloader.batch_size
    end_idx = start_idx + val_dataloader.batch_size
    img_files = val_dataset.img_files[start_idx:end_idx]

    # 计算并存储IoU
    for i, (img_file, pred_box, true_box) in enumerate(zip(img_files, pred_boxes, true_boxes)):
        iou = calculate_iou(pred_box.numpy() * 512, true_box.numpy() * 512)
        ious.append(iou)
        
        # 保存预测坐标到单个txt文件
        pred_box_denorm = (pred_box * 512).int().numpy()
        with open(pred_filename, 'a') as f:
            f.write(f'{img_file} 1 {" ".join(map(str, pred_box_denorm))}\n')

# 输出平均IoU
mean_iou = sum(ious) / len(ious)
print(f"Average IoU: {mean_iou}")
