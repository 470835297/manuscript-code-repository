import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm  
from torch.optim.lr_scheduler import ReduceLROnPlateau




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



# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coordinate_file = 'D:\\RAU-Net\\inputs\\CT\\train_GT_min_rect\\min_rect_coordinates.txt'
img_train_dir='D:\\RAU-Net\\inputs\\CT\\train'
dataset = RectBoxDataset(img_train_dir, coordinate_file)   # 加载图像数据（png）和矩形框数据（txt）
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=True)




# 初始化 IoU 存储列表和最佳 IoU
ious = []
best_iou = 0.0  

# 训练
for epoch in range(200):  # 例如，总共150个 epochs
    epoch_ious = []  # 存储每个 epoch 的 IoU
    all_pred_boxes = []  # 存储所有预测框
    all_true_boxes = []  # 存储所有真实框
    
    for imgs, boxes in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        # 移动数据到设备上
        imgs, boxes = imgs.to(device), boxes.to(device)
        
        # 前向和后向传播
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, boxes)
        loss.backward()
        optimizer.step()
        
        # 将预测框和真实框添加到列表中
        all_pred_boxes.extend(outputs.detach().cpu().numpy())
        all_true_boxes.extend(boxes.cpu().numpy())
        
    # 在整个 epoch 结束后计算平均 IoU
    for pred_box, true_box in zip(all_pred_boxes, all_true_boxes):
        iou = calculate_iou(pred_box * 512, true_box * 512)
        epoch_ious.append(iou)
        
    mean_iou = np.mean(epoch_ious)
    ious.append(mean_iou)
    print(f"Epoch {epoch + 1} Mean IoU: {mean_iou}")
    
    # 保存最佳模型
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), 'D:\\RAU-Net\\model\\SCNN_rect_model.pth')    #保存模型路径和名称
        print(f"New best model saved with IoU: {best_iou}")
    
    # 更新学习率
    scheduler.step(mean_iou)
    
# 输出整体平均 IoU
print(f"Overall Mean IoU: {np.mean(ious)}")
