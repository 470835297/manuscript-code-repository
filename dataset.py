import os
import cv2
import numpy as np
import torch
import torch.utils.data

# 定义数据集类，继承torch.utils.data.Dataset
class Dataset(torch.utils.data.Dataset):
    # 初始化方法，输入（对象，所有图片ids，图片路径，标签路径，图片扩展名，标签扩展名，不进行变换）
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    # 定义一个获取图片总数
    def __len__(self):
        return len(self.img_ids)

    # 定义一个使用索引访问元素的方法，输入（对象，索引）
    def __getitem__(self, idx):
        # 获取图片id,例如：第362张图片的id号是465
        img_id = self.img_ids[idx]

        # if idx == 0:
        #     img_id = self.img_ids[idx+len(self.img_ids)-1]
        #     img_id1 = self.img_ids[idx]
        #     img_id2 = self.img_ids[idx+1]
        
        # elif idx == len(self.img_ids)-1:
        #     img_id = self.img_ids[idx-1]
        #     img_id1 = self.img_ids[idx]
        #     img_id2 = self.img_ids[idx-len(self.img_ids)+1]

        # else :
        #     img_id = self.img_ids[idx-1]
        #     img_id1 = self.img_ids[idx]
        #     img_id2 = self.img_ids[idx+1]
      
        # 获取图片路径 inputs/dsb2018_256/images/465.png   img.shape:(256, 256, 3)
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        # img1 = cv2.imread(os.path.join(self.img_dir, img_id1 + self.img_ext))
        # img2 = cv2.imread(os.path.join(self.img_dir, img_id2 + self.img_ext))
        # 获取标签路径 inputs/dsb2018_256/masks/465.png     mask.shape:(256, 256, 1)
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
        # mask1 = cv2.imread(os.path.join(self.mask_dir, img_id1 + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
        # mask2 = cv2.imread(os.path.join(self.mask_dir, img_id2 + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
        
        # 如果传入的变换不为空，执行
        if self.transform is not None:
            # 对图片和标签进行数据增强
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            # img1 = augmented['image']
            # img2 = augmented['image']
            mask = augmented['mask']
            # mask1 = augmented['mask']
            # mask2 = augmented['mask']
        # 将img归一化，img.shape:(256, 256, 3)
        img = img.astype('float32') / 255.0  
        # img1 = img1.astype('float32') / 255
        # img2 = img2.astype('float32') / 255
        # 将img的（x，y，z）->（z，x，y）img.shape:(3, 256, 256)
        img = img.transpose(2, 0, 1)  
        # img1 = img1.transpose(2, 0, 1)
        # img2 = img2.transpose(2, 0, 1)
        # 将mask归一化，mask.shape:(256, 256, 1)
        mask = mask.astype('float32') / 255.0 
        # mask1 = mask1.astype('float32') / 255
        # mask2 = mask2.astype('float32') / 255
        #  将mask的（x，y，z）->（z，x，y）mask.shape:(1, 256, 256)
        mask = mask.transpose(2, 0, 1)
        # mask1 = mask1.transpose(2, 0, 1)
        # mask2 = mask2.transpose(2, 0, 1)
        

        return img,mask,{'img_id': img_id}
        # return [img,img1,img2],[mask,mask1,mask2], {'img_id': img_id}
