import os 

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()
    ])



class MyDataset(Dataset):
    def __init__(self,path):#调用即运行
        super().__init__()
        self.path = path#读取数据根目录
        self.name = os.listdir(path)#os.path.join拼接路径，存在/开头是以最后一个为准，前面丢弃。os.listdir读取文件夹所有文件

    def __len__(self):#获取长度
        return len(self.name)#返回self.name长度
    
    def __getitem__(self, index):#类变成运行函数
        segment_name = self.name[index]#读取文件名字
        Image_path = os.path.join(self.path,'rgb',segment_name)#找到源文件
        segment_path = os.path.join(self.path,'segmentsemantic',segment_name.replace('rgb', 'segmentsemantic'))#找到分割文件
        segment_image = keep_image_size_open(segment_path)#统一图像大小
        image = keep_image_size_open(Image_path)#统一图像大小
        return transform(image), transform(segment_image)

if __name__ == '__main__':
    data = MyDataset('data')
    print(data[0][0].shape)

