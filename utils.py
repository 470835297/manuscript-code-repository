from PIL import Image 


import argparse

# 定义一个字符串转布尔值的函数方法，输入（字符串）
def str2bool(v):
    # lower（）函数将字符串中所有的大写字母转换为小写字母
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 定义一个计算模型参数函数，输入（模型）
def count_params(model):
    # 获取模型参数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定义一个求均值类
class AverageMeter(object):
    # 计算并存储平均值和当前值
    # 初始化方法
    def __init__(self):
        # 重置
        self.reset()

    # 定义一个重置方法
    def reset(self):
        # 将初始值、平均值、和、计数值均置0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # 更新方法，输入（对象，初始值，n=1）
    def update(self, val, n=1):
        # 获取初始值
        self.val = val
        # 求和
        self.sum += val * n
        # 计数器加n
        self.count += n
        # 求均值
        self.avg = self.sum / self.count


def keep_image_size_open(path,size = (256, 256)):
    img = Image.open(path)#打开文件
    temp = max(img.size)#取得最大边
    mask = Image.new('RGB', (temp,temp),(0,0,0))#画底层最大图像
    mask.paste(img,(0,0))#叠加原图像
    mask = mask.resize(size)#裁剪图像
    return mask

