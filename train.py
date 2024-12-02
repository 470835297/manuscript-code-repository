import torch
import os
import argparse
import yaml
import tqdm
import dataset
from medpy import metric
import archs
import losses

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
from metrics import iou_score,dice_coef,acc_coef,sensi_coef,spec_coef
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf



from data import *

MODEL_FILE = archs.__all__
LOSS_NAMES = losses.__all__

torch.cuda.current_device()
torch.cuda._initialized = True

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weight_path = 'model'
# data_path = 'data'
# batch_size = 1
# epoch = 10
# save_img = 'save'

def parse_args():
    #建立解释器
    parser = argparse.ArgumentParser(description='test') #建立解释器

    # 模型名称
    parser.add_argument('-n', '--name', default='CLNHE', type=str, metavar='name',
                        help='model name: (default: arch+timestamp)')
    #：'-'表示简称，'--'表示全称
    #nargs：应该读取的命令行参数个数，可以是具体的数字
    #action：表示若触发则值变成store_下划线后面的值，未触发则为相反值，store_const表示触发才有const值，默认优先级低于default，但是一旦触发，优先级高于default
    #const - action 和 nargs 所需要的常量值。
    #default - 不指定参数时的默认值。
    #type - 命令行参数应该被转换成的类型。
    #choices - 参数可允许的值的一个容器。
    #required - 可选参数是否可以省略 (仅针对可选参数)。
    #help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
    parser.add_argument('--stage', default=1, type=int, 
                        help='model of stage')
    
    parser.add_argument('-epoch','--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-batch', '--batch_size', default=8, type=int, metavar='S',
                        help='mini-batch size (default: 16)')

    # parser.add_argument('--model', default='Mult_SEA_Unet', type=str, metavar='ACHR',
    #                     choices=MODEL_FILE,
    #                     help='model file:'+'|'.join(MODEL_FILE)+'(default: NestedUNet)') #SEA_Unet_7,SEA_DilatedConv_Unet_5,NA_Unet_5
    parser.add_argument('--model', default='SEA_Unet_7', type=str, metavar='ACHR') # MultDeform_SEA_SOT,Mult_SEA_Unet,MultDeform_SEA_Unet,CombinedMultSEAUnet.U_Net_original
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool, metavar='DEEP',
                        help='model of deep supervision')  
    
    parser.add_argument('--input_channels', default=3, type=int, metavar='CHANNEL',
                        help='model of input channels')

    parser.add_argument('--input_w', default=256, type=int,                         
                        help='image of width')

    parser.add_argument('--input_h', default=256, type=int, 
                        help='image of height')

    # parser.add_argument('--loss', default='FocalTverskyLoss', type=str, metavar='LOSS',
    #                     choices=LOSS_NAMES,
    #                     help='model loss:'+'|'.join(LOSS_NAMES)+'(default: BCEDiceLoss)')
    #fun_loss, DiceFocalLoss,BCEDiceLoss,BCEWithLogitsLoss ,CrossEntropyLoss ,CombinedLoss,TverskyFocalLoss，DiceLoss，TverskyLoss，FocalLoss,HBD_Loss
    parser.add_argument('--loss', default='BCEDiceLoss', type=str, metavar='LOSS') 

    parser.add_argument('-data', '--dataset', default='CT', type=str, metavar='DATA',
                        help='data file')

    parser.add_argument('--img_path', default='', type=str, metavar='IMAGE',
                        help='path of image')

    parser.add_argument('--img_ext', default='.png', type=str, metavar='IMAGE',
                        help='number of total epochs to run')

    parser.add_argument('--mask_ext', default='.png', type=str, metavar='IMAGE',
                        help='number of total epochs to run')

    parser.add_argument('--pretrained', default=False, type=str2bool, metavar='PRE',
                        help='model of pretain')

    parser.add_argument('--pretrained_files', default=None, type=str, metavar='PRE',
                        help='model of pretain')

    parser.add_argument('--fine_tuning', default=False, type=str2bool, metavar='FINE',
                        help='fine tuning')

    parser.add_argument('--fine_model', default=False, type=str2bool, metavar='FINE',
                       help='model of fine tuning')


    parser.add_argument('--optimizer', default='Adma', type=str,
                        choices=['Adma', 'SGD'],
                        help='loss: ' +' | '.join(['Adma', 'SGD']) +' (default: Adma)')

    parser.add_argument('--lr', '-learning_rate', default=1e-3, type=float,
                        help='initial learning rate')

    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')

    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')

    parser.add_argument('--nesterov', default=False, type=str2bool,   # default=False,True
                        help='nesterov')
    
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
                        # default='CosineAnnealingLR'

    parser.add_argument('--min_lr', default=1e-10, type=float,
                        help='minimum learning rate')

    parser.add_argument('--factor', default=0.5, type=float,
                        help='?')
    
    parser.add_argument('--patience', default=2, type=int)

    parser.add_argument('--milestones', default=[30,60], type=str)

    parser.add_argument('--gamma', default=2/3, type=float)

    parser.add_argument('--early_stopping', default=-1, type=int,
                        help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--benchmark', default=True, type=bool)
    
    config = parser.parse_args()
    return config

    
def train(config, train_loader, model, criterion, optimizer):
    # dices = []     
    # senss = []
    # specs = []
    # ious= []
    
    avg_meters = {'loss':AverageMeter(),
                  'iou': AverageMeter(),
                  'dice':AverageMeter()              

    }
    model.train()

    pbar = tqdm(total=len(train_loader))

    for input, target, _ in train_loader:
        # print("输入数据的形状: ", input.shape)
        # print("目标数据的形状: ", target.shape)
        input = input.cuda()
        #print(input.shape)  # 应该输出 [batch_size, 1, height, width]

        target = target.cuda()

       
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                
                loss += criterion(output, target)
            iou = iou_score(outputs[-1], target)
            dice = dice_coef(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output,target)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 单输入，求平均loss、iou，并更新参数
        avg_meters['loss'].update(loss.item(), input.size(0))
        #print('avg_meters=',avg_meters)
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        # 输入字典显示实验指标
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),]
                        )

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss':AverageMeter(),
                  'iou': AverageMeter(),
                  'dice':AverageMeter(),
                  'sensi': AverageMeter(),
                  'spec': AverageMeter(),
                  'acc': AverageMeter()             

    }
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))

        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
                sensi = sensi_coef(outputs[-1],target)  
                spec = spec_coef(outputs[-1],target) 
                acc = acc_coef(outputs[-1],target) 
            else:
                output = model(input)
                loss = criterion(output, target)
                
                sensi = sensi_coef(output,target)  
                spec = spec_coef(output,target) 
                acc = acc_coef(output,target) 
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output,target) 


            # 单输入，求平均loss、iou，并更新参数
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['sensi'].update(sensi, input.size(0))
            avg_meters['spec'].update(spec, input.size(0))
            avg_meters['acc'].update(acc, input.size(0))

        
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('sensi', avg_meters['sensi'].avg),
                ('spec', avg_meters['spec'].avg),
                ('acc', avg_meters['acc'].avg)
            ])
            # 输入字典显示实验指标
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('sensi', avg_meters['sensi'].avg),
                        ('spec', avg_meters['spec'].avg),
                        ('acc', avg_meters['acc'].avg),
                        ]
                        )





def main():
    # vars（）函数返回对象object的属性和属性值的字典对象
    config = vars(parse_args())
    config['name'] = config['name']+'_'+config['model']+'_'+config['dataset']
#   当配置文件里的name属性为空时执行
    if config['name'] is None:
        # 计算输出，如果配置里的deep_supervision属性为真（多输出）
        if config['deep_supervision']:
            # 模型名称 403_NestedUNet_wDS
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        # 如果配置里的deep_supervision属性为假（单输出）
        else:
            # 模型名称 403_NestedUNet_woDS
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    #创建目录
    os.makedirs('model/%s' % config['name'], exist_ok=True)

    print('*' * 100)
    for key in config:
        print('%s:%s' %(key, config[key]))
    print('*' * 100)

    with open ('model/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    if config['loss'] == 'BCEWithLogitsLoss':
        print("正在使用BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss'] == 'CrossEntropyLoss':
        print("正在使用CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()
    elif config['loss'] == 'fun_loss':
        print("正在使用fun_loss")
        criterion = losses.__dict__[config['loss']]()
    elif config['loss'] == 'TverskyFocalLoss':
        print("正在使用TverskyFocalLoss")
        criterion = losses.__dict__[config['loss']]()
    elif config['loss'] == 'DiceLoss':
        print("正在使用DiceLoss")
        criterion = losses.__dict__[config['loss']]()
    elif config['loss'] == 'TverskyLoss':
        print("正在使用TverskyLoss")
        criterion = losses.__dict__[config['loss']]()
    elif config['loss'] == 'FocalLoss':
        print("正在使用FocalLoss")
        criterion = losses.__dict__[config['loss']]()
    elif config['loss'] == 'BCEDiceLoss':
        print("正在使用BCEDiceLoss")     
        criterion = losses.__dict__[config['loss']]()
    elif config['loss'] == 'DiceFocalLoss':
        print("正在使用DiceFocalLoss")     
        criterion = losses.__dict__[config['loss']]()
    elif config['loss'] == 'HBD_Loss':
        print("正在使用HBD_Loss")     
        criterion = losses.__dict__[config['loss']]()
    
    else:
        print("loss no found")
        
    #适用场景是网络结构固定，网络的输入形状（包括 batch size，图片大小，输入的通道）不变的
    cudnn.benchmark = config['benchmark']

    print(" ====>>>> creating model %s" % config['model'])
    
    #model = archs.__dict__[config['model']](input_channels=config['input_channels'], num_classes=1)
    model = archs.__dict__[config['model']](num_classes=1)
   
    model = model.cuda()
    
    if config['pretrained']:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config['pretrained_files'])
        keys = []
        for k,v in pretrained_dict.item():
            keys.append(k)
        i = 0
        for k,v in model_dict.item():
            if v.size() == pretrained_dict():
                model_dict[k] = pretrained_dict[keys[i]]
                i += 1
        model.load_state_dict(model_dict)
    params = filter(lambda p: p.requires_grad, model.parameters()) 
    #params = model.parameters()
    #model.load_state_dict(torch.load('model.pth'))
    

    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
    else:
        print("No pre-trained model found. Starting from scratch.")


    if config['optimizer'] == 'Adma':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])

    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                                nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], 
                                                    eta_min=config['min_lr'])

    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                    verbose=1, min_lr=config['min_lr'])

    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], 
                                            gamma=config['gamma'])
    
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    
    else:
        raise NotImplementedError

    img_ids= []
   
        
    #train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.1, shuffle=False)  # 181 test_size=0.0625, 401 test_size=0.1
    train_img_ids = open('inputs/'+config['dataset']+'/train_256_adaptive_CLAHE.txt').read().strip().split()  #训练图像的路径
    val_img_ids = open('inputs/'+config['dataset']+'/test_256_adaptive_CLAHE.txt').read().strip().split()   #验证图像的路径


    

    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        OneOf([
            albu.HueSaturationValue(),
            albu.RandomBrightnessContrast(),
        ], p=1),
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])
    

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    train_dataset = dataset.Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'train_256_adaptive_CLAHE'),         #训练图像路径
        mask_dir=os.path.join('inputs', config['dataset'], 'train_GT_256_adaptive'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        transform=train_transform)    #图像的路径

    val_dataset = dataset.Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'test_256_adaptive_CLAHE'),
        mask_dir=os.path.join('inputs', config['dataset'], 'test_GT_256_adaptive'),          #验证图像路径
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        transform=val_transform)
        

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True)
    

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        # shuffle=True,
        drop_last=False)


    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    
    best_dice = 0
    best_iou = 0
    trigger = 0
    
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        
        train_log = train(config, train_loader, model, criterion, optimizer)
       
        val_log = validate(config, val_loader, model, criterion)

        
        if config['scheduler'] == 'CosineAnnealingLR':

            scheduler.step()

        elif config['scheduler'] == 'ReduceLROnPlateau':

            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f -dice=%.4f - val_loss=%.4f - val_iou=%.4f -val_dice=%.4f  sensi=%.4f - spec=%.4f -acc=%.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['sensi'], val_log['spec'], val_log['acc']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])


        pd.DataFrame(log).to_csv('model/%s/log.csv' %
                                 config['name'], index=False)
        
        trigger += 1
        # 保存最好的模型
        if val_log['iou'] > best_iou:
        #     torch.save(model.state_dict(), 'models/%s/model.pth' %
        #                config['name'])
             best_iou = val_log['iou']
        #     print("=> saved best model")
        #     trigger = 0

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'model/%s/model.pth' % config['name'])   
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0
        
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        
        if epoch % 10 == 0:
            print('dice=%.4f' % best_dice)
            print('iou=%.4f' % best_iou)
        # 释放显存
        torch.cuda.empty_cache() 
    print('dice=%.4f' % best_dice)
    print('iou=%.4f' % best_iou)







if __name__ == '__main__':
    main()





        
