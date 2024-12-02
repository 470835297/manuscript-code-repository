import torch
from torch import nn
import os
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from thop import profile
import torch.nn.functional as F


__all__ = ["UnetPlusPlus",'SOT_Unet_6','SE_Unet_5','Mult_SEA_Unet_6','SEA_Unet_3','SEA_Unet_4','NA_Unet_5','SEA_Unet_5','SEA_Unet_6','SEA_Unet_7','SEA_Unet','Dilated_SEA_Unet',
           'Mult_SEA_Unet','Mult_SEA_SOT','MultDeform_SEA_SOT','NL_Unet_5','SEA_DilatedConv_Unet_5','MultDeform_SEA_Unet','CombinedMultSEAUnet','Mult_SEA_SOT123','U_Net_original']
up_set = 'conv'#bilinear:双线性插值。conv:反卷积
Dropout = 0.5



class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        """
        ConvMixer块初始化函数。
        
        参数:
        dim -- 卷积层的维数。
        kernel_size -- 深度可分离卷积的核大小。
        """
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        
        # GELU激活函数。
        self.gelu = nn.GELU()
        
        # 批量归一化层。
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        x -- 输入特征图。
        
        返回:
        输出特征图。
        """
        # 深度可分离卷积。
        x = self.depthwise(x)
        x = self.gelu(x)
        x = self.bn(x)
        
        # 逐点卷积。
        x = self.pointwise(x)
        x = self.gelu(x)
        x = self.bn(x)
        
        return x

# # 假设我们的输入特征图维数是512，并且我们想要使用3x3的核。
# conv_mixer_block = ConvMixerBlock(dim=512, kernel_size=3)

# # 假设input是一个批量大小为32、通道数为512、宽度和高度为7的特征图。
# input = torch.randn(32, 512, 7, 7)

# # 通过ConvMixer块得到输出。
# output = conv_mixer_block(input)
# print(output.shape)  # 应该和输入的形状相同





class _NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)

        #print(f.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z





class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Query, Key, Value 的线性变换
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # B, C, W, H -> B, C, N
        B, C, W, H = x.size()
        N = W * H
        query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # B, N, C
        key = self.key_conv(x).view(B, -1, N)  # B, C, N
        value = self.value_conv(x).view(B, -1, N)  # B, C, N

        # Self-Attention
        attention = torch.bmm(query, key)  # B, N, N
        attention = self.softmax(attention)
        output = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, N
        output = output.view(B, C, W, H)  # B, C, W, H

        return output

class Conv_A_1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, Dropout=0.2):
        super(Conv_A_1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=1, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
        self.self_attention = SelfAttention(out_channel)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        return x
class Conv_A_2(nn.Module):
    def __init__(self,in_channel, out_channel,kernel_size=2,Dropout=Dropout):
        super(Conv_A_2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
        self.self_attention = SelfAttention(out_channel)
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        return x
    


class Conv_A_3(nn.Module):
    def __init__(self,in_channel, out_channel,kernel_size=6,Dropout=Dropout):
        super(Conv_A_3, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=2, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
        self.self_attention = SelfAttention(out_channel)
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.self_attention(x)
        return x


class Conv_block_1(nn.Module):
    def __init__(self,in_channel, out_channel,kernel_size=4,Dropout=Dropout):
        super(Conv_block_1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=1, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)

class Conv_block_2(nn.Module):
    def __init__(self,in_channel, out_channel,kernel_size=2,Dropout=Dropout):
        super(Conv_block_2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)
    


class Conv_block_3(nn.Module):
    def __init__(self,in_channel, out_channel,kernel_size=6,Dropout=Dropout):
        super(Conv_block_3, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=2, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)
class Att_mul(nn.Module):
    def __init__(self,channel, up_channel, Dropout=Dropout):
        super(Att_mul, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(up_channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
    def forward(self, x,x_1,e,e_1,a,a_1):
        x1 = self.conv1(x)
        x2 = self.conv2(x_1)

        e1 = self.conv1(e)
        e2 = self.conv2(e_1)

        a1 = self.conv1(a)
        a2 = self.conv2(a_1)

        add1 = self.conv3(x1+x2)
        add2 = self.conv3(e1+e2)
        add3 = self.conv3(a1+a2)
        
        return (x*add1+e*add2+a*add3)
    
class Conv_block(nn.Module):
    def __init__(self,in_channel, out_channel, Dropout=Dropout):
        super(Conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=8, stride=2, padding=3, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)



#空洞卷积


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, dropout=0.5):
        super(DilatedConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层用于下采样
        )

    def forward(self, x):
        return self.layer(x)



# 定义一个简单的自注意力模块
class SelfAttention_Deconv(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention_Deconv, self).__init__()
        self.query = nn.Conv2d(in_channel, in_channel//8, 1)
        self.key = nn.Conv2d(in_channel, in_channel//8, 1)
        self.value = nn.Conv2d(in_channel, in_channel, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        q = self.query(x).view(B, -1, W*H).permute(0, 2, 1)
        k = self.key(x).view(B, -1, W*H)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value(x).view(B, -1, W*H)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, W, H)
        return out + x  # 残差连接

# 修改后的 Deconv 类
class Deconv_A(nn.Module):
    def __init__(self, in_channel, out_channel, up_set='conv'):
        super(Deconv_A, self).__init__()
        self.attention = SelfAttention_Deconv(in_channel)  # 添加注意力模块
        if up_set == 'conv':
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
            )
        elif up_set == 'bilinear':
            self.layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.attention(x)  # 注意力模块
        return self.layer(x)




class Deconv(nn.Module):
    def __init__(self, in_channel, out_channel,up_set='conv'):
        super(Deconv, self).__init__()
        if up_set == 'conv':
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                #nn.ReLU()
            )
        elif up_set == 'bilinear':
            self.layer = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layer(x)

class Att(nn.Module):
    def __init__(self,channel, up_channel, Dropout=Dropout):
        super(Att, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(up_channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
    def forward(self, x,g):
        x1 = self.conv1(x)
        g1 = self.conv2(g)
        add = self.conv3(x1+g1)
        return x*add
        #消融 去掉Att
        #return x
    
class SE_Res(nn.Module):
    def __init__(self, channel, ratio):
        super(SE_Res, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // ratio, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.conv(x)
        # (batch,channel,height,width) (2,512,8,8)
        b, c, _, _ = y.size()
        # 全局平均池化 (2,512,8,8) -> (2,512,1,1) -> (2,512)
        a = self.avg_pool(x).view(b, c)
        # (2,512) -> (2,512//reducation) -> (2,512) -> (2,512,1,1)
        a = self.fc(a).view(b, c, 1, 1)
        # (2,512,8,8)* (2,512,1,1) -> (2,512,8,8)
        pro = a * y
        #return self.relu(x + pro)
        #消融
        return x

class SEA_Unet_7(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SEA_Unet_7, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        self.cov7 = Conv_block(512, 512)
        self.cov8 = Conv_block(512, 512)

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 512)
        self.dev4 = Deconv(1024, 256)
        self.dev5 = Deconv(512, 128)
        self.dev6 = Deconv(256, 64)
        
        self.att1 = Att(512, 512)
        self.att2 = Att(512, 512)
        self.att3 = Att(512, 512)
        self.att4 = Att(256, 256)
        self.att5 = Att(128, 128)
        self.att6 = Att(64, 64)
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            r5 = self.cov5(r4)#1/32
            r6 = self.cov6(r5)#1/64
            r7 = self.cov7(r6)#1/128
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r7), r6), self.se1(r6)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r5), self.se2(r5)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r4), self.se3(r4)], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11), r3), self.se4(r3)], dim=1)
            r13 = torch.cat([self.att5(self.dev5(r12), r2), self.se5(r2)], dim=1)  
            r14 = torch.cat([self.att6(self.dev6(r13), r1), self.se6(r1)], dim=1)
            

            #out = self.Th(self.out(r15))
            out = self.out(r14)

            return self.Th(out)
            #return out

class SEA_Unet_6(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SEA_Unet_6, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(256, 8)
        self.se4 = SE_Res(128, 8)
        self.se5 = SE_Res(64, 8)
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 256)
        self.dev4 = Deconv(512, 128)
        self.dev5 = Deconv(256, 64)
        
        
        self.att1 = Att(512, 512)
        self.att2 = Att(512, 512)
        self.att3 = Att(256, 256)
        self.att4 = Att(128, 128)
        self.att5 = Att(64, 64)
       
        
       
        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            r5 = self.cov5(r4)#1/32
            r6 = self.cov6(r5)#1/64
           
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r6), r5), self.se1(r5)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r4), self.se2(r4)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r3), self.se3(r3)], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11), r2), self.se4(r2)], dim=1)
            r13 = torch.cat([self.att5(self.dev5(r12), r1), self.se5(r1)], dim=1)  
            
            

            
            out = self.out(r13)

            return out


class SOT_Unet_6(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SOT_Unet_6, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(256, 8)
        self.se4 = SE_Res(128, 8)
        self.se5 = SE_Res(64, 8)
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 256)
        self.dev4 = Deconv(512, 128)
        self.dev5 = Deconv(256, 64)
        
        
        self.att1 = Att(512, 512)
        self.att2 = Att(512, 512)
        self.att3 = Att(256, 256)
        self.att4 = Att(128, 128)
        self.att5 = Att(64, 64)
       
        
        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        
        self.Th = nn.Sigmoid()
        # 添加一个小目标的二分类头
        self.small_object_head = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)


    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            r5 = self.cov5(r4)#1/32
            r6 = self.cov6(r5)#1/64
           
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r6), r5), self.se1(r5)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r4), self.se2(r4)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r3), self.se3(r3)], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11), r2), self.se4(r2)], dim=1)
            r13 = torch.cat([self.att5(self.dev5(r12), r1), self.se5(r1)], dim=1)  
            
            

            small_object_output = self.small_object_head(r1)
            out = self.out(r13)

            return out, small_object_output






class SEA_Unet_3(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SEA_Unet_3, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        

        self.se1 = SE_Res(128, 8)
        self.se2 = SE_Res(64, 8)
        
        
        
        
        

        self.dev1 = Deconv(256, 128)
        self.dev2 = Deconv(256, 64)
        
        
        
        
        
        self.att1 = Att(128, 128)
        self.att2 = Att(64, 64)
        
        
        
       
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            
            
            
           
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r3), r2), self.se1(r2)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r1), self.se2(r1)], dim=1)
           
            
            
            
            

            
            out = self.out(r10)

            return out







class SEA_Unet_4(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SEA_Unet_4, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        

        self.se1 = SE_Res(256, 8)
        self.se2 = SE_Res(128, 8)
        self.se3 = SE_Res(64, 8)
        
        
        
        

        self.dev1 = Deconv(512, 256)
        self.dev2 = Deconv(512, 128)
        self.dev3 = Deconv(256, 64)
        
        
        
        
        self.att1 = Att(256, 256)
        self.att2 = Att(128, 128)
        self.att3 = Att(64, 64)
        
        
       
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            
            
           
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r4), r3), self.se1(r3)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r2), self.se2(r2)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r1), self.se3(r1)], dim=1)
            
            
            
            

            
            out = self.out(r11)

            return out


class SEA_DilatedConv_Unet_5(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SEA_DilatedConv_Unet_5, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = DilatedConvBlock(input_channels, 64)
        self.cov2 = DilatedConvBlock(64,128)
        self.cov3 = DilatedConvBlock(128, 256)
        self.cov4 = DilatedConvBlock(256, 512)
        self.cov5 = DilatedConvBlock(512, 512)
        self.cov6 = DilatedConvBlock(512, 512)
        

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(256, 8)
        self.se3 = SE_Res(128, 8)
        self.se4 = SE_Res(64, 8)
        
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 256)
        self.dev3 = Deconv(512, 128)
        self.dev4 = Deconv(256, 64)
        
        
        
        self.att1 = Att(512, 512)
        self.att2 = Att(256, 256)
        self.att3 = Att(128, 128)
        self.att4 = Att(64, 64)
        
       
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            r5 = self.cov5(r4)#1/32
            
           
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r5), r4), self.se1(r4)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r3), self.se2(r3)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r2), self.se3(r2)], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11), r1), self.se4(r1)], dim=1)
            
            
            

            
            out = self.out(r12)

            return out


#非局部注意力替换att
class NL_Unet_5(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(NL_Unet_5, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        self.cov7 = ConvMixerBlock(dim=512, kernel_size=3)

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(256, 8)
        self.se3 = SE_Res(128, 8)
        self.se4 = SE_Res(64, 8)
        
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 256)
        self.dev3 = Deconv(512, 128)
        self.dev4 = Deconv(256, 64)
        
        
        # 对于2D图像，直接使用_NonLocalBlockND类


        self.att1 =_NonLocalBlockND(in_channels=512,dimension=2, sub_sample=True, bn_layer=True)
        self.att2 =_NonLocalBlockND(in_channels=256,dimension=2, sub_sample=True, bn_layer=True)
        self.att3 =_NonLocalBlockND(in_channels=128,dimension=2, sub_sample=True, bn_layer=True)
        self.att4 =_NonLocalBlockND(in_channels=64,dimension=2, sub_sample=True, bn_layer=True)
        
       
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            r5 = self.cov5(r4)#1/32
            r6 = self.cov7(r5)#1/32
           
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r5)), r4], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9)), r3], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10)), r2], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11)), r1], dim=1)
            
            
            

            
            out = self.out(r12)

            return out

class SEA_Unet_5(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SEA_Unet_5, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(256, 8)
        self.se3 = SE_Res(128, 8)
        self.se4 = SE_Res(64, 8)
        
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 256)
        self.dev3 = Deconv(512, 128)
        self.dev4 = Deconv(256, 64)
        
        
        
        self.att1 = Att(512, 512)
        self.att2 = Att(256, 256)
        self.att3 = Att(128, 128)
        self.att4 = Att(64, 64)
        
       
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1.shape)
            r2 = self.cov2(r1)#1/4
            #print("r2=",r2.shape)
            r3 = self.cov3(r2)#1/8
            #print("r3=",r3.shape)
            r4 = self.cov4(r3)#1/16
            #print("r4=",r4.shape)
            r5 = self.cov5(r4)#1/32
            #print("r5=",r5.shape)
            
           
         

   

            #print("dev1(r5)=",self.dev1(r5).shape)
            r9 = torch.cat([self.att1(self.dev1(r5), r4), self.se1(r4)], dim=1)
            #print("r9=",r9.shape)
            #print("dev2(r9)=",self.dev2(r9).shape)
            r10 = torch.cat([self.att2(self.dev2(r9), r3), self.se2(r3)], dim=1)
            #print("r10=",r10.shape)
            #print("dev3(r10)=",self.dev3(r10).shape)
            r11 = torch.cat([self.att3(self.dev3(r10), r2), self.se3(r2)], dim=1)
            #print("r11=",r11.shape)
            r12 = torch.cat([self.att4(self.dev4(r11), r1), self.se4(r1)], dim=1)
            #print("r12=",r12.shape)
            
            
            

            
            out = self.out(r12)

            return out

class SE_Unet_5(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SE_Unet_5, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(256, 8)
        self.se3 = SE_Res(128, 8)
        self.se4 = SE_Res(64, 8)
        
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 256)
        self.dev3 = Deconv(512, 128)
        self.dev4 = Deconv(256, 64)
        
        
        
        
        
       
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1.shape)
            r2 = self.cov2(r1)#1/4
            #print("r2=",r2.shape)
            r3 = self.cov3(r2)#1/8
            #print("r3=",r3.shape)
            r4 = self.cov4(r3)#1/16
            #print("r4=",r4.shape)
            r5 = self.cov5(r4)#1/32
            #print("r5=",r5.shape)
            
           
         

   

            #print("dev1(r5)=",self.dev1(r5).shape)
            r9 = torch.cat([self.se1(self.dev1(r5)), r4], dim=1)
            #print("r9=",r9.shape)
            #print("dev2(r9)=",self.dev2(r9).shape)
            r10 = torch.cat([self.se2(self.dev2(r9)), r3], dim=1)
            #print("r10=",r10.shape)
            #print("dev3(r10)=",self.dev3(r10).shape)
            r11 = torch.cat([self.se3(self.dev3(r10)), r2], dim=1)
            #print("r11=",r11.shape)
            r12 = torch.cat([self.se4(self.dev4(r11)), r1], dim=1)
            #print("r12=",r12.shape)
            
            
            

            
            out = self.out(r12)

            return out


#非局部注意力+att
class NA_Unet_5(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(NA_Unet_5, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        self.cov7 = ConvMixerBlock(dim=512, kernel_size=3)

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(256, 8)
        self.se3 = SE_Res(128, 8)
        self.se4 = SE_Res(64, 8)
        
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(256, 256)
        self.dev3 = Deconv(128, 128)
        self.dev4 = Deconv(64, 64)
        
        self.att1 = Att(512, 512)
        self.att2 = Att(256, 256)
        self.att3 = Att(128, 128)
        self.att4 = Att(64, 64)
        # 对于2D图像，直接使用_NonLocalBlockND类


        self.NL1 =_NonLocalBlockND(in_channels=512,dimension=2, sub_sample=True, bn_layer=True)
        self.NL2 =_NonLocalBlockND(in_channels=256,dimension=2, sub_sample=True, bn_layer=True)
        self.NL3 =_NonLocalBlockND(in_channels=128,dimension=2, sub_sample=True, bn_layer=True)
        self.NL4 =_NonLocalBlockND(in_channels=64,dimension=2, sub_sample=True, bn_layer=True)
        
        self.C_1024_256= nn.Conv2d(1024, 256, kernel_size=1)
        self.C_512_128= nn.Conv2d(512, 128, kernel_size=1)
        self.C_256_64= nn.Conv2d(256, 64, kernel_size=1)





        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            #print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1.shape)
            r2 = self.cov2(r1)#1/4
            #print("r2=",r2.shape)
            r3 = self.cov3(r2)#1/8
            #print("r3=",r3.shape)
            r4 = self.cov4(r3)#1/16
            #print("r4=",r4.shape)
            r5 = self.cov5(r4)#1/32
            #print("r5=",r5.shape)
            r6 = self.cov7(r5)#1/32

           

   
            
            #print('dev1(r5)=',self.dev1(r5).shape)
            r9 = torch.cat([self.att1(self.dev1(r5), r4), r4], dim=1)
            r9_1=self.C_1024_256(r9)
            r10 = torch.cat([self.att2(self.dev2(self.NL2(r9_1)), r3),r3], dim=1)
            r10_1=self.C_512_128(r10)
            r11 = torch.cat([self.att3(self.dev3(self.NL3(r10_1)), r2),r2], dim=1)
            r11_1=self.C_256_64(r11)
            r12 = torch.cat([self.att4(self.dev4(self.NL4(r11_1)), r1),r1], dim=1)
            
            
            

            
            out = self.out(r12)

            return out




class SEA_Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(SEA_Unet, self).__init__()

        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, filter[0])
        self.cov2 = Conv_block(filter[0], filter[1])
        self.cov3 = Conv_block(filter[1], filter[2])
        self.cov4 = Conv_block(filter[2], filter[3])
        self.cov5 = Conv_block(filter[3], filter[4])
        self.cov6 = Conv_block(filter[4], filter[5])
        self.cov7 = Conv_block(filter[5], filter[6])
        self.cov8 = Conv_block(filter[6], filter[7])

        self.se1 = SE_Res(filter[6], 8)
        self.se2 = SE_Res(filter[5], 8)
        self.se3 = SE_Res(filter[4], 8)
        self.se4 = SE_Res(filter[3], 8)
        self.se5 = SE_Res(filter[2], 8)
        self.se6 = SE_Res(filter[1], 8)
        self.se7 = SE_Res(filter[0], 8)

        self.dev1 = Deconv(Deconvs[0], filter[6])
        self.dev2 = Deconv(Deconvs[1], filter[5])
        self.dev3 = Deconv(Deconvs[2], filter[4])
        self.dev4 = Deconv(Deconvs[3], filter[3])
        self.dev5 = Deconv(Deconvs[4], filter[2])
        self.dev6 = Deconv(Deconvs[5], filter[1])
        self.dev7 = Deconv(Deconvs[6], filter[0])

        self.att1 = Att(filter[6], filter[6])
        self.att2 = Att(filter[5], filter[5])
        self.att3 = Att(filter[4], filter[4])
        self.att4 = Att(filter[3], filter[3])
        self.att5 = Att(filter[2], filter[2])
        self.att6 = Att(filter[1], filter[1])
        self.att7 = Att(filter[0], filter[0])

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            r5 = self.cov5(r4)#1/32
            r6 = self.cov6(r5)#1/64
            r7 = self.cov7(r6)#1/128
            r8 = self.cov8(r7)#1/256



            
            r9 = torch.cat([self.att1(self.dev1(r8), r7), self.se1(r7)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r6), self.se2(r6)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r5), self.se3(r5)], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11), r4), self.se4(r4)], dim=1)
            r13 = torch.cat([self.att5(self.dev5(r12), r3), self.se5(r3)], dim=1)  
            r14 = torch.cat([self.att6(self.dev6(r13), r2), self.se6(r2)], dim=1)
            r15 = torch.cat([self.att7(self.dev7(r14), r1), self.se7(r1)], dim=1)

            #out = self.Th(self.out(r15))
            out = self.out(r15)

            return out


class VGGBlock(nn.Module):
    # 初始化方法，输入（对象，输入通道数，中间通道数，输出通道数）
    def __init__(self, in_channels, middle_channels, out_channels):
        # 继承父类
        super().__init__()
        # 使用ReLU激活函数，进行覆盖运算
        self.relu = nn.ReLU(inplace=False)
        # 卷积层1，（输入通道数=in_channels，卷积产生的通道数=middle_channels，卷积核3*3，只对1层补0）
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # 批标准化层1，（输入形状=middle_channels）
        self.bn1 = nn.BatchNorm2d(middle_channels)
        # 卷积层2，（输入通道数=middle_channels，卷积产生的通道数=out_channels，卷积核3*3，只对1层补0）
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        # 批标准化层2，（输入形状=out_channels）
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(p=0.2)

    # 定义前向传播，输入（x）
    def forward(self, x):
        # 输出为输入x经过卷积层1
        out = self.conv1(x)
        # 输出为输入out经过批标准化层1
        out = self.bn1(out)
        # 输出为输入out经过ReLU激活函数
        out = self.relu(out)

        # 输出为输入out经过卷积层2
        out = self.conv2(out)
        # 输出为输入out经过批标准化层2
        out = self.bn2(out)
        # 输出为输入out经过ReLU激活函数
        out = self.relu(out)

        # out = self.dropout(out)

        return out

# SE模块
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        # 全局自适应池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)		
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # view()的作用相当于numpy中的reshape，重新定义矩阵的形状。
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class conv_b(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_b,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_c(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_c,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x




class U_Net_original(nn.Module):
    def __init__(self,img_ch=3,num_classes=1):
        super(U_Net_original,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_b(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_b(ch_in=64,ch_out=128)
        self.Conv3 = conv_b(ch_in=128,ch_out=256)
        self.Conv4 = conv_b(ch_in=256,ch_out=512)
        self.Conv5 = conv_b(ch_in=512,ch_out=1024)

        self.Up5 = up_c(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_b(ch_in=1024, ch_out=512)

        self.Up4 = up_c(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_b(ch_in=512, ch_out=256)
        
        self.Up3 = up_c(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_b(ch_in=256, ch_out=128)
        
        self.Up2 = up_c(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_b(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# 定义self-attention模块类
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self,g,x):
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi    # 形状与下采样的特征图一致


        



class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm = True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        if pre_Batch_Norm:
          self.Conv_forward = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
 
        else:
          self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x
 
class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]
        
        self.CONV3_1 = ContinusParalleConv(512*2, 512, pre_Batch_Norm = True)
 
        self.CONV2_2 = ContinusParalleConv(256*3, 256, pre_Batch_Norm = True)
        self.CONV2_1 = ContinusParalleConv(256*2, 256, pre_Batch_Norm = True)
 
        self.CONV1_1 = ContinusParalleConv(128*2, 128, pre_Batch_Norm = True)
        self.CONV1_2 = ContinusParalleConv(128*3, 128, pre_Batch_Norm = True)
        self.CONV1_3 = ContinusParalleConv(128*4, 128, pre_Batch_Norm = True)
 
        self.CONV0_1 = ContinusParalleConv(64*2, 64, pre_Batch_Norm = True)
        self.CONV0_2 = ContinusParalleConv(64*3, 64, pre_Batch_Norm = True)
        self.CONV0_3 = ContinusParalleConv(64*4, 64, pre_Batch_Norm = True)
        self.CONV0_4 = ContinusParalleConv(64*5, 64, pre_Batch_Norm = True)
 
 
        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm = False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm = False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm = False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm = False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm = False)
 
        self.pool = nn.MaxPool2d(2)
    
        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
 
        
        # 分割头
        self.final_super_0_1 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_2 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_3 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_4 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
 
        
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))
        
        x_0_1 = torch.cat([self.upsample_0_1(x_1_0) , x_0_0], 1)
        x_0_1 =  self.CONV0_1(x_0_1)
        
        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)
        
        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)
        
        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)
 
        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)
        
        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)
        
        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)
 
        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)
        
        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)
        
        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)


# 其他类（SE_Res, Att, Deconv）可以保持不变



k_size=6
pd=2
# 使用可变形卷积的卷积块
class DeformConv_block(nn.Module):
    def __init__(self, in_channel, out_channel, Dropout):
        super(DeformConv_block, self).__init__()
        # 预测offset的卷积层
        self.offset_conv = nn.Conv2d(in_channel, 2 * k_size *k_size, kernel_size=k_size, stride=2, padding=pd)
        self.deform_conv = DeformConv2d(in_channel, out_channel, kernel_size=k_size, stride=2, padding=pd)
        self.bn = nn.BatchNorm2d(out_channel)
        self.dropout = nn.Dropout(Dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算offset
        offset = self.offset_conv(x)
        
        # 应用可变形卷积
        x = self.deform_conv(x, offset)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        return x




# 使用Deformable Convolution的U-Net模型
class Deform_SEA_Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, **kwargs):
        super(Deform_SEA_Unet, self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]

        self.cov1 = DeformConv_block(input_channels, filter[0], 0.1)
        self.cov2 = DeformConv_block(filter[0], filter[1], 0.1)
        self.cov3 = DeformConv_block(filter[1], filter[2], 0.1)
        self.cov4 = DeformConv_block(filter[2], filter[3], 0.1)
        self.cov5 = DeformConv_block(filter[3], filter[4], 0.1)
        self.cov6 = DeformConv_block(filter[4], filter[5], 0.1)
        self.cov7 = DeformConv_block(filter[5], filter[6], 0.1)
        self.cov8 = DeformConv_block(filter[6], filter[7], 0.1)

        self.se1 = SE_Res(filter[6], 8)
        self.se2 = SE_Res(filter[5], 8)
        self.se3 = SE_Res(filter[4], 8)
        self.se4 = SE_Res(filter[3], 8)
        self.se5 = SE_Res(filter[2], 8)
        self.se6 = SE_Res(filter[1], 8)
        self.se7 = SE_Res(filter[0], 8)
        #          0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(Deconvs[0], filter[6])
        self.dev2 = Deconv(Deconvs[1], filter[5])
        self.dev3 = Deconv(Deconvs[2], filter[4])
        self.dev4 = Deconv(Deconvs[3], filter[3])
        self.dev5 = Deconv(Deconvs[4], filter[2])
        self.dev6 = Deconv(Deconvs[5], filter[1])
        self.dev7 = Deconv(Deconvs[6], filter[0])

        self.att1 = Att(filter[6], filter[6])
        self.att2 = Att(filter[5], filter[5])
        self.att3 = Att(filter[4], filter[4])
        self.att4 = Att(filter[3], filter[3])
        self.att5 = Att(filter[2], filter[2])
        self.att6 = Att(filter[1], filter[1])
        self.att7 = Att(filter[0], filter[0])

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        r1 = self.cov1(x)
        #print('r1=',r1.shape)
        r2 = self.cov2(r1)
        #print('r2=',r2.shape)
        r3 = self.cov3(r2)
        #print('r3=',r3.shape)
        r4 = self.cov4(r3)
        #print('r4=',r4.shape)
        r5 = self.cov5(r4)
        #print('r5=',r5.shape)
        r6 = self.cov6(r5)
        #print('r6=',r6.shape)
        r7 = self.cov7(r6)
        #print('r7=',r7.shape)
        r8 = self.cov8(r7)
        #print('r8=',r8.shape)

        r9 = torch.cat([self.att1(self.dev1(r8), r7), self.se1(r7)], dim=1)
        r10 = torch.cat([self.att2(self.dev2(r9), r6), self.se2(r6)], dim=1)
        r11 = torch.cat([self.att3(self.dev3(r10), r5), self.se3(r5)], dim=1)
        r12 = torch.cat([self.att4(self.dev4(r11), r4), self.se4(r4)], dim=1)
        r13 = torch.cat([self.att5(self.dev5(r12), r3), self.se5(r3)], dim=1)
        r14 = torch.cat([self.att6(self.dev6(r13), r2), self.se6(r2)], dim=1)
        r15 = torch.cat([self.att7(self.dev7(r14), r1), self.se7(r1)], dim=1)
        out = self.out(r15)

        return out
    



class Mult_SEA_Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(Mult_SEA_Unet,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_block_2(input_channels, filter[0])#128*128
        self.cov2_2 = Conv_block_2(filter[0], filter[1])
        self.cov3_2 = Conv_block_2(filter[1], filter[2])
        self.cov4_2 = Conv_block_2(filter[2], filter[3])
        self.cov5_2 = Conv_block_2(filter[3], filter[4])
        self.cov6_2 = Conv_block_2(filter[4], filter[5])
        self.cov7_2 = Conv_block_2(filter[5], filter[6])
        self.cov8_2 = Conv_block_2(filter[6], filter[7])


        self.cov1_4 = Conv_block_1(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_block_1(filter[0], filter[1])
        self.cov3_4 = Conv_block_1(filter[1], filter[2])
        self.cov4_4 = Conv_block_1(filter[2], filter[3])
        self.cov5_4 = Conv_block_1(filter[3], filter[4])
        self.cov6_4 = Conv_block_1(filter[4], filter[5])
        self.cov7_4 = Conv_block_1(filter[5], filter[6])
        self.cov8_4 = Conv_block_1(filter[6], filter[7])


        self.cov1_6 = Conv_block_3(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_block_3(filter[0], filter[1])
        self.cov3_6 = Conv_block_3(filter[1], filter[2])
        self.cov4_6 = Conv_block_3(filter[2], filter[3])
        self.cov5_6 = Conv_block_3(filter[3], filter[4])
        self.cov6_6 = Conv_block_3(filter[4], filter[5])
        self.cov7_6 = Conv_block_3(filter[5], filter[6])
        self.cov8_6 = Conv_block_3(filter[6], filter[7])





        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 512)
        self.dev4 = Deconv(1024, 256)
        self.dev5 = Deconv(512, 128)
        self.dev6 = Deconv(256, 64)
        

        self.att1 = Att_mul(512, 512)
        self.att2 = Att_mul(512, 512)
        self.att3 = Att_mul(512, 512)
        self.att4 = Att_mul(256, 256)
        self.att5 = Att_mul(128, 128)
        self.att6 = Att_mul(64, 64)
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()
        self.R = nn.ReLU()
    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            r1_7 = self.cov7_2(r1_6)#1/128
            #print(r1_7.shape)
            
            
            d1_0=self.dev1(r1_7)



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            r2_7 = self.cov7_4(r2_6)#1/128
            #print(r2_7.shape)
            

            d2_0=self.dev1(r2_7)


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
            r3_7 = self.cov7_6(r3_6)#1/128
            #print(r3_7.shape)
            
            d3_0=self.dev1(r3_7)
           
            
            
            att1=self.att1(x=d1_0, x_1=r1_6, e=d2_0, e_1=r2_6, a=d3_0, a_1=r3_6)     
            se1=self.se1(r1_6)+self.se1(r2_6)+self.se1(r3_6)
            d1=torch.cat((att1,se1),dim=1)
            # print("d1.shape=",d1.shape)
            # print("r1_6=",r1_6.shape)

            att2=self.att2(x=self.dev2(d1),x_1=r1_5,e=self.dev2(d1),e_1=r2_5,a=self.dev2(d1),a_1=r3_5)     
            se2=self.se2(r1_5)+self.se2(r2_5)+self.se2(r3_5)
            d2=torch.cat((att2,se2),dim=1)
            # print("d2.shape=",d2.shape)
            # print("r1_5=",r1_5.shape)
            # print("r2_5=",r2_5.shape)
            # print("r3_5=",r3_5.shape)
            # print("self.dev2(d2)=",self.dev2(d2).shape)
            

            att3=self.att3(x=self.dev3(d2),x_1=r1_4,e=self.dev2(d2),e_1=r2_4,a=self.dev2(d2),a_1=r3_4)     
            se3=self.se3(r1_4)+self.se3(r2_4)+self.se3(r3_4)
            d3=torch.cat((att3,se3),dim=1)
            

            att4=self.att4(x=self.dev4(d3),x_1=r1_3,e=self.dev4(d3),e_1=r2_3,a=self.dev4(d3),a_1=r3_3) 
            se4=self.se4(r1_3)+self.se4(r2_3)+self.se4(r3_3)
            d4=torch.cat((att4,se4),dim=1)

            att5=self.att5(x=self.dev5(d4),x_1=r1_2,e=self.dev5(d4),e_1=r2_2,a=self.dev5(d4),a_1=r3_2)     
            se5=self.se5(r1_2)+self.se5(r2_2)+self.se5(r3_2)
            d5=torch.cat((att5,se5),dim=1)

            att6=self.att6(x=self.dev6(d5),x_1=r1_1,e=self.dev6(d5),e_1=r2_1,a=self.dev6(d5),a_1=r3_1)     
            se6=self.se6(r1_1)+self.se6(r2_1)+self.se6(r3_1)
            d6=torch.cat((att6,se6),dim=1)

            

            out = self.out(d6)
            #out = self.R(out)

            return out




class Mult_SEA_Unet_6(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(Mult_SEA_Unet_6,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_block_2(input_channels, 64)#128*128
        self.cov2_2 = Conv_block_2(64, 128)#64*64
        self.cov3_2 = Conv_block_2(128, 256)#32*32
        self.cov4_2 = Conv_block_2(256, 512)#16*16
        self.cov5_2 = Conv_block_2(512, 512)#8*8
        self.cov6_2 = Conv_block_2(512, 512)#4*4
        

        self.cov1_4 = Conv_block_1(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_block_1(filter[0], filter[1])
        self.cov3_4 = Conv_block_1(filter[1], filter[2])
        self.cov4_4 = Conv_block_1(filter[2], filter[3])
        self.cov5_4 = Conv_block_1(filter[3], filter[4])
        self.cov6_4 = Conv_block_1(filter[4], filter[5])
        


        self.cov1_6 = Conv_block_3(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_block_3(filter[0], filter[1])
        self.cov3_6 = Conv_block_3(filter[1], filter[2])
        self.cov4_6 = Conv_block_3(filter[2], filter[3])
        self.cov5_6 = Conv_block_3(filter[3], filter[4])
        self.cov6_6 = Conv_block_3(filter[4], filter[5])
        




        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(256, 8)
        self.se4 = SE_Res(128, 8)
        self.se5 = SE_Res(64, 8)
        
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 256)
        self.dev4 = Deconv(512, 128)
        self.dev5 = Deconv(256, 64)
        
        

        self.att1 = Att(512, 512)
        self.att2 = Att(512, 512)
        self.att3 = Att(256, 256)
        self.att4 = Att(128, 128)
        self.att5 = Att(64, 64)
        
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()
        self.R = nn.ReLU()
    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            
            
            
            



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            
            

            


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
       
            
           
            r123_1=r1_1+r2_1+r3_1
            r123_2=r1_2+r2_2+r3_2
            r123_3=r1_3+r2_3+r3_3
            r123_4=r1_4+r2_4+r3_4
            r123_5=r1_5+r2_5+r3_5
            r123_6=r1_6+r2_6+r3_6
            
            

            r9 = torch.cat([self.att1(self.dev1(r123_6), r123_5), self.se1(r123_5)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r123_4), self.se2(r123_4)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r123_3), self.se3(r123_3)], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11), r123_2), self.se4(r123_2)], dim=1)
            r13 = torch.cat([self.att5(self.dev5(r12), r123_1), self.se5(r123_1)], dim=1)  
            



            
            
            out = self.out(r13)
            #out = self.R(out)

            return out









class CombinedMultSEAUnet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, **kwargs):
        super(CombinedMultSEAUnet, self).__init__()
        
        # 初始化四个独立的Mult_SEA_Unet网络#SEA_Unet
        self.unet1 = U_Net_original(input_channels, num_classes, **kwargs)
        self.unet2 = U_Net_original(input_channels, num_classes, **kwargs)
        self.unet3 = U_Net_original(input_channels, num_classes, **kwargs)
        self.unet4 = U_Net_original(input_channels, num_classes, **kwargs)
    
    def split_image_into_4(self, x):
        """
        将输入图像x(假设尺寸为[B, C, 512, 512])分割成4个小块。
        """
        upper_left = x[:, :, :256, :256]
        upper_right = x[:, :, :256, 256:]
        lower_left = x[:, :, 256:, :256]
        lower_right = x[:, :, 256:, 256:]
        return upper_left, upper_right, lower_left, lower_right
    
    def merge_4_into_image(self, ul, ur, ll, lr):
        """
        将4个小块合并为一个完整的图像。
        在重合部分取最大值。
        """
        # 上半部分和下半部分，取重合区域的最大值
        upper = torch.max(ul[:, :, :, :256], ur[:, :, :, :256])
        lower = torch.max(ll[:, :, :, :256], lr[:, :, :, :256])
        
        # 最终拼接
        upper = torch.cat([ul[:, :, :, :256], upper, ur[:, :, :, 256:]], dim=3)
        lower = torch.cat([ll[:, :, :, :256], lower, lr[:, :, :, 256:]], dim=3)
        return torch.cat([upper, lower], dim=2)
    
    def forward(self, x):
        # 分割图像
        #print("Before split:", x.shape)
        ul, ur, ll, lr = self.split_image_into_4(x)
        #print("After split:", ul.shape, ur.shape, ll.shape, lr.shape)
        # 通过四个独立的Mult_SEA_Unet网络进行处理
        ul_out = self.unet1(ul)
        ur_out = self.unet2(ur)
        ll_out = self.unet3(ll)
        lr_out = self.unet4(lr)
        
        # 将4个小块重新组合为一个完整的图像
        out = self.merge_4_into_image(ul_out, ur_out, ll_out, lr_out)
        
        return out








class Mult_AAS_Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(Mult_AAS_Unet,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_A_2(input_channels, filter[0])#128*128
        self.cov2_2 = Conv_A_2(filter[0], filter[1])
        self.cov3_2 = Conv_A_2(filter[1], filter[2])
        self.cov4_2 = Conv_A_2(filter[2], filter[3])
        self.cov5_2 = Conv_A_2(filter[3], filter[4])
        self.cov6_2 = Conv_A_2(filter[4], filter[5])
        self.cov7_2 = Conv_A_2(filter[5], filter[6])
        self.cov8_2 = Conv_A_2(filter[6], filter[7])


        self.cov1_4 = Conv_A_1(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_A_1(filter[0], filter[1])
        self.cov3_4 = Conv_A_1(filter[1], filter[2])
        self.cov4_4 = Conv_A_1(filter[2], filter[3])
        self.cov5_4 = Conv_A_1(filter[3], filter[4])
        self.cov6_4 = Conv_A_1(filter[4], filter[5])
        self.cov7_4 = Conv_A_1(filter[5], filter[6])
        self.cov8_4 = Conv_A_1(filter[6], filter[7])


        self.cov1_6 = Conv_A_3(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_A_3(filter[0], filter[1])
        self.cov3_6 = Conv_A_3(filter[1], filter[2])
        self.cov4_6 = Conv_A_3(filter[2], filter[3])
        self.cov5_6 = Conv_A_3(filter[3], filter[4])
        self.cov6_6 = Conv_A_3(filter[4], filter[5])
        self.cov7_6 = Conv_A_3(filter[5], filter[6])
        self.cov8_6 = Conv_A_3(filter[6], filter[7])





        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 512)
        self.dev4 = Deconv(1024, 256)
        self.dev5 = Deconv(512, 128)
        self.dev6 = Deconv(256, 64)
        

        self.att1 = Att_mul(512, 512)
        self.att2 = Att_mul(512, 512)
        self.att3 = Att_mul(512, 512)
        self.att4 = Att_mul(256, 256)
        self.att5 = Att_mul(128, 128)
        self.att6 = Att_mul(64, 64)
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            r1_7 = self.cov7_2(r1_6)#1/128
            #print(r1_7.shape)
            
            
            d1_0=self.dev1(r1_7)



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            r2_7 = self.cov7_4(r2_6)#1/128
            #print(r2_7.shape)
            

            d2_0=self.dev1(r2_7)


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
            r3_7 = self.cov7_6(r3_6)#1/128
            #print(r3_7.shape)
            
            d3_0=self.dev1(r3_7)
           
            
            
            att1=self.att1(x=d1_0, x_1=r1_6, e=d2_0, e_1=r2_6, a=d3_0, a_1=r3_6)     
            se1=self.se1(r1_6)+self.se1(r2_6)+self.se1(r3_6)
            d1=torch.cat((att1,se1),dim=1)
            # print("d1.shape=",d1.shape)
            # print("r1_6=",r1_6.shape)

            att2=self.att2(x=self.dev2(d1),x_1=r1_5,e=self.dev2(d1),e_1=r2_5,a=self.dev2(d1),a_1=r3_5)     
            se2=self.se2(r1_5)+self.se2(r2_5)+self.se2(r3_5)
            d2=torch.cat((att2,se2),dim=1)
            # print("d2.shape=",d2.shape)
            # print("r1_5=",r1_5.shape)
            # print("r2_5=",r2_5.shape)
            # print("r3_5=",r3_5.shape)
            # print("self.dev2(d2)=",self.dev2(d2).shape)
            

            att3=self.att3(x=self.dev3(d2),x_1=r1_4,e=self.dev2(d2),e_1=r2_4,a=self.dev2(d2),a_1=r3_4)     
            se3=self.se3(r1_4)+self.se3(r2_4)+self.se3(r3_4)
            d3=torch.cat((att3,se3),dim=1)
            

            att4=self.att4(x=self.dev4(d3),x_1=r1_3,e=self.dev4(d3),e_1=r2_3,a=self.dev4(d3),a_1=r3_3) 
            se4=self.se4(r1_3)+self.se4(r2_3)+self.se4(r3_3)
            d4=torch.cat((att4,se4),dim=1)

            att5=self.att5(x=self.dev5(d4),x_1=r1_2,e=self.dev5(d4),e_1=r2_2,a=self.dev5(d4),a_1=r3_2)     
            se5=self.se5(r1_2)+self.se5(r2_2)+self.se5(r3_2)
            d5=torch.cat((att5,se5),dim=1)

            att6=self.att6(x=self.dev6(d5),x_1=r1_1,e=self.dev6(d5),e_1=r2_1,a=self.dev6(d5),a_1=r3_1)     
            se6=self.se6(r1_1)+self.se6(r2_1)+self.se6(r3_1)
            d6=torch.cat((att6,se6),dim=1)

            

            out = self.out(d6)

            return out

class Mult_SAA_Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(Mult_SAA_Unet,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_block_2(input_channels, filter[0])#128*128
        self.cov2_2 = Conv_block_2(filter[0], filter[1])
        self.cov3_2 = Conv_block_2(filter[1], filter[2])
        self.cov4_2 = Conv_block_2(filter[2], filter[3])
        self.cov5_2 = Conv_block_2(filter[3], filter[4])
        self.cov6_2 = Conv_block_2(filter[4], filter[5])
        self.cov7_2 = Conv_block_2(filter[5], filter[6])
        self.cov8_2 = Conv_block_2(filter[6], filter[7])


        self.cov1_4 = Conv_block_1(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_block_1(filter[0], filter[1])
        self.cov3_4 = Conv_block_1(filter[1], filter[2])
        self.cov4_4 = Conv_block_1(filter[2], filter[3])
        self.cov5_4 = Conv_block_1(filter[3], filter[4])
        self.cov6_4 = Conv_block_1(filter[4], filter[5])
        self.cov7_4 = Conv_block_1(filter[5], filter[6])
        self.cov8_4 = Conv_block_1(filter[6], filter[7])


        self.cov1_6 = Conv_block_3(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_block_3(filter[0], filter[1])
        self.cov3_6 = Conv_block_3(filter[1], filter[2])
        self.cov4_6 = Conv_block_3(filter[2], filter[3])
        self.cov5_6 = Conv_block_3(filter[3], filter[4])
        self.cov6_6 = Conv_block_3(filter[4], filter[5])
        self.cov7_6 = Conv_block_3(filter[5], filter[6])
        self.cov8_6 = Conv_block_3(filter[6], filter[7])





        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv_A(512, 512)
        self.dev2 = Deconv_A(1024, 512)
        self.dev3 = Deconv_A(1024, 512)
        self.dev4 = Deconv_A(1024, 256)
        self.dev5 = Deconv_A(512, 128)
        self.dev6 = Deconv_A(256, 64)
        

        self.att1 = Att_mul(512, 512)
        self.att2 = Att_mul(512, 512)
        self.att3 = Att_mul(512, 512)
        self.att4 = Att_mul(256, 256)
        self.att5 = Att_mul(128, 128)
        self.att6 = Att_mul(64, 64)
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            r1_7 = self.cov7_2(r1_6)#1/128
            #print(r1_7.shape)
            
            
            d1_0=self.dev1(r1_7)



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            r2_7 = self.cov7_4(r2_6)#1/128
            #print(r2_7.shape)
            

            d2_0=self.dev1(r2_7)


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
            r3_7 = self.cov7_6(r3_6)#1/128
            #print(r3_7.shape)
            
            d3_0=self.dev1(r3_7)
           
            
            
            att1=self.att1(x=d1_0, x_1=r1_6, e=d2_0, e_1=r2_6, a=d3_0, a_1=r3_6)     
            se1=self.se1(r1_6)+self.se1(r2_6)+self.se1(r3_6)
            d1=torch.cat((att1,se1),dim=1)
            # print("d1.shape=",d1.shape)
            # print("r1_6=",r1_6.shape)

            att2=self.att2(x=self.dev2(d1),x_1=r1_5,e=self.dev2(d1),e_1=r2_5,a=self.dev2(d1),a_1=r3_5)     
            se2=self.se2(r1_5)+self.se2(r2_5)+self.se2(r3_5)
            d2=torch.cat((att2,se2),dim=1)
            # print("d2.shape=",d2.shape)
            # print("r1_5=",r1_5.shape)
            # print("r2_5=",r2_5.shape)
            # print("r3_5=",r3_5.shape)
            # print("self.dev2(d2)=",self.dev2(d2).shape)
            

            att3=self.att3(x=self.dev3(d2),x_1=r1_4,e=self.dev2(d2),e_1=r2_4,a=self.dev2(d2),a_1=r3_4)     
            se3=self.se3(r1_4)+self.se3(r2_4)+self.se3(r3_4)
            d3=torch.cat((att3,se3),dim=1)
            

            att4=self.att4(x=self.dev4(d3),x_1=r1_3,e=self.dev4(d3),e_1=r2_3,a=self.dev4(d3),a_1=r3_3) 
            se4=self.se4(r1_3)+self.se4(r2_3)+self.se4(r3_3)
            d4=torch.cat((att4,se4),dim=1)

            att5=self.att5(x=self.dev5(d4),x_1=r1_2,e=self.dev5(d4),e_1=r2_2,a=self.dev5(d4),a_1=r3_2)     
            se5=self.se5(r1_2)+self.se5(r2_2)+self.se5(r3_2)
            d5=torch.cat((att5,se5),dim=1)

            att6=self.att6(x=self.dev6(d5),x_1=r1_1,e=self.dev6(d5),e_1=r2_1,a=self.dev6(d5),a_1=r3_1)     
            se6=self.se6(r1_1)+self.se6(r2_1)+self.se6(r3_1)
            d6=torch.cat((att6,se6),dim=1)

            

            out = self.out(d6)

            return out




class Mult_SEA_SOT(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(Mult_SEA_SOT,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_block_2(input_channels, filter[0])#128*128
        self.cov2_2 = Conv_block_2(filter[0], filter[1])
        self.cov3_2 = Conv_block_2(filter[1], filter[2])
        self.cov4_2 = Conv_block_2(filter[2], filter[3])
        self.cov5_2 = Conv_block_2(filter[3], filter[4])
        self.cov6_2 = Conv_block_2(filter[4], filter[5])
        self.cov7_2 = Conv_block_2(filter[5], filter[6])
        self.cov8_2 = Conv_block_2(filter[6], filter[7])


        self.cov1_4 = Conv_block_1(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_block_1(filter[0], filter[1])
        self.cov3_4 = Conv_block_1(filter[1], filter[2])
        self.cov4_4 = Conv_block_1(filter[2], filter[3])
        self.cov5_4 = Conv_block_1(filter[3], filter[4])
        self.cov6_4 = Conv_block_1(filter[4], filter[5])
        self.cov7_4 = Conv_block_1(filter[5], filter[6])
        self.cov8_4 = Conv_block_1(filter[6], filter[7])


        self.cov1_6 = Conv_block_3(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_block_3(filter[0], filter[1])
        self.cov3_6 = Conv_block_3(filter[1], filter[2])
        self.cov4_6 = Conv_block_3(filter[2], filter[3])
        self.cov5_6 = Conv_block_3(filter[3], filter[4])
        self.cov6_6 = Conv_block_3(filter[4], filter[5])
        self.cov7_6 = Conv_block_3(filter[5], filter[6])
        self.cov8_6 = Conv_block_3(filter[6], filter[7])





        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 512)
        self.dev4 = Deconv(1024, 256)
        self.dev5 = Deconv(512, 128)
        self.dev6 = Deconv(256, 64)
        

        self.att1 = Att_mul(512, 512)
        self.att2 = Att_mul(512, 512)
        self.att3 = Att_mul(512, 512)
        self.att4 = Att_mul(256, 256)
        self.att5 = Att_mul(128, 128)
        self.att6 = Att_mul(64, 64)
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()
        self.R = nn.ReLU()
        # 添加一个小目标的二分类头
        self.small_object_head = nn.Conv2d(in_channels=1536, out_channels=2, kernel_size=1)
    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            r1_7 = self.cov7_2(r1_6)#1/128
            #print(r1_7.shape)
            
            
            d1_0=self.dev1(r1_7)



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            r2_7 = self.cov7_4(r2_6)#1/128
            #print(r2_7.shape)
            

            d2_0=self.dev1(r2_7)


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
            r3_7 = self.cov7_6(r3_6)#1/128
            #print(r3_7.shape)
            
            d3_0=self.dev1(r3_7)
           
            
            
            att1=self.att1(x=d1_0, x_1=r1_6, e=d2_0, e_1=r2_6, a=d3_0, a_1=r3_6)     
            se1=self.se1(r1_6)+self.se1(r2_6)+self.se1(r3_6)
            d1=torch.cat((att1,se1),dim=1)
            # print("d1.shape=",d1.shape)
            # print("r1_6=",r1_6.shape)

            att2=self.att2(x=self.dev2(d1),x_1=r1_5,e=self.dev2(d1),e_1=r2_5,a=self.dev2(d1),a_1=r3_5)     
            se2=self.se2(r1_5)+self.se2(r2_5)+self.se2(r3_5)
            d2=torch.cat((att2,se2),dim=1)
            # print("d2.shape=",d2.shape)
            # print("r1_5=",r1_5.shape)
            # print("r2_5=",r2_5.shape)
            # print("r3_5=",r3_5.shape)
            # print("self.dev2(d2)=",self.dev2(d2).shape)
            

            att3=self.att3(x=self.dev3(d2),x_1=r1_4,e=self.dev2(d2),e_1=r2_4,a=self.dev2(d2),a_1=r3_4)     
            se3=self.se3(r1_4)+self.se3(r2_4)+self.se3(r3_4)
            d3=torch.cat((att3,se3),dim=1)
            

            att4=self.att4(x=self.dev4(d3),x_1=r1_3,e=self.dev4(d3),e_1=r2_3,a=self.dev4(d3),a_1=r3_3) 
            se4=self.se4(r1_3)+self.se4(r2_3)+self.se4(r3_3)
            d4=torch.cat((att4,se4),dim=1)

            att5=self.att5(x=self.dev5(d4),x_1=r1_2,e=self.dev5(d4),e_1=r2_2,a=self.dev5(d4),a_1=r3_2)     
            se5=self.se5(r1_2)+self.se5(r2_2)+self.se5(r3_2)
            d5=torch.cat((att5,se5),dim=1)
            # print("d5=",d5.shape)

            att6=self.att6(x=self.dev6(d5),x_1=r1_1,e=self.dev6(d5),e_1=r2_1,a=self.dev6(d5),a_1=r3_1)     
            se6=self.se6(r1_1)+self.se6(r2_1)+self.se6(r3_1)
            d6=torch.cat((att6,se6),dim=1)

            combined_r1 = torch.cat([r1_1, r2_1, r3_1], dim=1)
            combined_r2 = torch.cat([r1_2, r2_2, r3_2], dim=1)
            combined_r3 = torch.cat([r1_3, r2_3, r3_3], dim=1)
            combined_r4 = torch.cat([r1_4, r2_4, r3_4], dim=1)
            combined_r5 = torch.cat([r1_5, r2_5, r3_5], dim=1)
            combined_r6 = torch.cat([r1_6, r2_6, r3_6], dim=1)
            combined_r7 = torch.cat([r1_7, r2_7, r3_7], dim=1)
            small_object_output = self.small_object_head(combined_r4)
            out = self.out(d6)
            #out = self.R(out)

            return out, small_object_output







class Mult_SEA_SOT123(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(Mult_SEA_SOT123,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_block_2(input_channels, filter[0])#128*128
        self.cov2_2 = Conv_block_2(filter[0], filter[1])
        self.cov3_2 = Conv_block_2(filter[1], filter[2])
        self.cov4_2 = Conv_block_2(filter[2], filter[3])
        self.cov5_2 = Conv_block_2(filter[3], filter[4])
        self.cov6_2 = Conv_block_2(filter[4], filter[5])
        self.cov7_2 = Conv_block_2(filter[5], filter[6])
        self.cov8_2 = Conv_block_2(filter[6], filter[7])


        self.cov1_4 = Conv_block_1(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_block_1(filter[0], filter[1])
        self.cov3_4 = Conv_block_1(filter[1], filter[2])
        self.cov4_4 = Conv_block_1(filter[2], filter[3])
        self.cov5_4 = Conv_block_1(filter[3], filter[4])
        self.cov6_4 = Conv_block_1(filter[4], filter[5])
        self.cov7_4 = Conv_block_1(filter[5], filter[6])
        self.cov8_4 = Conv_block_1(filter[6], filter[7])


        self.cov1_6 = Conv_block_3(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_block_3(filter[0], filter[1])
        self.cov3_6 = Conv_block_3(filter[1], filter[2])
        self.cov4_6 = Conv_block_3(filter[2], filter[3])
        self.cov5_6 = Conv_block_3(filter[3], filter[4])
        self.cov6_6 = Conv_block_3(filter[4], filter[5])
        self.cov7_6 = Conv_block_3(filter[5], filter[6])
        self.cov8_6 = Conv_block_3(filter[6], filter[7])





        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 512)
        self.dev4 = Deconv(1024, 256)
        self.dev5 = Deconv(512, 128)
        self.dev6 = Deconv(256, 64)
        

        self.att1 = Att_mul(512, 512)
        self.att2 = Att_mul(512, 512)
        self.att3 = Att_mul(512, 512)
        self.att4 = Att_mul(256, 256)
        self.att5 = Att_mul(128, 128)
        self.att6 = Att_mul(64, 64)
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()
        self.R = nn.ReLU()
        # 添加3个小目标的二分类头
        self.small_object_head1 = nn.Conv2d(in_channels=384, out_channels=2, kernel_size=1)
        self.small_object_head2 = nn.Conv2d(in_channels=1536, out_channels=2, kernel_size=1)
        self.small_object_head3 = nn.Conv2d(in_channels=1536, out_channels=2, kernel_size=1)
    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            r1_7 = self.cov7_2(r1_6)#1/128
            #print(r1_7.shape)
            
            
            d1_0=self.dev1(r1_7)



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            r2_7 = self.cov7_4(r2_6)#1/128
            #print(r2_7.shape)
            

            d2_0=self.dev1(r2_7)


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
            r3_7 = self.cov7_6(r3_6)#1/128
            #print(r3_7.shape)
            
            d3_0=self.dev1(r3_7)
           
            
            
            att1=self.att1(x=d1_0, x_1=r1_6, e=d2_0, e_1=r2_6, a=d3_0, a_1=r3_6)     
            se1=self.se1(r1_6)+self.se1(r2_6)+self.se1(r3_6)
            d1=torch.cat((att1,se1),dim=1)
            # print("d1.shape=",d1.shape)
            # print("r1_6=",r1_6.shape)

            att2=self.att2(x=self.dev2(d1),x_1=r1_5,e=self.dev2(d1),e_1=r2_5,a=self.dev2(d1),a_1=r3_5)     
            se2=self.se2(r1_5)+self.se2(r2_5)+self.se2(r3_5)
            d2=torch.cat((att2,se2),dim=1)
            # print("d2.shape=",d2.shape)
            # print("r1_5=",r1_5.shape)
            # print("r2_5=",r2_5.shape)
            # print("r3_5=",r3_5.shape)
            # print("self.dev2(d2)=",self.dev2(d2).shape)
            

            att3=self.att3(x=self.dev3(d2),x_1=r1_4,e=self.dev2(d2),e_1=r2_4,a=self.dev2(d2),a_1=r3_4)     
            se3=self.se3(r1_4)+self.se3(r2_4)+self.se3(r3_4)
            d3=torch.cat((att3,se3),dim=1)
            

            att4=self.att4(x=self.dev4(d3),x_1=r1_3,e=self.dev4(d3),e_1=r2_3,a=self.dev4(d3),a_1=r3_3) 
            se4=self.se4(r1_3)+self.se4(r2_3)+self.se4(r3_3)
            d4=torch.cat((att4,se4),dim=1)

            att5=self.att5(x=self.dev5(d4),x_1=r1_2,e=self.dev5(d4),e_1=r2_2,a=self.dev5(d4),a_1=r3_2)     
            se5=self.se5(r1_2)+self.se5(r2_2)+self.se5(r3_2)
            d5=torch.cat((att5,se5),dim=1)
            # print("d5=",d5.shape)

            att6=self.att6(x=self.dev6(d5),x_1=r1_1,e=self.dev6(d5),e_1=r2_1,a=self.dev6(d5),a_1=r3_1)     
            se6=self.se6(r1_1)+self.se6(r2_1)+self.se6(r3_1)
            d6=torch.cat((att6,se6),dim=1)

            combined_r1 = torch.cat([r1_1, r2_1, r3_1], dim=1)
            combined_r2 = torch.cat([r1_2, r2_2, r3_2], dim=1)
            combined_r3 = torch.cat([r1_3, r2_3, r3_3], dim=1)
            combined_r4 = torch.cat([r1_4, r2_4, r3_4], dim=1)
            combined_r5 = torch.cat([r1_5, r2_5, r3_5], dim=1)
            combined_r6 = torch.cat([r1_6, r2_6, r3_6], dim=1)
            combined_r7 = torch.cat([r1_7, r2_7, r3_7], dim=1)
            small_object_output1 = self.small_object_head1(combined_r2)
            small_object_output2 = self.small_object_head2(combined_r4)
            small_object_output3 = self.small_object_head3(combined_r6)
            out = self.out(d6)
            #out = self.R(out)

            return out, small_object_output1, small_object_output2, small_object_output3




# 定义用于生成偏移字段和进行可变卷积的块
class DeformConvBlock_1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, Dropout=0.5):
        super(DeformConvBlock_1, self).__init__()
        self.offsets = nn.Conv2d(in_channel, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.deform_conv = DeformConv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        offsets = self.offsets(x)
        x = self.deform_conv(x, offsets)
        x = self.layer(x)
        return x

# 替换 Conv_block_1 中的普通卷积
class Conv_block_1_with_DeformConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, Dropout=0.5):
        super(Conv_block_1_with_DeformConv, self).__init__()
        self.layer = DeformConvBlock_1(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=1, Dropout=Dropout)
        
    def forward(self, x):
        return self.layer(x)

# 替换 Conv_block_2 中的普通卷积
class Conv_block_2_with_DeformConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=2, Dropout=0.5):
        super(Conv_block_2_with_DeformConv, self).__init__()
        self.layer = DeformConvBlock_1(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=0, Dropout=Dropout)
        
    def forward(self, x):
        return self.layer(x)

# 替换 Conv_block_3 中的普通卷积
class Conv_block_3_with_DeformConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=6, Dropout=0.5):
        super(Conv_block_3_with_DeformConv, self).__init__()
        self.layer = DeformConvBlock_1(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=2, Dropout=Dropout)
        
    def forward(self, x):
        return self.layer(x)


class MultDeform_SEA_Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(MultDeform_SEA_Unet,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_block_2_with_DeformConv(input_channels, filter[0])#128*128
        self.cov2_2 = Conv_block_2_with_DeformConv(filter[0], filter[1])
        self.cov3_2 = Conv_block_2_with_DeformConv(filter[1], filter[2])
        self.cov4_2 = Conv_block_2_with_DeformConv(filter[2], filter[3])
        self.cov5_2 = Conv_block_2_with_DeformConv(filter[3], filter[4])
        self.cov6_2 = Conv_block_2_with_DeformConv(filter[4], filter[5])
        self.cov7_2 = Conv_block_2_with_DeformConv(filter[5], filter[6])
        self.cov8_2 = Conv_block_2_with_DeformConv(filter[6], filter[7])


        self.cov1_4 = Conv_block_1_with_DeformConv(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_block_1_with_DeformConv(filter[0], filter[1])
        self.cov3_4 = Conv_block_1_with_DeformConv(filter[1], filter[2])
        self.cov4_4 = Conv_block_1_with_DeformConv(filter[2], filter[3])
        self.cov5_4 = Conv_block_1_with_DeformConv(filter[3], filter[4])
        self.cov6_4 = Conv_block_1_with_DeformConv(filter[4], filter[5])
        self.cov7_4 = Conv_block_1_with_DeformConv(filter[5], filter[6])
        self.cov8_4 = Conv_block_1_with_DeformConv(filter[6], filter[7])


        self.cov1_6 = Conv_block_3_with_DeformConv(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_block_3_with_DeformConv(filter[0], filter[1])
        self.cov3_6 = Conv_block_3_with_DeformConv(filter[1], filter[2])
        self.cov4_6 = Conv_block_3_with_DeformConv(filter[2], filter[3])
        self.cov5_6 = Conv_block_3_with_DeformConv(filter[3], filter[4])
        self.cov6_6 = Conv_block_3_with_DeformConv(filter[4], filter[5])
        self.cov7_6 = Conv_block_3_with_DeformConv(filter[5], filter[6])
        self.cov8_6 = Conv_block_3_with_DeformConv(filter[6], filter[7])





        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 512)
        self.dev4 = Deconv(1024, 256)
        self.dev5 = Deconv(512, 128)
        self.dev6 = Deconv(256, 64)
        

        self.att1 = Att_mul(512, 512)
        self.att2 = Att_mul(512, 512)
        self.att3 = Att_mul(512, 512)
        self.att4 = Att_mul(256, 256)
        self.att5 = Att_mul(128, 128)
        self.att6 = Att_mul(64, 64)
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()
        self.R = nn.ReLU()
        
    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            r1_7 = self.cov7_2(r1_6)#1/128
            #print(r1_7.shape)
            
            
            d1_0=self.dev1(r1_7)



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            r2_7 = self.cov7_4(r2_6)#1/128
            #print(r2_7.shape)
            

            d2_0=self.dev1(r2_7)


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
            r3_7 = self.cov7_6(r3_6)#1/128
            #print(r3_7.shape)
            
            d3_0=self.dev1(r3_7)
           
            
            
            att1=self.att1(x=d1_0, x_1=r1_6, e=d2_0, e_1=r2_6, a=d3_0, a_1=r3_6)     
            se1=self.se1(r1_6)+self.se1(r2_6)+self.se1(r3_6)
            d1=torch.cat((att1,se1),dim=1)
            # print("d1.shape=",d1.shape)
            # print("r1_6=",r1_6.shape)

            att2=self.att2(x=self.dev2(d1),x_1=r1_5,e=self.dev2(d1),e_1=r2_5,a=self.dev2(d1),a_1=r3_5)     
            se2=self.se2(r1_5)+self.se2(r2_5)+self.se2(r3_5)
            d2=torch.cat((att2,se2),dim=1)
            # print("d2.shape=",d2.shape)
            # print("r1_5=",r1_5.shape)
            # print("r2_5=",r2_5.shape)
            # print("r3_5=",r3_5.shape)
            # print("self.dev2(d2)=",self.dev2(d2).shape)
            

            att3=self.att3(x=self.dev3(d2),x_1=r1_4,e=self.dev2(d2),e_1=r2_4,a=self.dev2(d2),a_1=r3_4)     
            se3=self.se3(r1_4)+self.se3(r2_4)+self.se3(r3_4)
            d3=torch.cat((att3,se3),dim=1)
            

            att4=self.att4(x=self.dev4(d3),x_1=r1_3,e=self.dev4(d3),e_1=r2_3,a=self.dev4(d3),a_1=r3_3) 
            se4=self.se4(r1_3)+self.se4(r2_3)+self.se4(r3_3)
            d4=torch.cat((att4,se4),dim=1)

            att5=self.att5(x=self.dev5(d4),x_1=r1_2,e=self.dev5(d4),e_1=r2_2,a=self.dev5(d4),a_1=r3_2)     
            se5=self.se5(r1_2)+self.se5(r2_2)+self.se5(r3_2)
            d5=torch.cat((att5,se5),dim=1)

            att6=self.att6(x=self.dev6(d5),x_1=r1_1,e=self.dev6(d5),e_1=r2_1,a=self.dev6(d5),a_1=r3_1)     
            se6=self.se6(r1_1)+self.se6(r2_1)+self.se6(r3_1)
            d6=torch.cat((att6,se6),dim=1)

            
            
            out = self.out(d6)
            #out = self.R(out)

            return out




class MultDeform_SEA_SOT(nn.Module):
    def __init__(self, input_channels=3, num_classes=1,  **kwargs):
        super(MultDeform_SEA_SOT,self).__init__()
        #          0   1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5   6  7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
       
       
        
        self.cov1_2 = Conv_block_2_with_DeformConv(input_channels, filter[0])#128*128
        self.cov2_2 = Conv_block_2_with_DeformConv(filter[0], filter[1])
        self.cov3_2 = Conv_block_2_with_DeformConv(filter[1], filter[2])
        self.cov4_2 = Conv_block_2_with_DeformConv(filter[2], filter[3])
        self.cov5_2 = Conv_block_2_with_DeformConv(filter[3], filter[4])
        self.cov6_2 = Conv_block_2_with_DeformConv(filter[4], filter[5])
        self.cov7_2 = Conv_block_2_with_DeformConv(filter[5], filter[6])
        self.cov8_2 = Conv_block_2_with_DeformConv(filter[6], filter[7])


        self.cov1_4 = Conv_block_1_with_DeformConv(input_channels, filter[0])#128*128
        self.cov2_4 = Conv_block_1_with_DeformConv(filter[0], filter[1])
        self.cov3_4 = Conv_block_1_with_DeformConv(filter[1], filter[2])
        self.cov4_4 = Conv_block_1_with_DeformConv(filter[2], filter[3])
        self.cov5_4 = Conv_block_1_with_DeformConv(filter[3], filter[4])
        self.cov6_4 = Conv_block_1_with_DeformConv(filter[4], filter[5])
        self.cov7_4 = Conv_block_1_with_DeformConv(filter[5], filter[6])
        self.cov8_4 = Conv_block_1_with_DeformConv(filter[6], filter[7])


        self.cov1_6 = Conv_block_3_with_DeformConv(input_channels, filter[0])#128*128
        self.cov2_6 = Conv_block_3_with_DeformConv(filter[0], filter[1])
        self.cov3_6 = Conv_block_3_with_DeformConv(filter[1], filter[2])
        self.cov4_6 = Conv_block_3_with_DeformConv(filter[2], filter[3])
        self.cov5_6 = Conv_block_3_with_DeformConv(filter[3], filter[4])
        self.cov6_6 = Conv_block_3_with_DeformConv(filter[4], filter[5])
        self.cov7_6 = Conv_block_3_with_DeformConv(filter[5], filter[6])
        self.cov8_6 = Conv_block_3_with_DeformConv(filter[6], filter[7])





        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(512, 8)
        self.se3 = SE_Res(512, 8)
        self.se4 = SE_Res(256, 8)
        self.se5 = SE_Res(128, 8)
        self.se6 = SE_Res(64, 8)
        
        #           0   1    2    3    4    5    6    7
        #filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #            0     1     2     3     4    5   6     7
        #Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 512)
        self.dev3 = Deconv(1024, 512)
        self.dev4 = Deconv(1024, 256)
        self.dev5 = Deconv(512, 128)
        self.dev6 = Deconv(256, 64)
        

        self.att1 = Att_mul(512, 512)
        self.att2 = Att_mul(512, 512)
        self.att3 = Att_mul(512, 512)
        self.att4 = Att_mul(256, 256)
        self.att5 = Att_mul(128, 128)
        self.att6 = Att_mul(64, 64)
        

        self.out = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()
        self.R = nn.ReLU()
        # 添加一个小目标的二分类头
        self.small_object_head = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
    def forward(self, x):
            r1_1 = self.cov1_2(x)#1/2
            #print("r1_1.shape=",r1_1.shape)
            r1_2 = self.cov2_2(r1_1)#1/4
            #print(r1_2.shape)
            r1_3 = self.cov3_2(r1_2)#1/8
            #print(r1_3.shape)
            r1_4 = self.cov4_2(r1_3)#1/16
            #print(r1_4.shape)
            r1_5 = self.cov5_2(r1_4)#1/32
            #print(r1_5.shape)
            r1_6 = self.cov6_2(r1_5)#1/64
            #print(r1_6.shape)
            r1_7 = self.cov7_2(r1_6)#1/128
            #print(r1_7.shape)
            
            
            d1_0=self.dev1(r1_7)



            r2_1 = self.cov1_4(x)#1/2
            #print("r2_1.shape=",r2_1.shape)
            r2_2 = self.cov2_4(r2_1)#1/4
            #print(r2_2.shape)
            r2_3 = self.cov3_4(r2_2)#1/8
            #print(r2_3.shape)
            r2_4 = self.cov4_4(r2_3)#1/16
            #print(r2_4.shape)
            r2_5 = self.cov5_4(r2_4)#1/32
            #print(r2_5.shape)
            r2_6 = self.cov6_4(r2_5)#1/64
            #print(r2_6.shape)
            r2_7 = self.cov7_4(r2_6)#1/128
            #print(r2_7.shape)
            

            d2_0=self.dev1(r2_7)


            r3_1 = self.cov1_6(x)#1/2
            #print("r3_1.shape=",r3_1.shape)
            r3_2 = self.cov2_6(r3_1)#1/4
            #print(r3_2.shape)
            r3_3 = self.cov3_6(r3_2)#1/8
            #print(r3_3.shape)
            r3_4 = self.cov4_6(r3_3)#1/16
            #print(r3_4.shape)
            r3_5 = self.cov5_6(r3_4)#1/32
            #print(r3_5.shape)
            r3_6 = self.cov6_6(r3_5)#1/64
            #print(r3_6.shape)
            r3_7 = self.cov7_6(r3_6)#1/128
            #print(r3_7.shape)
            
            d3_0=self.dev1(r3_7)
           
            
            
            att1=self.att1(x=d1_0, x_1=r1_6, e=d2_0, e_1=r2_6, a=d3_0, a_1=r3_6)     
            se1=self.se1(r1_6)+self.se1(r2_6)+self.se1(r3_6)
            d1=torch.cat((att1,se1),dim=1)
            # print("d1.shape=",d1.shape)
            # print("r1_6=",r1_6.shape)

            att2=self.att2(x=self.dev2(d1),x_1=r1_5,e=self.dev2(d1),e_1=r2_5,a=self.dev2(d1),a_1=r3_5)     
            se2=self.se2(r1_5)+self.se2(r2_5)+self.se2(r3_5)
            d2=torch.cat((att2,se2),dim=1)
            # print("d2.shape=",d2.shape)
            # print("r1_5=",r1_5.shape)
            # print("r2_5=",r2_5.shape)
            # print("r3_5=",r3_5.shape)
            # print("self.dev2(d2)=",self.dev2(d2).shape)
            

            att3=self.att3(x=self.dev3(d2),x_1=r1_4,e=self.dev2(d2),e_1=r2_4,a=self.dev2(d2),a_1=r3_4)     
            se3=self.se3(r1_4)+self.se3(r2_4)+self.se3(r3_4)
            d3=torch.cat((att3,se3),dim=1)
            

            att4=self.att4(x=self.dev4(d3),x_1=r1_3,e=self.dev4(d3),e_1=r2_3,a=self.dev4(d3),a_1=r3_3) 
            se4=self.se4(r1_3)+self.se4(r2_3)+self.se4(r3_3)
            d4=torch.cat((att4,se4),dim=1)

            att5=self.att5(x=self.dev5(d4),x_1=r1_2,e=self.dev5(d4),e_1=r2_2,a=self.dev5(d4),a_1=r3_2)     
            se5=self.se5(r1_2)+self.se5(r2_2)+self.se5(r3_2)
            d5=torch.cat((att5,se5),dim=1)

            att6=self.att6(x=self.dev6(d5),x_1=r1_1,e=self.dev6(d5),e_1=r2_1,a=self.dev6(d5),a_1=r3_1)     
            se6=self.se6(r1_1)+self.se6(r2_1)+self.se6(r3_1)
            d6=torch.cat((att6,se6),dim=1)

            
            small_object_output = self.small_object_head(d6)
            out = self.out(d6)
            #out = self.R(out)

            return out, small_object_output







#调试网络U++ 
# if __name__ == "__main__":
#     print("deep_supervision: False")
#     deep_supervision = False
#     device = torch.device('cpu')
#     inputs = torch.randn((1, 3, 256, 256)).to(device)
#     model = UnetPlusPlus(num_classes=1, deep_supervision=deep_supervision).to(device)
#     outputs = model(inputs)
#     print(outputs.shape)    
    
#     print("deep_supervision: True")
#     deep_supervision = True
#     model = UnetPlusPlus(num_classes=3, deep_supervision=deep_supervision).to(device)
#     outputs = model(inputs)
#     for out in outputs:
#       print(out.shape)
 

 






#调试网络Unet和其他自建
# modle_test=Mult_SAA_Unet
# if __name__ == '__main__':
#     #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#     #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#     torch.cuda.empty_cache()
#     '''x = torch.randn(2, 3, 256, 256)
#     x = x.cuda()
#     net = SEA_Unet(input_channels=3, num_classes=2).cuda()
#     x = net(x)
#     print(x.shape)
#     torch.cuda.empty_cache() '''
#     x = torch.randn(2, 3, 256, 256)
#     x = x.cuda()
#     model = modle_test(input_channels=3, num_classes=3).cuda()
#     outputs = model(x)
#     a = fun_loss()
#     loss = a(x, outputs)
#     print(loss)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     #print(torch.cuda.is_available())
#     torch.cuda.empty_cache()



#查看模型的空间复杂度
# model = SEA_Unet()

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)



# print(f"模型的参数量: {count_parameters(model)}")


#空间复杂度和时间复杂度

# model = NL_Unet_5(num_classes=1).cuda()#UnetPlusPlus SEA_Unet U_Net_original

# dummy_input = torch.randn(1, 3, 160, 224).cuda()
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
