import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
class USAM(nn.Module):
    def __init__(self, kernel_size=3, padding=1, polish=True):
        super(USAM, self).__init__()

        kernel = torch.ones((kernel_size, kernel_size))
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        

        kernel2 = torch.ones((1, 1)) * (kernel_size * kernel_size)
        kernel2 = kernel2.unsqueeze(0).unsqueeze(0)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

        self.polish = polish
        self.pad = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def __call__(self, x):
        fmap = x.sum(1, keepdim=True)      
        x1 = F.conv2d(fmap, self.weight, padding=self.pad)
        x2 = F.conv2d(fmap, self.weight2, padding=0) 
        
        att = x2 - x1
        att = self.bn(att)
        att = self.relu(att)

        if self.polish:
            mask = torch.ones_like(att)
            mask[:, :, :, 0] = 0
            mask[:, :, :, -1] = 0
            mask[:, :, 0, :] = 0
            mask[:, :, -1, :] = 0
            att = att * mask

        output = x + att * x

        return output


class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim = 2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        # m.inplace=False
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# class ClassBlock(nn.Module):
#     def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
#         super(ClassBlock, self).__init__()
#         self.return_f = return_f
#         add_block = []
#         if linear:
#             add_block += [nn.Linear(input_dim, num_bottleneck)]
#         else:
#             num_bottleneck = input_dim
#         if bnorm:
#             add_block += [nn.BatchNorm1d(num_bottleneck)]
#         if relu:
#             add_block += [nn.LeakyReLU(0.1)]
#         if droprate>0:
#             add_block += [nn.Dropout(p=droprate)]
#         add_block = nn.Sequential(*add_block)
#         add_block.apply(weights_init_kaiming)

#         classifier = []
#         classifier += [nn.Linear(num_bottleneck, class_num)]
#         classifier = nn.Sequential(*classifier)
#         classifier.apply(weights_init_classifier)

#         self.add_block = add_block
#         self.classifier = classifier
#     def forward(self, x):
#         x = self.add_block(x)
#         if self.return_f:
#             f = x
#             x = self.classifier(x)
#             return x,f
#         else:
#             x = self.classifier(x)
#             return x

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num, sp, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        # 初始化池化层（根据 pool 类型）
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            # 若用 GeM，需确保 GeM 是 nn.Module 子类（参考你的代码补充）
            model_ft.gem2 = GeM(dim=2048)  

        if init_model is not None:
            model_ft = init_model.model
            self.pool = init_model.pool

        self.usam_1 = USAM()  # 确保 USAM 是 nn.Module 子类
        self.usam_2 = USAM()

        # 构建模块列表（无 lambda）
        self.total_module_list = [
            model_ft.conv1,
            model_ft.bn1,
            model_ft.relu,
            self.usam_1,
            model_ft.maxpool,
            
            model_ft.layer1,
            self.usam_2,
            model_ft.layer2,
            model_ft.layer3,
            model_ft.layer4
        ]

        # 根据 pool 类型扩展模块列表
        if pool == 'avg+max':
            self.total_module_list.extend([
                model_ft.avgpool2,
                model_ft.maxpool2,
                Concat(),   # 替换 lambda 拼接
                Flatten()   # 替换 lambda 展平
            ])
        elif pool == 'avg':
            self.total_module_list.extend([
                model_ft.avgpool2,
                Flatten()   # 替换 lambda 展平
            ])
        elif pool == 'max':
            self.total_module_list.extend([
                model_ft.maxpool2,
                Flatten()   # 替换 lambda 展平
            ])
        elif pool == 'gem':
            self.total_module_list.extend([
                model_ft.gem2,
                Flatten()   # 替换 lambda 展平
            ])

        # 分割为 f 和 g
        self.f = nn.Sequential(*self.total_module_list[0:sp])
        self.g = nn.Sequential(*self.total_module_list[sp:])

    def forward(self, x):
        x = self.f(x)
        x_attack = x.clone().detach()
        x = self.g(x)
        return x, x_attack
# Define the ResNet50-based Model
class blackbox_attack0(nn.Module):
    def __init__(self, num_classes, sp=5, in_channels=3, fig_size=192):
        # 确保 sp 范围有效
        assert 0 < sp <= 17, "sp 需满足 0 < sp <= 17"
        super(blackbox_attack0, self).__init__()  # 修正继承，原为错误的 VGG9
        
        # 网络配置（通道数序列）
        cfg = [32, 64]
        
        # ------------------- 新增：上采样转置卷积层 -------------------
        # 目标：将 192×192 放大到 384×384（尺寸翻倍）
        self.upsample = nn.ConvTranspose2d(
            in_channels=cfg[1],    # 输入通道数（与 conv1 输出匹配）
            out_channels=cfg[1],   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        # self.upsample2 = nn.ConvTranspose2d(
        #     in_channels=cfg[2],    # 输入通道数（与 conv1 输出匹配）
        #     out_channels=cfg[2],   # 输出通道数（保持与输入一致）
        #     kernel_size=3,         # 卷积核大小
        #     stride=2,              # 步长=2 → 尺寸×2
        #     padding=1,             # 填充
        #     output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
        #     bias=False             # 配合 BatchNorm 更稳定
        # )
        # 可选：添加 BatchNorm 和 ReLU 增强非线性（根据需求调整）
        self.upsample_bn = nn.BatchNorm2d(cfg[1])
        self.upsample_relu = nn.ReLU(inplace=True)
        
        # ------------------- 原有卷积层 -------------------
     
        self.conv1 = nn.Conv2d(cfg[1], cfg[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(cfg[0], in_channels, kernel_size=3, stride=1, padding=1)
        
        # ------------------- 构建模块列表（插入上采样流程） -------------------
        self.total_module_list = [
            # 插入上采样流程：64×192×192 → 64×384×384
            self.upsample, 
            self.upsample_bn,     
            self.upsample_relu,   
            self.conv1,            
            nn.ReLU(inplace=True), 
            self.conv2, 
               
            
        ]

    def forward(self, x):
        # 按模块列表顺序执行
        for module in self.total_module_list:
            x = module(x)
        return x

class blackbox_attack3(nn.Module):
    def __init__(self, num_classes, sp=5, in_channels=3, fig_size=192):
        # 确保 sp 范围有效
        assert 0 < sp <= 17, "sp 需满足 0 < sp <= 17"
        super(blackbox_attack3, self).__init__()  # 修正继承，原为错误的 VGG9
        
        # 网络配置（通道数序列）
        cfg = [32,64, 64]
        
        # ------------------- 新增：上采样转置卷积层 -------------------
        # 目标：将 192×192 放大到 384×384（尺寸翻倍）
        self.upsample = nn.ConvTranspose2d(
            in_channels=cfg[1],    # 输入通道数（与 conv1 输出匹配）
            out_channels=cfg[1],   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        # self.upsample2 = nn.ConvTranspose2d(
        #     in_channels=cfg[2],    # 输入通道数（与 conv1 输出匹配）
        #     out_channels=cfg[2],   # 输出通道数（保持与输入一致）
        #     kernel_size=3,         # 卷积核大小
        #     stride=2,              # 步长=2 → 尺寸×2
        #     padding=1,             # 填充
        #     output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
        #     bias=False             # 配合 BatchNorm 更稳定
        # )
        # 可选：添加 BatchNorm 和 ReLU 增强非线性（根据需求调整）
        self.upsample_bn = nn.BatchNorm2d(cfg[1])
        self.upsample_relu = nn.ReLU(inplace=True)
        
        # ------------------- 原有卷积层 -------------------
        self.conv1 = nn.Conv2d(cfg[2], cfg[1], kernel_size=3, stride=1, padding=1)
     
        self.conv2 = nn.Conv2d(cfg[1], cfg[0], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(cfg[0], in_channels, kernel_size=3, stride=1, padding=1)
        
        # ------------------- 构建模块列表（插入上采样流程） -------------------
        self.total_module_list = [
            # 插入上采样流程：64×192×192 → 64×384×384
            self.upsample, 
            self.upsample_bn,     
            self.upsample_relu,   
            self.conv1,            
            nn.ReLU(inplace=True), 
            self.conv2,
            nn.ReLU(inplace=True), 
            self.conv3,
               
            
        ]

    def forward(self, x):
        # 按模块列表顺序执行
        for module in self.total_module_list:
            x = module(x)
        return x

class blackbox_attack5(nn.Module):
    def __init__(self, num_classes, sp=5, in_channels=3, fig_size=192):
        # 确保 sp 范围有效
        assert 0 < sp <= 17, "sp 需满足 0 < sp <= 17"
        super(blackbox_attack5, self).__init__()  # 修正继承，原为错误的 VGG9
        
        # 网络配置（通道数序列）
        cfg = [32, 64, 256, 512]
        
        # ------------------- 新增：上采样转置卷积层 -------------------
        # 目标：将 192×192 放大到 384×384（尺寸翻倍）
        self.upsample = nn.ConvTranspose2d(
            in_channels=cfg[2],    # 输入通道数（与 conv1 输出匹配）
            out_channels=cfg[2],   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=cfg[2],    # 输入通道数（与 conv1 输出匹配）
            out_channels=cfg[2],   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        # 可选：添加 BatchNorm 和 ReLU 增强非线性（根据需求调整）
        self.upsample_bn = nn.BatchNorm2d(cfg[2])
        self.upsample_relu = nn.ReLU(inplace=True)
        
        # ------------------- 原有卷积层 -------------------
        self.conv1 = nn.Conv2d(cfg[2], cfg[1], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(cfg[1], cfg[0], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(cfg[0], in_channels, kernel_size=3, stride=1, padding=1)
        
        # ------------------- 构建模块列表（插入上采样流程） -------------------
        self.total_module_list = [
            # 插入上采样流程：64×192×192 → 64×384×384
            self.upsample, 
            self.upsample2,
            self.upsample_bn,     
            self.upsample_relu,   
            self.conv1,            
            nn.ReLU(inplace=True), 
            self.conv2, 
            nn.ReLU(inplace=True),
            self.conv3,            
            
        ]

    def forward(self, x):
        # 按模块列表顺序执行
        for module in self.total_module_list:
            x = module(x)
        return x

class blackbox_attack8(nn.Module):
    def __init__(self, num_classes, out_channels= 1024, in_channels=3, fig_size=192):
        # 确保 sp 范围有效
        
        super(blackbox_attack8, self).__init__()  # 修正继承，原为错误的 VGG9
        
        # 网络配置（通道数序列）
        cfg = [32, 64, 256, 512, 1024] 
        
        # ------------------- 新增：上采样转置卷积层 -------------------
        # 目标：将 192×192 放大到 384×384（尺寸翻倍）
        self.upsample = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample4 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        # 可选：添加 BatchNorm 和 ReLU 增强非线性（根据需求调整）
        self.upsample_bn = nn.BatchNorm2d(out_channels)
        self.upsample_relu = nn.ReLU(inplace=True)
        
        # ------------------- 原有卷积层 -------------------
        self.conv1 = nn.Conv2d(cfg[4], cfg[3], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(cfg[3], cfg[2], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(cfg[2], cfg[1], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(cfg[1], cfg[0], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(cfg[0], in_channels, kernel_size=3, stride=1, padding=1)
        
        # ------------------- 构建模块列表（插入上采样流程） -------------------
        self.total_module_list = [
            # 插入上采样流程：64×192×192 → 64×384×384
            self.upsample, 
            # self.upsample_relu,
            self.upsample2,
            self.upsample_relu,
            self.upsample3,
            self.upsample,
            self.upsample_bn,     
            self.upsample_relu,   
            self.conv1,            
            nn.ReLU(inplace=True), 
            self.conv2, 
            nn.ReLU(inplace=True),
            self.conv3,    
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True),
            self.conv5
        ]

    def forward(self, x):
        # 按模块列表顺序执行
        for module in self.total_module_list:
            x = module(x)
        return x

class blackbox_attack7(nn.Module):
    def __init__(self, num_classes, out_channels= 512, in_channels=3, fig_size=192):
        # 确保 sp 范围有效
        
        super(blackbox_attack7, self).__init__()  # 修正继承，原为错误的 VGG9
        
        # 网络配置（通道数序列）
        cfg = [32, 64, 256, 512]
        
        # ------------------- 新增：上采样转置卷积层 -------------------
        # 目标：将 192×192 放大到 384×384（尺寸翻倍）
        self.upsample = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        # 可选：添加 BatchNorm 和 ReLU 增强非线性（根据需求调整）
        self.upsample_bn = nn.BatchNorm2d(out_channels)
        self.upsample_relu = nn.ReLU(inplace=True)
        
        # ------------------- 原有卷积层 -------------------
        self.conv1 = nn.Conv2d(cfg[3], cfg[0], kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(cfg[2], cfg[1], kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(cfg[1], cfg[0], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(cfg[0], in_channels, kernel_size=3, stride=1, padding=1)
        
        # ------------------- 构建模块列表（插入上采样流程） -------------------
        self.total_module_list = [
            # 插入上采样流程：64×192×192 → 64×384×384
            self.upsample, 
            # self.upsample_relu,
            self.upsample2,
            self.upsample_relu,
            self.upsample3,
            self.upsample_bn,     
            self.upsample_relu,   
            self.conv1,            
            # nn.ReLU(inplace=True), 
            # self.conv2, 
            # nn.ReLU(inplace=True),
            # self.conv3,    
            nn.ReLU(inplace=True),
            self.conv4
            
        ]

    def forward(self, x):
        # 按模块列表顺序执行
        for module in self.total_module_list:
            x = module(x)
        return x

class blackbox_attack9(nn.Module):
    def __init__(self, num_classes, out_channels= 2048, in_channels=3, fig_size=192):
        # 确保 sp 范围有效
        
        super(blackbox_attack9, self).__init__()  # 修正继承，原为错误的 VGG9
        
        # 网络配置（通道数序列）
        cfg = [32, 64, 256, 512, 1024, 2048]
        
        # ------------------- 新增：上采样转置卷积层 -------------------
        # 目标：将 192×192 放大到 384×384（尺寸翻倍）
        self.upsample = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample4 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        self.upsample5 = nn.ConvTranspose2d(
            in_channels=out_channels,    # 输入通道数（与 conv1 输出匹配）
            out_channels=out_channels,   # 输出通道数（保持与输入一致）
            kernel_size=3,         # 卷积核大小
            stride=2,              # 步长=2 → 尺寸×2
            padding=1,             # 填充
            output_padding=1,      # 确保输出尺寸精准翻倍（192→384）
            bias=False             # 配合 BatchNorm 更稳定
        )
        # 可选：添加 BatchNorm 和 ReLU 增强非线性（根据需求调整）
        self.upsample_bn = nn.BatchNorm2d(out_channels)
        self.upsample_relu = nn.ReLU(inplace=True)
        
        # ------------------- 原有卷积层 -------------------
        self.conv1 = nn.Conv2d(cfg[5], cfg[4], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(cfg[4], cfg[3], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(cfg[3], cfg[2], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(cfg[2], cfg[1], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(cfg[1], cfg[0], kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(cfg[0], in_channels, kernel_size=3, stride=1, padding=1)
        
        # ------------------- 构建模块列表（插入上采样流程） -------------------
        self.total_module_list = [
            # 插入上采样流程：64×192×192 → 64×384×384
            self.upsample, 
            # self.upsample_relu,
            self.upsample2,
            self.upsample_relu,
            self.upsample3,
            self.upsample4,
            self.upsample_relu,
            self.upsample5,
            self.upsample_bn,     
            self.upsample_relu,
            self.conv1,            
            self.conv2, 
            nn.ReLU(inplace=True),
            self.conv3,    
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True), 
            self.conv5,
            nn.ReLU(inplace=True), 
            self.conv6
            
        ]

    def forward(self, x):
        # 按模块列表顺序执行
        for module in self.total_module_list:
            x = module(x)
        return x
# # Define the ResNet50-based Model
# class ft_net(nn.Module):

#     def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
#         super(ft_net, self).__init__()
#         model_ft = models.resnet50(pretrained=True)
#         # avg pooling to global pooling
#         if stride == 1:
#             model_ft.layer4[0].downsample[0].stride = (1,1)
#             model_ft.layer4[0].conv2.stride = (1,1)

#         self.pool = pool
#         if pool =='avg+max':
#             model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
#             model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
#             #self.classifier = ClassBlock(4096, class_num, droprate)
#         elif pool=='avg':
#             model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
#             #self.classifier = ClassBlock(2048, class_num, droprate)
#         elif pool=='max':
#             model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
#         elif pool=='gem':
#             model_ft.gem2 = GeM(dim=2048)

#         self.model = model_ft

#         if init_model!=None:
#             self.model = init_model.model
#             self.pool = init_model.pool
#             #self.classifier.add_block = init_model.classifier.add_block

#         self.usam_1 = USAM()
#         self.usam_2 = USAM()
        
#         # 修改压缩模块通道数：64 -> 51 -> 64 (保持20%压缩率)
#         # self.compression_conv = nn.Sequential(
#         #     nn.Conv2d(64, 51, kernel_size=1, stride=1, padding=0),  # 64*0.8=51.2
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(51, 64, kernel_size=1, stride=1, padding=0)   # 恢复原始通道数
#         # )

#     def forward(self, x):
#         x = self.model.conv1(x)
#         x_attack = x.clone().detach() 
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.usam_1(x)
#         x = self.model.maxpool(x)
#         x = self.model.layer1(x)
#         x = self.usam_2(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
        
#         if self.pool == 'avg+max':
#             x1 = self.model.avgpool2(x)
#             x2 = self.model.maxpool2(x)
#             x = torch.cat((x1,x2), dim = 1)
#         elif self.pool == 'avg':
#             x = self.model.avgpool2(x)
#         elif self.pool == 'max':
#             x = self.model.maxpool2(x)
#         elif self.pool == 'gem':
#             x = self.model.gem2(x)

#         x = x.view(x.size(0), x.size(1))
#         return x, x_attack 

# class ft_net_VGG16(nn.Module):

#     def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
#         super(ft_net_VGG16, self).__init__()
#         model_ft = models.vgg16_bn(pretrained=True)
#         # avg pooling to global pooling
#         #if stride == 1:
#         #    model_ft.layer4[0].downsample[0].stride = (1,1)
#         #    model_ft.layer4[0].conv2.stride = (1,1)

#         self.pool = pool
#         if pool =='avg+max':
#             model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
#             model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
#             #self.classifier = ClassBlock(4096, class_num, droprate)
#         elif pool=='avg':
#             model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
#             #self.classifier = ClassBlock(2048, class_num, droprate)
#         elif pool=='max':
#             model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
#         elif pool=='gem':
#             model_ft.gem2 = GeM(dim = 512)

#         self.model = model_ft

#         if init_model!=None:
#             self.model = init_model.model
#             self.pool = init_model.pool
#             #self.classifier.add_block = init_model.classifier.add_block

#     def forward(self, x):
#         x = self.model.features(x)
#         if self.pool == 'avg+max':
#             x1 = self.model.avgpool2(x)
#             x2 = self.model.maxpool2(x)
#             x = torch.cat((x1,x2), dim = 1)
#         elif self.pool == 'avg':
#             x = self.model.avgpool2(x)
#         elif self.pool == 'max':
#             x = self.model.maxpool2(x)
#         elif self.pool=='gem':
#             x = self.model.gem2(x)

#         x = x.view(x.size(0), x.size(1))

#         #x = self.classifier(x)
#         return x


    
# class two_view_net(nn.Module):
#     def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False):
#         super(two_view_net, self).__init__()
#         if VGG16:
#             self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
#         else:
#             self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
#         if share_weight:
#             self.model_2 = self.model_1
#         else:
#             if VGG16:
#                 self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
#             else:
#                 self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

#         self.classifier = ClassBlock(2048, class_num, droprate)
#         if pool =='avg+max':
#             self.classifier = ClassBlock(4096, class_num, droprate)
#         if VGG16:
#             self.classifier = ClassBlock(512, class_num, droprate)
#             if pool =='avg+max':
#                 self.classifier = ClassBlock(1024, class_num, droprate)

#     def forward(self, x1, x2):
#         if x1 is None:
#             y1 = None
#         else:
#             x1 = self.model_1(x1)
#             y1 = self.classifier(x1)

#         if x2 is None:
#             y2 = None
#         else:
#             x2 = self.model_2(x2)
#             y2 = self.classifier(x2)
#         return y1, y2


# class three_view_net(nn.Module):
#     def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False):
#         super(three_view_net, self).__init__()
#         if VGG16:
#             self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
#             self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
#         else:
#             self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
#             self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

#         if share_weight:
#             self.model_3 = self.model_1
#         else:
#             if VGG16:
#                 self.model_3 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
#             else:
#                 self.model_3 =  ft_net(class_num, stride = stride, pool = pool)
#         self.classifier = ClassBlock(2048, class_num, droprate)
#         if pool =='avg+max':
#             self.classifier = ClassBlock(4096, class_num, droprate)

#     def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
#         if x1 is None:
#             y1 = None
#         else:
#             x1 = self.model_1(x1)
#             y1 = self.classifier(x1)

#         if x2 is None:
#             y2 = None
#         else:
#             x2 = self.model_2(x2)
#             y2 = self.classifier(x2)

#         if x3 is None:
#             y3 = None
#         else:
#             x3 = self.model_3(x3)
#             y3 = self.classifier(x3)

#         if x4 is None:
#             return y1, y2, y3
#         else:
#             x4 = self.model_2(x4)
#             y4 = self.classifier(x4)
#             return y1, y2, y3, y4


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(751, droprate=0.5, VGG16=True)
    #net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    output,output = net(input,input)
    print('net output size:')
    print(output.shape)
