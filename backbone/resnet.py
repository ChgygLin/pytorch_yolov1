
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


"""
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
                    groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    
    input size (N,C_in,H,W) --> (N,C_out,H_out,W_out)

    使用C_out个卷积核即可增加通道数

    feture map: O = (I-K+2P)/S + 1


        kernel_size:    int或tuple; (2, 3)分别表示宽高的卷积核
        stride:         int或tuple; (2, 3)分别表示宽高的步长
        padding:        在feature map外围增加几圈0
                        当padding=1时,如果原来大小为3x3,那么处理之后为5x5,即在宽、高在上下左右均增加1
        dilation:       控制卷积核之间的间距, 控制输出为正常卷积或空洞卷积, 默认正常卷积

        bias:           若卷积后面接BN操作,最好不设置bias,不起作用且占用显存
                        https://blog.csdn.net/u013289254/article/details/98785869
    
    3x3卷积,stride=1,padding=1
    1x1卷积,stride=1,padding=0
"""


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    # 1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)       # inplace表示什么？？
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x                # 什么意思？

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            # 加下划线与不加的区别？？
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        # Zero-initialize the last BN in each residual branch, 
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), 
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        C_1 = self.conv1(x)         # x:    [32, 3, 416, 416]
        C_1 = self.bn1(C_1)         # C_1:  [32, 64, 208, 208]
        C_1 = self.relu(C_1)        # C_1:  [32, 64, 208, 208]
        C_1 = self.maxpool(C_1)     # C_1:  [32, 64, 104, 104]

        C_2 = self.layer1(C_1)      # C_2:  [32, 64, 104, 104]
        C_3 = self.layer2(C_2)      # C_3:  [32, 128, 52, 52]
        C_4 = self.layer3(C_3)      # C_4:  [32, 256, 26, 26]
        C_5 = self.layer4(C_4)      # C_5:  [32, 512, 13, 13]

        return C_5


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # TODO: model_zoo做什么用的？？
        # strict = False as we don't need fc layer params.
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

    return model



if __name__ == '__main__':
    print("Found", torch.cuda.device_count, " GPU(s)")
    device = torch.device("cuda")
    model = resnet18(detection=True).to(device)
    print(model)

    intput = torch.randn(1, 3, 512, 512).to(device)
    output = model(input)










