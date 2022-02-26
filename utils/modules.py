import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True) -> None:
        super(Conv, self).__init__()

        # dilation， groups什么意思？
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )
    
    def forward(self, x):
        return self.convs(x)

class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """

    # TODO: 为什么自动生成的__init__模板代码，有时候有-> None，有时候没有
    def __init__(self) -> None:
        super(SPP, self).__init__()
    
    def forward(self, x):
        '''
            nn.ReLU         --->    nn.functional.relu
            nn.Maxpool2d    --->    nn.functional.max_pool2d
        
            前者作为一个层结构，必须添加在nn.Module容器中才能使用（功能上类似于c/c++函数指针）
            后者单纯作为一个函数调用，直接使用

            self.relu = nn.ReLU(inplace=True)
            self.relu(x)

            nn.functional.relu(x)
        '''

        # padding的值计算方法，保证输出的维度与输入维度相同
        x_1 = nn.functional.max_pool2d(x, 5, stride=1, padding=2)       # [32, 512, 13, 13]
        x_2 = nn.functional.max_pool2d(x, 9, stride=1, padding=4)       # [32, 512, 13, 13]
        x_3 = nn.functional.max_pool2d(x, 13, stride=1, padding=6)      # [32, 512, 13, 13]

        # dim=1详细解释: 将4个矩阵按照第一个维度堆叠起来，在此处即为 512+512+512+512 -> 2048
        x = torch.cat([x, x_1, x_2, x_3], dim=1)                        # [32, 2048, 13, 13]

        return x