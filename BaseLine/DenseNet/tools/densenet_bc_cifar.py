# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)  # 在此体现dense connection，两个特征图拼接起来
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3   # 4 表示第一个卷积层 + 2个transition + 1个FC层输出分类 ；；；3表示3个block
        if bottleneck:
            nDenseBlocks //= 2  # 2表示一个基础操作模块中有2层

        nChannels = 2*growthRate    # 第一个卷积的卷积核个数为 growthRate的两倍
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)

        # 第一个 denseblock
        self.dense1 = self._make_denseblock(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate                 # 计算经过denseblock之后有多少个通道
        nOutChannels = int(math.floor(nChannels*reduction))  # 计算经过Compression后，还有几个通道reduction对应论文中的θ
        self.trans1 = Transition(nChannels, nOutChannels)

        # 第二个 denseblock
        nChannels = nOutChannels
        self.dense2 = self._make_denseblock(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        # 第三个 denseblock
        nChannels = nOutChannels
        self.dense3 = self._make_denseblock(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        # 分类输出层
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        # 权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_denseblock(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        """
        创建denseblock
        :param nChannels: 进入block的特征图的通道数
        :param growthRate: int
        :param nDenseBlocks: blocks堆叠的数量
        :param bottleneck: boolean，是否需要bottleneck， densenet-b
        :return:
        """
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate     # 通道数逐渐增加，从此看出block中各层的输入均包含进入block的特征图
        return nn.Sequential(*layers)

    def forward(self, x):

        # 1/3 头部的卷积
        out = self.conv1(x)

        # 2/3 dense block 的堆叠
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        # 3/3 池化+全连接，输出分类
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        out = self.fc(out)
        return out