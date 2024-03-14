import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet模型类
        Alexnet在[3.5 Overall Architecture]中详细阐述了AlexNet的结构。
        但为了方便使用pytorch提供的预训练模型，SpeedPaper优先使用pytorch官方提供的预训练模型。

    Args:
        num_classes (int): 分类的类别数量，默认为1000

    Attributes:
        features (nn.Sequential): 卷积层和池化层的序列模型
        avgpool (nn.AdaptiveAvgPool2d): 适应性平均池化层
        classifier (nn.Sequential): 全连接层的序列模型

    Methods:
        forward(x): 前向传播函数
    """

    def __init__(self, num_classes=1000):
        """
        AlexNet模型的构造函数

        Args:
            num_classes (int): 分类的类别数量，默认为1000
        """
        super(AlexNet, self).__init__()  # 调用父类（nn.Module）的构造函数
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # alexnet在[3.4 Overlapping Pooling]中的写到其使用的是重叠池化
        # 本次使用自适应均值池化替代
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        前向传播函数

        Args:
            x (torch.Tensor): 输入的张量

        Returns:
            torch.Tensor: 经过模型计算后的张量
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
