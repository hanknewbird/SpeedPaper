import torch
import torch.nn as nn

# 这是没加辅助损失的googlenetv1


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

        self.output = nn.Conv2d(ch1x1 + ch3x3 + ch5x5 + pool_proj, 1, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return self.output(torch.cat(outputs, 1))


class GoogLeNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNetV1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Inception(192, 64, 96, 128, 16, 32, 3),
            Inception(256, 128, 128, 192, 32, 96, 3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Inception(480, 192, 96, 208, 16, 48, 3),
            Inception(512, 160, 112, 224, 24, 64, 3),
            Inception(512, 160, 112, 224, 24, 64, 3),
            Inception(512, 160, 112, 224, 24, 64, 3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )
        self.classifier = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
