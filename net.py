import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        if isinstance(kernel_size, tuple):
            padding = (max(kernel_size) - 1) // 2
        else:
            padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        ))
        self.bn_normal = nn.BatchNorm2d(out_planes)
        self.bn_adversial = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, adversial=False):
        x = self.conv(x)
        if adversial:
            x = self.bn_adversial(x)
        else:
            x = self.bn_normal(x)
        x = self.act(x)

        return x

class StandardCNN(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))

        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))

        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv6 = nn.utils.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))

        self.conv7 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dense = nn.utils.spectral_norm(nn.Linear(512 * 4 * 4, 1))

    def forward(self, x):

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        # x = self.pool(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        # x = self.pool(x)
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        # x = self.pool(x)
        x = self.act(self.conv7(x))

        x = self.dense(x.view(x.shape[0], -1))

        return x


class StandardBNCNN(nn.Module):
    def __init__(self):
        super(StandardBNCNN, self).__init__()
        self.conv1 = ConvBNReLU(3, 64)
        self.conv2 = ConvBNReLU(64, 64)
        self.conv3 = ConvBNReLU(64, 128)
        self.conv4 = ConvBNReLU(128, 128)
        self.conv5 = ConvBNReLU(128, 256)
        self.conv6 = ConvBNReLU(256, 256)
        self.conv7 = ConvBNReLU(256, 512)
        self.conv8 = ConvBNReLU(512, 512)
        self.max_pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.utils.spectral_norm(nn.Linear(512, 1))

    def forward(self, x, adversial=False):
        x = self.conv1(x, adversial)
        x = self.conv2(x, adversial)
        x = self.max_pool(x)
        x = self.conv3(x, adversial)
        x = self.conv4(x, adversial)
        x = self.max_pool(x)
        x = self.conv5(x, adversial)
        x = self.conv6(x, adversial)
        x = self.max_pool(x)
        x = self.conv7(x, adversial)
        x = self.conv8(x, adversial)
        # (B, 512, 4, 4)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
