import torch.nn as nn
import torch.nn.functional as F
from .utils import ResBlock


class resnet_34(nn.Module):
    def __init__(self, in_channel=64, num_classes=10):
        super(resnet_34, self).__init__()
        self.in_channel = in_channel
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # 3->64
        # 1 layer with 2 ResBlock
        self.layer1 = self.make_layer(ResBlock, in_channel, 3, stride=1)
        self.layer2 = self.make_layer(ResBlock, in_channel * 2, 4, stride=2)
        self.layer3 = self.make_layer(ResBlock, in_channel * 4, 6, stride=2)
        self.layer4 = self.make_layer(ResBlock, in_channel * 8, 3, stride=2)
        self.fc = nn.Linear(in_channel * 8, num_classes)

    def make_layer(self, block, out_channel, num_blocks, stride):
        # pass 1 layer, channels * 2, features downsample
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_res34():
    return resnet_34()
