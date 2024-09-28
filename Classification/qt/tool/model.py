import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from torch.autograd import Variable


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
def conv3x3x3(in_channel, out_channel, stride=1):
    return nn.Conv3d(in_channel,
                     out_channel,
                     kernel_size=(3, 3, 3),
                     stride=stride,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False)


# 3d残差块
class residual_block_3d(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block_3d, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channel)

        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv3d(in_channel, out_channel, 1, stride=stride)

        # self.max = nn.MaxPool3d(kernel_size = (1,2,2),stride = (1,2,2))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        if not self.same_shape:
            x = self.conv3(x)
            # print("x.shape ",x.shape)
        return F.relu(x + out, True)


# 实现一个 ResNet3d，它就是 residual block 3d模块的堆叠
class resnet3d(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet3d, self).__init__()
        self.verbose = verbose
        # 1*24*40*40
        self.block1 = nn.Conv3d(in_channel, 64, 1, 1)  # 64*20*36*36

        self.block2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 64*20*18*18
            residual_block_3d(64, 64),
            residual_block_3d(64, 64),  # 64*20*18*18
            torch.nn.Dropout(0.5)
        )

        self.block3 = nn.Sequential(
            residual_block_3d(64, 128, False),  # 128*10*9*9
            nn.Conv3d(128, 128, kernel_size=(1, 2, 2), stride=1, padding=0, dilation=1, groups=1, bias=False),
            # 128*10*8*8
            residual_block_3d(128, 128),  # 128*10*8*8
            torch.nn.Dropout(0.5)
        )

        self.block4 = nn.Sequential(
            residual_block_3d(128, 256, False),  # 256*5*4*4
            nn.Conv3d(256, 256, kernel_size=(2, 1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False),
            # 256*4*4*4
            residual_block_3d(256, 256),  # 256*4*4*4
            torch.nn.Dropout(0.5)
        )

        self.block5 = nn.Sequential(
            residual_block_3d(256, 512, False),  # 512*2*2*2
            residual_block_3d(512, 512),  # 512*2*2*2
            nn.AvgPool3d((2, 2, 2), 1),  # 512*1*1*1
            torch.nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = self.sigmoid(x)  # 归一化到 0~1之间
        # print("end: ",x.shape)
        return x


if __name__ == "__main__":
    # 测试1
    # 1.输入输出形状相同
    test_net = residual_block_3d(32, 32)
    test_x = torch.zeros(1, 32, 20, 36, 36)
    print('input: {}'.format(test_x.shape))
    test_y = test_net(test_x)
    print('output: {}'.format(test_y.shape))
    """
    """
    # 2.输入输出形状不同
    test_net = residual_block_3d(64, 128, False)
    test_x = torch.zeros(1, 64, 20, 36, 36)
    print('input: {}'.format(test_x.shape))
    test_y = test_net(test_x)
    print('output: {}'.format(test_y.shape))
    # 测试2
    dummy_input = Variable(torch.rand(8, 1, 20, 36, 36))  # 假设输入13张1*28*28的图片
    model = resnet3d(1, 2, True)
    test = model(dummy_input)
    print("test.shape: ", test.shape)
