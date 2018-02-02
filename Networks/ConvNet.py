import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StandardConv(nn.Module):
    def __init__(self, inchan, outchan, kernsize=5, padding=False):
        super(StandardConv, self).__init__()
        if padding:
            self.conv = nn.Conv2d(inchan, outchan, kernsize, padding=(kernsize - 1) /2)
        else:
            self.conv = nn.Conv2d(inchan, outchan, kernsize)
        self.bn = nn.BatchNorm2d(outchan)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# class ConvNet(nn.Module):
#     def __init__(self, indim):
#         super(ConvNet,self).__init__()
#         self.indim = indim
#         self.kern1 = StandardConv(1, 32)
#         self.kern2 = StandardConv(32, 64)
#         self.kern3 = StandardConv(64, 128)
#         self.kern4 = StandardConv(128, 256)
#         self.kern5 = StandardConv(256, 128)
#         self.kern6 = StandardConv(128, 64)
#         self.kern7 = StandardConv(64, 32)
#         self.kern8 = nn.Conv2d(32, 8, 5)
#         self.linear = nn.Linear(2, 2, bias=True)
#         self.linear.bias.data.fill_(float(indim)/2.)
#
#     def forward(self, x):
#         x = F.avg_pool2d(x, 2)
#         x = self.kern2(self.kern1(x))
#         x = F.max_pool2d(x, 2)
#         x = self.kern4(self.kern3(x))
#         x = F.max_pool2d(x, 2)
#         x = self.kern8(self.kern7(self.kern6(self.kern5(x))))
#         x = F.max_pool2d(x, x.data.size()[2])
#         x = x.view(-1, 2)
#         x = self.linear(x)
#         x = x.view(-1, 4, 2)
#         return x


class ConvNet(nn.Module):
    def __init__(self, indim):
        super(ConvNet,self).__init__()
        self.indim = indim
        self.initBn = nn.BatchNorm2d(1)
        self.kern1 = StandardConv(1, 32, kernsize=5, padding=True)
        self.kern2 = StandardConv(32, 64, kernsize=3, padding=True)
        self.kern3 = StandardConv(64, 128, kernsize=3, padding=True)
        self.kern4 = StandardConv(128, 256, kernsize=3, padding=True)
        self.kern5 = StandardConv(256, 512, kernsize=2, padding=True)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        x = F.avg_pool2d(x, 8)
        x = self.initBn(x)
        x = F.max_pool2d(self.kern1(x), 2)
        x = F.max_pool2d(self.kern2(x), 2)
        x = F.max_pool2d(self.kern3(x), 2)
        x = F.max_pool2d(self.kern4(x), 2)
        x = F.max_pool2d(self.kern5(x), 2)
        x = self.fc2(self.fc1(x.squeeze()))
        # x = F.avg_pool2d(x, x.data.size()[2]) # B, C, H, W
        x = x.view(-1, 3, 2)
        # x = x
        return x