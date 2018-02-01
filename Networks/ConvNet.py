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
        self.kern1 = StandardConv(1, 32, kernsize=5)
        self.kern2 = StandardConv(32, 64, kernsize=3)
        self.kern3 = StandardConv(64, 128, kernsize=3)
        self.kern9 = StandardConv(128, 256, kernsize=3)
        self.kern0 = StandardConv(256, 128, kernsize=3)
        self.kern4 = StandardConv(128, 64, kernsize=3)
        self.kern5 = StandardConv(64, 32, kernsize=3)
        self.kern6 = StandardConv(32, 16, kernsize=3)
        self.kern8 = StandardConv(16, 6, kernsize=3)
        self.kerns = nn.Sequential(*[self.kern1, self.kern2, self.kern3, self.kern4,
                                     self.kern5, self.kern6, self.kern8])

    def forward(self, x):
        x = F.avg_pool2d(x, 8)
        x = self.initBn(x)
        x = self.kerns(x)
        x = F.avg_pool2d(x, x.data.size()[2]) # B, C, H, W
        x = x.view(-1, 3, 2).contiguous()
        x = x * self.indim
        return x