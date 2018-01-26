import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardConv(nn.Module):
    def __init__(self, inchan, outchan, kernsize=5, padding=False):
        super(StandardConv, self).__init__()
        self.padding = (kernsize - 1) /2 if padding else 0
        self.conv = nn.Conv2d(inchan, outchan, kernsize, padding=padding)
        self.bn = nn.BatchNorm2d(outchan)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.kern1 = StandardConv(1, 32)
        self.kern2 = StandardConv(32, 64)
        self.kern3 = StandardConv(64, 128)
        self.kern4 = StandardConv(128, 64)
        self.kern4 = StandardConv(64, 32)
        self.kern5 = StandardConv(32, 8)
        self.kerns = nn.Sequential(*[self.kern1, self.kern2, self.kern3,
                                     self.kern4, self.kern5, self.kern5])



    def forward(self, x):
        return x