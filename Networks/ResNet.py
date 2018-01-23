import torch
import torch.nn as nn
import torch.nn.functional as F

class Kernel(nn.Module):
    def __init__(self, inchan, outchan, kernsize=5):
        super(Kernel, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernsize, padding=(kernsize - 1)/2 )
        self.bn = nn.BatchNorm2d(outchan)

    def forward(self, x):
        # x = self.conv(x)
        # x = self.bn(x)
        # x = F.relu(x)
        x = F.relu(self.bn(self.conv(x)))
        return x

class ResKernel(nn.Module):
    def __init__(self, chan):
        super(ResKernel, self).__init__()
        self.k1 = Kernel(chan, chan)
        self.k2 = Kernel(chan, chan)

    def forward(self, x):
        c = self.k1(x)
        c = self.k2(c)
        c = c + x
        return c

class ResNet(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(ResNet, self).__init__()
        self.depth = depth
        self.initkern = Kernel(inchan, inchan)
        self.kerns = nn.Sequential(*[ResKernel(inchan) for i in xrange((depth - 1)/2)])
        self.outkern = Kernel(outchan, outchan)

    def forward(self, x):
        c = self.initkern(x)
        c = c + x
        c = self.kerns.forward(c)
        c = self.outkern(c)
        c = c + x
        return c