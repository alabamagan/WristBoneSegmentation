import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CNN6(nn.Module):
    def __init__(self, inchan, numOfClass):
        super(CNN6,self).__init__()
        self.initBn = nn.BatchNorm2d(inchan)
        self.kern1 = StandardConv(inchan, 32, kernsize=5, padding=True)
        self.kern2 = StandardConv(32, 64, kernsize=3, padding=True)
        self.kern3 = StandardConv(64, 128, kernsize=3, padding=True)
        self.kern4 = StandardConv(128, 256, kernsize=3, padding=True)
        self.kern5 = StandardConv(256, 512, kernsize=2, padding=True)
        self.fc1 = nn.Linear(512, numOfClass)

    def forward(self, x):
        x = self.initBn(x)
        x = F.max_pool2d(self.kern1(x), 2)
        x = F.max_pool2d(self.kern2(x), 2)
        x = F.max_pool2d(self.kern3(x), 2)
        x = F.max_pool2d(self.kern4(x), 2)
        x = F.max_pool2d(self.kern5(x), 2)
        x = self.fc1(x.squeeze())
        return x

class WNet(nn.Module):
    def __init__(self, numOfClass):
        super(WNet, self).__init__()
        self.numOfClass = numOfClass
        self.initBn = nn.BatchNorm2d(2)
        self.cnn1 = CNN6(1, numOfClass)
        self.cnn2 = CNN6(1, numOfClass)
        self.cnn3 = CNN6(2, numOfClass)
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.initBn(x)
        x1 = x[:,0].unsqueeze(1)
        x2 = x[:,1].unsqueeze(1)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x)

        x = self.fc1(torch.cat([x1, x2], -1))
        x = self.fc2(torch.cat([x3, x], -1))
        x = F.log_softmax(x)
        return x