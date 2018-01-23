import torch
import torch.nn as nn
import torch.nn.functional as F

class DownTransition(nn.Module):
    def __init__(self, upslice):
        super(DownTransition, self).__init__()
        self.conv0 = nn.Conv3d(1, 32, kernel_size=(2, 5, 5), padding=(0, 2, 2))
        self.conv1 = nn.Conv3d(32, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.conv2 = nn.Conv3d(64, upslice, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn0 = nn.BatchNorm3d(32)
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(upslice)
        self.alpha = nn.Parameter(data=torch.FloatTensor([1.]))



    def forward(self, x):
        out = []
        c = F.relu(self.bn0(self.conv0(x)))
        c = F.relu(self.bn1(self.conv1(c)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = c * self.alpha.expand_as(c)
        for i in xrange(x.data.size()[2] - 1):
            out.append(x[:,:,i])
            for j in xrange(c.data.size()[1]):
                out.append(c[:,j,i].unsqueeze(1))
        out.append(x[:,:,-1])
        out = torch.cat(out, 1)
        return out



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.down1 = DownTransition(3)

    def forward(self, x):
        return self.down1(x)