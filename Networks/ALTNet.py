import torch
import torch.nn as nn
import torch.nn.functional as F

class DownTransition(nn.Module):
    def __init__(self, inlayer):
        super(DownTransition, self).__init__()
        self.conv = nn.Conv2d(inlayer, inlayer-1, 5, padding=2)
        self.bn = nn.BatchNorm2d(inlayer - 1)


    def forward(self, x):
        out = []
        # c = self.conv(x)
        # c = self.bn(c)
        # c = F.relu(c)
        c = F.relu(self.bn(self.conv(x)))
        for i in xrange(c.data.size()[1]):
            out.append(x[:,i].unsqueeze(1))
            out.append(c[:,i].unsqueeze(1))
        out.append(x[:,-1].unsqueeze(1))
        out = torch.cat(out, 1)
        return out


class ALTNet(nn.Module):
    def __init__(self, inlayers, upscale=2):
        super(ALTNet, self).__init__()
        # if isinstance(upscale, int):
        #     self.upscale = [upscale, upscale, upscale]
        # elif isinstance(upscale, list) or isinstance(upscale, tuple) or isinstance(upscale, np.ndarray):
        #     self.upscale = list(upscale)
        #
        self.down1 = DownTransition(inlayers)
        self.down2 = DownTransition(inlayers*2 - 1)

    def forward(self, x):
        return self.down2(self.down1(x))