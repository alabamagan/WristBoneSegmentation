import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualInformationLoss(nn.Module):
    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def forward(self, x, y):
        matched = torch.pow(F.pairwise_distance(x, y), 2)
        mismatched = torch.pow(F.pairwise_distance())