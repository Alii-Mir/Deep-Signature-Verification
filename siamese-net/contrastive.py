import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=0.1):   # margin=2.0, 0.5, 0.1, 0.05, 0.01
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        # euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)

        genuine_loss = (1 - label) * torch.pow(euclidean_distance, 2)
        forged_loss = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        separation_loss = 0   # (label) * torch.pow(2*self.margin - euclidean_distance, 2)  #Distance Loss Term

        loss_contrastive = torch.mean(genuine_loss + forged_loss + separation_loss)

        return loss_contrastive

# For genuine pairs (label = 0), the loss is euclidean_distance².
# For forged pairs (label = 1), the loss is (max(0, margin - euclidean_distance))².