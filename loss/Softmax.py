import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self, config):
        super(Softmax, self).__init__()
        self.temp = config.temp

    def forward(self, x, logits, labels=None):

        logits = F.softmax(logits)
        if labels is None: return logits, 0
        loss = F.cross_entropy(logits / self.temp, labels)
        return logits, loss
