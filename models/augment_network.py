import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class SingleLayer(nn.Module):
    def __init__(self, inc=32, ouc=3, kernel_size=3, tanh=False, sig=False):
        super(SingleLayer, self).__init__()
        self.conv1 = nn.Conv2d(3,inc,kernel_size,1,kernel_size//2)
        self.conv2 = nn.Conv2d(inc,ouc,kernel_size,1,kernel_size//2)
        self.tanh = tanh
        self.sig = sig
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.tanh:
            x = nn.Tanh()(x)
        if self.sig:
            x = nn.Sigmoid()(x)
            
        return x
