import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils import spectral_norm

class NNBlock(Module):
    def __init__(self, fin, fout):
        super().__init__()
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_0 = spectral_norm(self.conv_0)
        self.norm_0 = nn.BatchNorm2d(fin, affine=True)

    def forward(self, x):
        dx = self.conv_0(self.actvn(self.norm_0(x)))
        return dx

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)