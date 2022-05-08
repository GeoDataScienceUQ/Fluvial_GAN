import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from .nnblk import NNBlock

class Generator(Module):
    def __init__(self, args):
        super().__init__()
        self.nf = args.g_num_filter
        self.img_size = args.gen_input_size
        self.sw, self.sh = self.compute_latent_vector_size(5)#8

        self.fc = nn.Linear(args.gen_hidden_size, 16 * self.nf * self.sw * self.sh)

        self.head_0 = NNBlock(16 * self.nf, 16 * self.nf)
        self.G_middle_0 = NNBlock(16 * self.nf, 16 * self.nf)
        self.up_0 = NNBlock(16 * self.nf, 8 * self.nf)
        self.up_1 = NNBlock(8 * self.nf, 4 * self.nf)
        self.up_2 = NNBlock(4 * self.nf, 2 * self.nf)
        self.up_3 = NNBlock(2 * self.nf, self.nf)
        self.conv_img = nn.Conv2d(self.nf, 7 * args.img_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, num_up_layers):
        sw = self.img_size // (2**num_up_layers)
        sh = round(sw / 1)

        return sw, sh

    def forward(self, noise):

        x = self.fc(noise)
        x = x.view(-1, 16 * self.nf, self.sh, self.sw)#8,8

        x = self.head_0(x)

        x = self.up(x)#16,16
        x = self.G_middle_0(x)

        x = self.up(x)#32,32
        x = self.up_0(x)
        x = self.up(x)#64,64
        x = self.up_1(x)
        x = self.up(x)#128,128
        x = self.up_2(x)
        
        x = self.up(x)#256,256
        x = self.up_3(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.softmax(x,dim=1)

        return x