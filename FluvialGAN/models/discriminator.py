import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def get_nonspade_norm_layer(norm_type='spectralinstance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)
    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
            
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

class HybridDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        subnetD = self.create_single_discriminator(args)
        self.add_module('discriminator_0', subnetD)

        subnetDD = self.create_single_dilateddiscriminator(args)
        self.add_module('discriminator_1', subnetDD)

    def create_single_discriminator(self, args):
        netD = NLayerDiscriminator(args)
        return netD

    def create_single_dilateddiscriminator(self, args):
        netD = NLayerDilatedDiscriminator(args)
        return netD

    def forward(self, input):
        result = []
        for name, D in self.named_children():
            out = D(input)
            out = [out]
            result.append(out)

        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = args.d_num_filter
        n_first = nf
        input_nc = 7*args.img_nc + 3 # 10 = 6+2+2
        norm_layer = get_nonspade_norm_layer('spectralinstance')

        sequence = [[nn.Conv2d(input_nc, n_first, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]#129,129

        nf_prev = n_first
        for n in range(1, args.d_num_layer):
            nf = min(nf * 2, 1024)#66,66 34,34 35,35 / 66,66 67,67 / 130,130
            stride = 1 if n == args.d_num_layer - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]
            nf_prev = nf

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]] #35,35

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        return results[-1]

class NLayerDilatedDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = args.d_num_filter
        n_first = nf
        input_nc = 7*args.img_nc + 3 # 10 = 6+2+2
        norm_layer = get_nonspade_norm_layer('spectralinstance')

        sequence = [[nn.Conv2d(input_nc, n_first, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]#129,129

        nf_prev = n_first
        for n in range(1, args.d_num_layer):
            nf = min(nf * 2, 1024)#66,66 34,34 35,35 / 66,66 67,67 / 130,130
            stride = 1 if n == args.d_num_layer - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw,dilation=2)),
                          nn.LeakyReLU(0.2, False)
                          ]]
            nf_prev = nf

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]] #35,35

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        return results[-1]