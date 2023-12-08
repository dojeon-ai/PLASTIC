import torch.nn as nn
import torch
from einops import rearrange
from src.models.backbones.base import BaseBackbone
from src.common.train_utils import orthogonal_init, renormalize, CReLU, LayerNorm


class Nature(BaseBackbone):
    name = 'nature'
    def __init__(self,
                 obs_shape,
                 action_size,
                 init_type,
                 renormalize,
                 activation,
                 normalization):
        super().__init__()
        self.obs_shape = obs_shape
        f, c, h, w = obs_shape
        in_channels = f * c

        if activation == 'ReLU':
            if 'layernorm' in normalization:
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
                    LayerNorm(32, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
                    LayerNorm(64, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
                    LayerNorm(64, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(),
                    nn.Flatten()
                )
            else:
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
                    nn.ReLU(),
                    nn.Flatten()
                )

        elif 'CReLU' in activation:
            if 'input' in activation:
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
                    CReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2), 
                    CReLU(),
                    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), 
                    CReLU(),
                    nn.Flatten()
                )
            elif 'output' in activation:
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4), 
                    CReLU(),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2), 
                    CReLU(),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1), 
                    CReLU(),
                    nn.Flatten()
                )

        if init_type == 'orthogonal':
            self.apply(orthogonal_init)
        self.renormalize = renormalize
            
    def forward(self, x):
        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> (n t) (f c) h w')
        x = self.layers(x)
        if self.renormalize:
            x = renormalize(x)
        x = rearrange(x, '(n t) d -> n t d', t=t)
        info = {}
            
        return x, info