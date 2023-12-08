import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


def init_normalization(channels, norm_type="ln", one_d=False):
    assert norm_type in ["bn", "ln",None]
    if norm_type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=True, momentum=0.01)
        else:
            return nn.BatchNorm2d(channels, affine=True, momentum=0.01)
        
    elif norm_type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=True)
        else:
            return nn.GroupNorm(1, channels, affine=True)
    
    elif norm_type is None:
        return nn.Identity()
    
    
class LocalSignalMixing(nn.Module):
    def __init__(self, pad, fixed_batch=False):
        """LIX regularization layer

        pad : float
            maximum regularization shift (maximum S)
        fixed batch : bool
            compute independent regularization for each sample (slower)
        """
        super().__init__()
        # +1 to avoid that the sampled values at the borders get smoothed with 0
        self.pad = int(math.ceil(pad)) + 1
        self.base_normalization_ratio = (2 * pad + 1) / (2 * self.pad + 1)
        self.fixed_batch = fixed_batch

    def get_random_shift(self, n, c, h, w, x):
        if self.fixed_batch:
            return torch.rand(size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        else:
            return torch.rand(size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)

    def forward(self, x, max_normalized_shift=1.0):
        """
        x : Tensor
            input features
        max_normalized_shift : float
            current regularization shift in relative terms (current S)
        """
        if self.training:
            max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
            n, c, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, 'replicate')
            arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
            shift = self.get_random_shift(n, c, h, w, x)
            shift_offset = (1 - max_normalized_shift) / 2
            shift = (shift * max_normalized_shift) + shift_offset
            shift *= (2 * self.pad + 1)  # can start up to idx 2*pad + 1 - ignoring the left pad
            grid = base_grid + shift
            # normalize in [-1, 1]
            grid = grid * 2.0 / (h + 2 * self.pad) - 1
            return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        else:
            return x
