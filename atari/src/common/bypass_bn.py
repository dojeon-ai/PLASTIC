## ref: https://github.com/davda54/sam/issues/17

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from ..models.policies.rainbow_policy import NoisyLinear


def disable_running_stats(model, noise=False):
    """
    :: Args
        - noise : if False, turn off the noise
    """
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
        if isinstance(module, NoisyLinear):
            if not noise:
                module.backup_weight_epsilon = module.weight_epsilon
                module.backup_bias_epsilon = module.bias_epsilon
                module.weight_epsilon = torch.zeros_like(module.weight_epsilon)
                module.bias_epsilon = torch.zeros_like(module.bias_epsilon)

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
        if isinstance(module, NoisyLinear):
            if hasattr(module, 'backup_weight_epsilon'):
                module.weight_epsilon = module.backup_weight_epsilon
            if hasattr(module, 'backup_bias_epsilon'):
                module.bias_epsilon = module.backup_bias_epsilon


    model.apply(_enable)