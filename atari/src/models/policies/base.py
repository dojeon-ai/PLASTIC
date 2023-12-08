from abc import *
import torch.nn as nn
import torch


class BasePolicy(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def reset_parameters(self, **kwargs):
        for name, layer in self.named_children():
            modules = [m for m in layer.children()]
            for module in modules:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()