from abc import *
import torch.nn as nn
from .base import BaseHead


class IdentityHead(BaseHead):
    name = 'identity'
    def __init__(self, in_dim):
        super().__init__()
        
    def forward(self, x):
        info = {}
        return x, info
