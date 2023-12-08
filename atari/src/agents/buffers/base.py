import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from abc import *


class BaseBuffer(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def store(self, obs: np.ndarray, action: int, reward: float, done: bool, next_obs: np.ndarray):
        pass
    
    def sample(self, batch_size: int) -> dict:
        pass

