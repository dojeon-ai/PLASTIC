import numpy as np
import math
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.optim.lr_scheduler import _LRScheduler
from einops import rearrange


def set_global_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

######################
# architecture
class CReLU(nn.Module):
    def __init__(self, normalization = None):
        super(CReLU, self).__init__()

    def forward(self, x):
        if len(x.shape) > 3:
            out_dim = 1
        else:
            out_dim = 2
        return torch.cat((nn.ReLU()(x), nn.ReLU()(-x)), dim=out_dim)

    
######################
# init & norm
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        gain = 1.0
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0)
    
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0)
        
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)    
        nn.init.constant_(m.bias, 0)
    
    return m

def transformer_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
        
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)    
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.Parameter):
        nn.init.normal_(m, std=.02)
    
    return m

def renormalize(tensor, first_dim=1):
    # [params] first_dim: starting dimension to normalize the embedding
    eps = 1e-6
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    _max = torch.max(flat_tensor, first_dim, keepdim=True).values
    _min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - _min)/(_max - _min + eps)

    return flat_tensor.view(*tensor.shape)

def init_normalization(channels, norm_type="bn", one_d=False):
    assert norm_type in ["bn", "bn_nt", "ln", "ln_nt", None]
    if norm_type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=True, momentum=0.01)
        else:
            return nn.BatchNorm2d(channels, affine=True, momentum=0.01)
        
    elif norm_type == "bn_nt":
        if one_d:
            return nn.BatchNorm1d(channels, affine=False, momentum=0.01)
        else:
            return nn.BatchNorm2d(channels, affine=False, momentum=0.01)
        
    elif norm_type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=True)
        else:
            return nn.GroupNorm(1, channels, affine=True)
    
    elif norm_type == "ln_nt":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=False)
        else:
            return nn.GroupNorm(1, channels, affine=False)
    
    elif norm_type is None:
        return nn.Identity()

######################
# scheduler
class LinearScheduler(object):
    # linear scheduler to control epsilon rate
    def __init__(self, initial_value, final_value, step_size):
        """
        Linear Interpolation between initial_value to the final_value
        [params] initial_value (float) initial output value
        [params] final_value (float) final output value
        [params] step_size (int) number of timesteps to lineary anneal initial value to the final value
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        self.step_size = step_size
        self.step = 0
        
    def get_value(self):
        """
        Return the scheduled value
        """
        self.step += 1
        interval = (self.initial_value - self.final_value) / self.step_size
        # After the schedule_timesteps, final value is returned
        if self.final_value < self.initial_value:
            return max(self.initial_value - interval * self.step, self.final_value)
        else:
            return min(self.initial_value - interval * self.step, self.final_value)
        

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_ratio : float = 0.2,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = int(warmup_ratio * first_cycle_steps) # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
    
########################
# optimization
def get_grad_norm_stats(model):
    grad_norm = []
    stats = {}
    for p in model.parameters():
        if p.grad is not None:
            grad_norm.append(p.grad.detach().data.norm(2))
    grad_norm = torch.stack(grad_norm)
    stats['min_grad_norm'] = torch.min(grad_norm).item()
    stats['mean_grad_norm'] = torch.mean(grad_norm).item()
    stats['max_grad_norm'] = torch.max(grad_norm).item()

    return stats

class ScaleGrad(torch.autograd.Function):
    """Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""

    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of 
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None
    


class CReLU(nn.Module):
    def __init__(self, normalization = None):
        super(CReLU, self).__init__()

    def forward(self, x):
        if len(x.shape) > 3:
            out_dim = 1
        else:
            out_dim = 2
        return torch.cat((nn.ReLU()(x), nn.ReLU()(-x)), dim=out_dim)
    
## From ConvNeXt
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x