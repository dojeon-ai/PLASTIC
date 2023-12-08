import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from .base import BasePolicy
from src.common.train_utils import ScaleGrad
from src.common.train_utils import CReLU


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input,
                            self.weight_mu, 
                            self.bias_mu)


class RainbowCReLUPolicy(BasePolicy):
    name = 'rainbow_crelu'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 action_size,
                 num_atoms,
                 noisy_std,
                 duel,
                 width):
        super().__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.duel = duel
        hid_dim = hid_dim * width

        self.fc_v = nn.Sequential(
            NoisyLinear(in_dim, hid_dim, noisy_std),
            CReLU(),
            NoisyLinear(hid_dim*2, num_atoms, noisy_std)
        )
        self.fc_adv = nn.Sequential(
            NoisyLinear(in_dim, hid_dim, noisy_std),
            CReLU(),
            NoisyLinear(hid_dim*2, action_size * num_atoms, noisy_std)
        )
        self.grad_scale = 2 ** (-1 / 2)


    def forward(self, x, log=False):
        if self.duel:
            scale_grad = getattr(ScaleGrad, "apply", None)
            x = scale_grad(x, self.grad_scale)
            v = self.fc_v(x)
            adv = self.fc_adv(x)

            v = v.view(-1, 1, self.num_atoms)
            adv = adv.view(-1, self.action_size, self.num_atoms)
            
            # numerical stability for dueling
            q = v + adv - adv.mean(1, keepdim=True)
        else:
            q = self.fc_adv(x)
            q = q.view(-1, self.action_size, self.num_atoms)

        # (batch_size, action_size, num_atoms)
        log_q = F.log_softmax(q, -1)
        q = torch.exp(log_q)
        
        info = {'log': log_q}
        
        return q, info

    def reset_noise(self):
        for name, layer in self.named_children():
            modules = [m for m in layer.children()]
            for module in modules:
                if hasattr(module, 'reset_noise'):
                    module.reset_noise()

    def get_num_atoms(self):
        return self.num_atoms