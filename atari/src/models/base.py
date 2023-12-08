import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, backbone, head, policy):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.policy = policy

    def forward(self, x):
        """
        [backbone]: (n,t,c,h,w)-> Tuple((n,t,d), info)
        [head]: (n,t,d)-> Tuple((n,t,d), info)
        """
        x, b_info = self.backbone(x)
        x, h_info = self.head(x)
        x, p_info = self.policy(x)
        info = {
            'backbone': b_info,
            'head': h_info,
            'policy': p_info
        }
        return x, info
