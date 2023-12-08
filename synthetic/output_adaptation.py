'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, random_split

import torchvision
import torchvision.transforms as transforms
from einops import rearrange

from dotmap import DotMap
import argparse
import os
import tqdm
import random
import numpy as np
import wandb
import copy

from utils.SAM import SAM
from utils.metric_utils import get_param_cnt_ratio, AverageMeterSet
from utils.train_utils import CReLU, init_normalization, LocalSignalMixing

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def run(args):
    args = DotMap(args)
    
    wandb.init(project="gpa_cifar10")
    wandb.config.update(args)

    ########################
    ## Setup Configuration
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    ########################
    ## Prepare Dataset
    print('==> Preparing data..')
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
    
    def load_to_gpu(dataset, device):
        data = torch.empty((len(dataset), 3, 32, 32), device=device)
        labels = torch.empty(len(dataset), dtype=torch.long, device=device)

        for i, (image, label) in enumerate(dataset):
            data[i] = image.to(device)
            labels[i] = torch.tensor(label, device=device)

        return torch.utils.data.TensorDataset(data, labels)

    trainset = load_to_gpu(trainset, device)
    testset = load_to_gpu(testset, device)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False)

    ########################################
    ## Prepare Model, Criterion, Optimizer
    class ConvNet(nn.Module):
        def __init__(self, 
                     num_classes, 
                     backbone_norm_type=None, 
                     policy_norm_type=None,
                     backbone_crelu=False,
                     policy_crelu=False):
            
            super(ConvNet, self).__init__()
            self.num_classes = num_classes

            self.backbone_norm_type = backbone_norm_type
            self.policy_norm_type = policy_norm_type
            self.backbone_crelu = backbone_crelu
            self.policy_crelu = policy_crelu
            
            self._init_backbone()
            self._init_policy()
            
        def _init_backbone(self):
            if self.backbone_crelu:
                channels = [(3,16), (32,32), (64,32)] 
            else:
                channels = [(3,32), (32,64), (64,64)]
            layers = []
            for ch in channels:
                if self.backbone_crelu:
                    activation = CReLU()
                else:
                    activation = nn.ReLU()

                layers.extend([
                    nn.Conv2d(in_channels=ch[0], out_channels=ch[1], kernel_size=3, stride=3 if ch == channels[0] else 1),
                    init_normalization(ch[1], norm_type=self.backbone_norm_type),
                    activation
                ])
            self.backbone = nn.Sequential(*layers)
            
        def _init_policy(self):
            if self.policy_crelu:
                channels = [(2304, 384), (768, 192), (384, self.num_classes)]
            else:
                channels = [(2304, 512), (512, 128), (128, self.num_classes)]

            layers = []
            for ch in channels[:-1]:  # Exclude the last channel as there's no activation after the last linear layer
                if self.policy_crelu:
                    activation = CReLU()
                else:
                    activation = nn.ReLU(inplace=False)

                layers.extend([
                    nn.Linear(ch[0], ch[1]),
                    init_normalization(ch[1], norm_type=self.policy_norm_type, one_d=True),
                    activation
                ])

            # Add the last linear layer without activation
            layers.append(nn.Linear(channels[-1][0], channels[-1][1]))

            self.policy = nn.Sequential(*layers)
        
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
            
        def reinit_backbone(self):
            def weight_reset(m):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.reset_parameters()

            self.backbone.apply(weight_reset)
                
        def reinit_policy(self):
            def weight_reset(m):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.reset_parameters()

            self.policy.apply(weight_reset)

        def forward(self, x):
            out = self.backbone(x)
            
            out = torch.flatten(out, 1)
            out = self.policy(out)
            return out

    model = ConvNet(num_classes=10,
                    backbone_norm_type=args.backbone_norm, 
                    policy_norm_type=args.policy_norm,
                    backbone_crelu=args.backbone_crelu,
                    policy_crelu=args.policy_crelu).to(device)
    wandb.watch(model)
    
    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer_type = args.optimizer_type

    if optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    elif optimizer_type == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay, 
            momentum=args.momentum,
            rho=args.rho
        )
    else:
        raise NotImplemented

    ####################
    # Training
    # setup
    num_iters = args.num_iters
    num_chunks = args.num_chunks
    num_iters_per_chunk = num_iters // num_chunks
    
    chunks = random_split(trainset, [len(trainset) // num_chunks] * num_chunks)
    buffer = []
    
    # training loop
    test_acc_list = []
    iter = 0
    for chunk_idx in tqdm.tqdm(range(num_chunks)):
        model.train()
        meter_set = AverageMeterSet()

        done_with_chunk = 0
        while not done_with_chunk:
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = outputs.max(1)
                # label shift
                _targets = (targets + chunk_idx) % 10
                loss = criterion(outputs, _targets)
                loss.backward()
                
                if optimizer_type == 'sgd':
                    optimizer.step()
                    
                elif optimizer_type == 'sam':
                    optimizer.first_step(zero_grad=True)
                    second_loss = criterion(model(inputs), _targets)
                    second_loss.backward()
                    optimizer.second_step(zero_grad=True)

                acc = preds.eq(_targets).float().mean()
                iter = iter + 1
                if iter % num_iters_per_chunk == 0:
                    done_with_chunk = True
                    break
                
                meter_set.update('train_loss', loss.item())
                meter_set.update('train_acc', acc.item())
        
        # evaluation
        logs = {}
        train_logs = meter_set.averages()
        test_logs, layer_wise_outputs = test(test_loader, model, device, chunk_idx)
        test_acc_list.append(test_logs['test_acc'])
        test_logs['test_auc'] = np.mean(test_acc_list)
        
        # if redo, reinitialize for every 10 chunks
        if args.redo:
            new_model = copy.deepcopy(model)
            new_model.reinit_backbone()
            new_model.reinit_policy()
            
            for layer_name, activations in layer_wise_outputs.items():
                with torch.no_grad():
                    _layer_name = layer_name.split('.')[2]
                    _layer_idx = int(layer_name.split('.')[-1])
                    in_weight_idx = _layer_idx - 2
                    out_weight_idx = _layer_idx + 1
                    if (_layer_name == 'ReLU') or (_layer_name == 'CReLU'):
                        abs_activations = torch.abs(activations)
                        score = abs_activations / torch.mean(abs_activations)
                        score = torch.mean(score, 0) # average over batches
                    else:
                        continue
                    
                if 'backbone' in layer_name:
                    for c in range(score.size(0)):
                        for h in range(score.size(1)):
                            for w in range(score.size(2)):
                                if score[c, h, w] <= args.redo_tau:                                    
                                    model.backbone[in_weight_idx].weight[c].data[:] \
                                        = new_model.backbone[in_weight_idx].weight[c].data[:] 
                                    
                                    if out_weight_idx < len(model.backbone):  
                                        model.backbone[out_weight_idx].weight[:, c].data[:] = 0
                                    else:
                                        idx = c * score.size(1) * score.size(2) + h * score.size(2) + w
                                        model.policy[0].weight[:, idx].data[:] = 0
                                        
                
                if 'policy' in layer_name:
                    for d in range(score.size(0)):
                        if score[d] <= args.redo_tau:
                            model.policy[in_weight_idx].weight[d, :].data[:] = new_model.policy[in_weight_idx].weight[d, :].data[:]
                            
                            if out_weight_idx < len(model.policy):  
                                model.policy[out_weight_idx].weight[:, d].data[:] = 0

        # if reset, reset for every 10 chunks
        if chunk_idx % 10 == 0:
            if args.backbone_reset:
                model.reinit_backbone()
            if args.policy_reset:
                model.reinit_policy()
        
        logs.update(train_logs)
        logs.update(test_logs)
        wandb.log(logs, step=iter)

    
def test(loader, model, device, chunk_idx):
    meter_set = AverageMeterSet()
    model.eval()

    # register hook
    global layer_wise_outputs
    layer_wise_outputs = {}
    
    def save_outputs_hook(layer_id):
        def fn(_, __, output):
            layer_wise_outputs[layer_id] = output

        return fn

    def get_all_layers(model, prefix=''):
        for name, layer in model._modules.items():
            layer_id = prefix + '.' + name

            # If the current layer is a Sequential, recursively explore its children
            if isinstance(layer, nn.Sequential):
                for idx, sub_layer in enumerate(layer):
                    sub_layer_id = layer_id + '.' + sub_layer.__class__.__name__ + '.' + str(idx)
                    sub_layer.register_forward_hook(save_outputs_hook(sub_layer_id))
    
    get_all_layers(model, 'net')
    
    # evaluation loop
    criterion = nn.CrossEntropyLoss().to(device)
    for inputs, targets in loader:
        outputs = model(inputs)
        _, preds = outputs.max(1)
        _targets = (targets + chunk_idx) % 10
        loss = criterion(outputs, _targets)
        acc = preds.eq(_targets).float().mean()

        # dead neuron w.r.t gradient
        num_zero_grad_params, num_grad_params = {}, {}
        for layer_name, param in model.named_parameters():
            if param.grad is not None:
                num_zero_grad_params[layer_name] = torch.sum(param.grad == 0).item()
                num_grad_params[layer_name] = param.grad.numel()

        # dead neuron w.r.t activation
        num_zero_activation_params, num_activation_params = {}, {}

        for layer_name, activations in layer_wise_outputs.items():
            num_zero_activation_params[layer_name] = torch.sum(activations == 0).item()
            num_activation_params[layer_name] = activations.numel()

        zero_grad_ratio = get_param_cnt_ratio(
            num_zero_grad_params, num_grad_params, 'zero_grad_ratio')
        zero_activation_ratio = get_param_cnt_ratio(
            num_zero_activation_params, num_activation_params, 'zero_activation_ratio', activation=True)

        logs = {
            'test_loss': loss.item(),
            'test_acc': acc.item()
        }
        
        logs.update(zero_grad_ratio)
        logs.update(zero_activation_ratio)

        for key, value in logs.items():
            meter_set.update(key, value)

    test_logs = meter_set.averages()    
    return test_logs, layer_wise_outputs
    
    
if __name__=='__main__':
    # basic config
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default='test')
    parser.add_argument('--adapt_type',       type=str, default='output')
    parser.add_argument('--checkpoints',      type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--optimizer_type',   type=str, default='sgd', choices=['sgd', 'sam'])
    parser.add_argument('--rho',              type=float, default=0.1)
    parser.add_argument('--redo',             type=str2bool, default=False)
    parser.add_argument('--redo_tau',         type=float, default=0.1)
    parser.add_argument('--backbone_reset',   type=str2bool, default=False)
    parser.add_argument('--policy_reset',     type=str2bool, default=False)
    parser.add_argument('--backbone_norm',    type=str, default=None)
    parser.add_argument('--policy_norm',      type=str, default=None)
    parser.add_argument('--backbone_crelu',   type=str2bool, default=False)
    parser.add_argument('--policy_crelu',     type=str2bool, default=False)
    parser.add_argument('--weight_decay',     type=float, default=0.001)
    parser.add_argument('--batch_size',       type=int, default=128)
    parser.add_argument('--learning_rate',    type=float, default=0.01)
    parser.add_argument('--momentum',         type=float, default=0.9)
    parser.add_argument('--gpu',              type=int, default=0)
    parser.add_argument('--num_iters',        type=int, default=50000)
    parser.add_argument('--num_chunks',       type=int, default=100)
    parser.add_argument('--num_workers',   type=int, default=0)
    parser.add_argument('--seed',          type=int, default=2021)

    args = parser.parse_args()
    run(vars(args))


