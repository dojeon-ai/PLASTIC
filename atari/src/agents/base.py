import torch
import torch.nn as nn
import torch.optim as optim

import tqdm
import copy

import numpy as np
import wandb
from abc import *
import matplotlib.pyplot as plt
from typing import Tuple
from src.common.hessian import Hessian
from src.common.train_utils import LinearScheduler
from src.common.eval_utils import get_param_cnt_ratio
from .optimizer.sam import SAM
from src.common.bypass_bn import enable_running_stats, disable_running_stats

class BaseAgent(metaclass=ABCMeta):
    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 logger, 
                 buffer,
                 aug_func,
                 model):
        
        super().__init__()  
        self.cfg = cfg  
        self.device = device
        self.train_env = train_env
        self.eval_env = eval_env
        self.logger = logger
        self.buffer = buffer
        self.aug_func = aug_func.to(self.device)
        self.model = model.to(self.device)
        self.optimizer_type = cfg.optimizer['type']
        self.optimizer = self._build_optimizer(self.model.parameters(), cfg.optimizer)
        self.eps_scheduler = LinearScheduler(**cfg.eps_scheduler)
        self.noise_scheduler = LinearScheduler(**cfg.noise_scheduler)

    @classmethod
    def get_name(cls):
        return cls.name

    def _build_optimizer(self, param_group, optimizer_cfg):
        if 'type' in optimizer_cfg:
            optimizer_type = optimizer_cfg.pop('type')

        if optimizer_type == 'adam':
            optimizer = optim.Adam(param_group, 
                              **optimizer_cfg)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(param_group, 
                              **optimizer_cfg)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(param_group, 
                                 **optimizer_cfg)
        elif optimizer_type == 'sam':
            self.sam_backbone = True if 'sam_backbone' not in optimizer_cfg else optimizer_cfg.pop('sam_backbone')
            self.sam_head = True if 'sam_head' not in optimizer_cfg else optimizer_cfg.pop('sam_head')
            optimizer = SAM(param_group, **optimizer_cfg)
        else:
            raise ValueError
        
        optimizer_cfg['type'] = optimizer_type
        return optimizer
    
    @abstractmethod
    def predict(self, obs, eps) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [output] loss
        [output] pred: prediction Q-value
        [output] target: target Q-value
        """
        pass

    def _train_loss_and_regularizers(self,
                                     batch,
                                     optimize_step,
                                     online_model,
                                     target_model,
                                     reset_noise=True) -> Tuple[torch.Tensor, ...]:
        """
        [output] rl_loss
        """
        rl_loss, preds, targets = self.forward(online_model, target_model, batch, mode='train', reset_noise=reset_noise)

        return rl_loss
    
    def train(self):
        optimize_step = 1
        eps = 1.0
        exp_noise = 1.0

        obs = self.train_env.reset()
        self.initial_model = copy.deepcopy(self.model)
        online_model = self.model
        target_model = self.target_model

        if self.cfg.exploration_model == 'online':
            exploration_model = self.model
        else:
            exploration_model = self.target_model        

        for env_step in tqdm.tqdm(range(1, self.cfg.num_timesteps+1)):
            ####################
            # collect trajectory
            exploration_model.train()              
            obs_tensor = self.buffer.encode_obs(obs, prediction=True)
            action = self.predict(exploration_model, obs_tensor, eps, exp_noise)
            next_obs, reward, done, info = self.train_env.step(action)
            self.buffer.store(obs, action, reward, done, next_obs)
            self.logger.step(obs, reward, done, info, mode='train')

            if info.traj_done:
                obs = self.train_env.reset()
            else:
                obs = next_obs

            if env_step >= self.cfg.min_buffer_size:
                ################
                # train
                eps = self.eps_scheduler.get_value()
                exp_noise = self.noise_scheduler.get_value()

                if self.cfg.train_online_mode == 'train':
                    online_model.train()
                else:
                    online_model.eval()

                if self.cfg.train_target_mode == 'train':
                    target_model.train()
                else:
                    target_model.eval()

                # optimize
                for _ in range(self.cfg.optimize_per_env_step):
                    batch = self.buffer.sample(self.cfg.batch_size, mode='train')
                    
                    # SAM optimizer requires a closure that independently computes loss.
                    closure = None
                    if self.optimizer_type in ['sam']:
                        # sharpness-aware minimization is computation-heavy; do less if you can.
                        step_per_sam = self.cfg.get('sam_update_period', 1)
                        noise_in_perturbation = self.cfg.get('noise_in_sam_perturbation', True)  # if True, it turns on the noise in noisy layer
                        reuse_noise = self.cfg.get('sam_reuse_noise_per_opt', False)

                        if (optimize_step % step_per_sam == 0) and (step_per_sam != -1):
                            # perturbation step
                            disable_running_stats(online_model, noise=noise_in_perturbation)
                            disable_running_stats(target_model, noise=noise_in_perturbation)
                            # update step with closure
                            def closure():
                                enable_running_stats(online_model)
                                enable_running_stats(target_model)
                                if not self.sam_backbone:
                                    for param in online_model.backbone.parameters():
                                        param.requires_grad = True
                                        param.grad = param.grad_backup
                                if not self.sam_head:
                                    for param in online_model.policy.parameters():
                                        param.requires_grad = True
                                        param.grad = param.grad_backup
                                loss_and_regs = self._train_loss_and_regularizers(batch, optimize_step, online_model,
                                                                                  target_model, reset_noise=not reuse_noise)
                                rl_loss = loss_and_regs
                                loss = rl_loss 

                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    self.cfg.clip_grad_norm
                                )
                                return loss
                    
                    # compute loss, regularizer, and determine whether or not reset.
                    # if we do SAM update, postpone the reset.
                    loss_and_regs = self._train_loss_and_regularizers(batch,
                                                                      optimize_step,
                                                                      online_model,
                                                                      target_model,
                                                                      reset_noise=closure is None or noise_in_perturbation)
                    rl_loss = loss_and_regs
                    
                    # optimization
                    loss = rl_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    param_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.cfg.clip_grad_norm
                    )
                    if self.optimizer_type == 'sam':
                        if not self.sam_backbone and not self.sam_head:
                            self.optimizer.base_optimizer.step()
                        else:
                            if not self.sam_backbone:
                                for param in online_model.backbone.parameters():
                                    self.optimizer.state[param]["old_p"] = param.data.clone()
                                    param.grad_backup = param.grad.data.clone()
                                    param.requires_grad = False
                            if not self.sam_head:
                                for param in online_model.policy.parameters():
                                    self.optimizer.state[param]["old_p"] = param.data.clone()
                                    param.grad_backup = param.grad.data.clone()
                                    param.requires_grad = False
                            self.optimizer.step(closure)  # apply closure
                    else:
                        self.optimizer.step(closure)  # apply closure

                    train_logs = {
                        'eps': eps,
                        'exp_noise': exp_noise,
                        'loss': loss.item(),
                        'rl_loss': rl_loss.item(),  
                        'param_grad_norm': param_grad_norm.item(),                   
                    }
                    if self.optimizer_type == 'sam':
                        if (optimize_step % step_per_sam == 0) and (step_per_sam != -1) and (self.sam_backbone or self.sam_head):
                            train_logs['sam_first_grad_norm'] = self.optimizer.first_grad_norm.item()
                            train_logs['sam_second_grad_norm'] = self.optimizer.second_grad_norm.item()

                    # target_update
                    # if you want to jointly update the buffer information (e.g., running stats from bn),
                    # you should update the state_dict not parameters
                    for online, target in zip(online_model.parameters(), target_model.parameters()):
                        target.data = self.cfg.target_tau * target.data + (1 - self.cfg.target_tau) * online.data

                    if self.cfg.update_state_dict:
                        for online, target in zip(online_model.buffers(), target_model.buffers()):
                            target.data = self.cfg.target_tau * target.data + (1 - self.cfg.target_tau) * online.data

                    # reset
                    # 1. decide reset model (original vs random) 
                    # 2. decide which parameters to keep (last vs weight vs gradient)
                    # 3. reset parameters (opt) reset target
                    reset_per_step = self.cfg.reset_per_optimize_step
                    reset_type = self.cfg.reset_type
                    reset_weight_type = self.cfg.reset_weight_type
                    reset_target = self.cfg.reset_target
                    shrink_perturb = self.cfg.shrink_perturb
                    match_noise = self.cfg.match_noise

                    if ((optimize_step % reset_per_step == 0) 
                        and (reset_per_step != -1) 
                        and (env_step != self.cfg.num_timesteps)):

                        if reset_weight_type == 'random':
                            reset_model = copy.deepcopy(online_model)
                            reset_model.backbone.reset_parameters()
                            reset_model.policy.reset_parameters()

                        elif reset_weight_type == 'original':
                            reset_model = copy.deepcopy(self.initial_model)

                        # keep_mask_dict: {key: param_mask} 
                        # this dictionary indicates which parameters to keep (1: keep, 0: reset)
                        keep_mask_dict = {}

                        # reset policy layers as a primacy bias reset
                        # https://arxiv.org/abs/2205.07802
                        if reset_type == 'llf':
                            for key, param in online_model.named_parameters():
                                if 'policy' in key:
                                    keep_mask_dict[key] = torch.zeros_like(param)
                                else:
                                    if shrink_perturb:
                                        sp_alpha = self.cfg.shrink_perturb_alpha
                                        keep_mask_dict[key] = torch.ones_like(param) * sp_alpha
                                    else:
                                        keep_mask_dict[key] = torch.ones_like(param)

                        # reset all layers 
                        if reset_type == 'all':
                            for key, param in online_model.named_parameters():
                                keep_mask_dict[key] = torch.zeros_like(param)

                        num_reset_params, num_params = {}, {}
                        for (key, online), (_, reset) in zip(
                            online_model.named_parameters(), reset_model.named_parameters()):

                            if key in keep_mask_dict:
                                keep_mask = keep_mask_dict[key].float()
                                online.data = keep_mask * online.data + (1-keep_mask) * reset.data

                                num_params[key] = online.numel()
                                num_reset_params[key] = online.numel() - torch.sum(keep_mask).item()

                        if reset_target:
                            for (key, target), (_, reset) in zip(
                                target_model.named_parameters(), reset_model.named_parameters()):

                                if key in keep_mask_dict:
                                    keep_mask = keep_mask_dict[key].float()
                                    target.data = keep_mask * target.data + (1-keep_mask) * reset.data

                        reset_ratio = get_param_cnt_ratio(num_reset_params, num_params, 'reset_ratio')
                        train_logs.update(reset_ratio)
                        
                        # reinitialize the optimizer at reset
                        self.optimizer = self._build_optimizer(online_model.parameters(), self.cfg.optimizer)

                        if self.cfg.pd_per_optimize_step != -1:
                            self.initial_model.policy = copy.deepcopy(online_model.policy)

                    self.logger.update_log(mode='train', **train_logs)
                    optimize_step += 1

                ################
                # evaluate
                online_model.eval()
                target_model.eval()
                
                # evaluate
                if (env_step % self.cfg.evaluate_freq == 0) and (self.cfg.evaluate_freq != -1):
                    eval_logs = self.evaluate()
                    self.logger.update_log(mode='eval', **eval_logs)

                if (env_step % self.cfg.rollout_freq == 0) and (self.cfg.rollout_freq != -1):
                    self.rollout()

                ################
                # log
                if env_step % self.cfg.log_freq == 0:
                    self.logger.write_log(mode='train')
                    self.logger.write_log(mode='eval')

    def evaluate(self):
        EPS = 1e-7

        ######################
        # Forward
        online_model = self.model
        target_model = self.target_model

        # get output of each sublayer with hook
        layer_wise_outputs = {}
        def save_outputs_hook(layer_id):
            def fn(_, __, output):
                layer_wise_outputs[layer_id] = output
            return fn

        def get_all_layers(net, prefix=''):
            for name, layer in net._modules.items():
                if isinstance(layer, nn.Sequential):
                    for layer_idx, sub_layer in enumerate(layer):
                        sub_layer.register_forward_hook(
                            save_outputs_hook(
                                prefix + '.' + name + '.' 
                                + sub_layer.__class__.__name__ + '.' + str(layer_idx)
                            )
                        )
                else:
                    get_all_layers(layer, prefix)

        get_all_layers(online_model.backbone, 'backbone')
        get_all_layers(online_model.policy, 'policy')

        # forward
        batch = self.buffer.sample(self.cfg.batch_size, mode='eval')
        batch['obs'].requires_grad=True
        rl_loss, preds, targets = self.forward(online_model, target_model, batch, mode='eval')

        # explained variance    
        pred_var = torch.var(preds)
        target_var = torch.var(targets)
        diff_var = torch.var(preds - targets)
        exp_var = 1 - diff_var / (target_var + EPS)

        ##########################
        # Smoothness
        # smoothness of prediction
        # hessian analysis    
        # maximum eigenvalue for the hessian

        eval_eigen = self.cfg.eval_eigen
        if eval_eigen:
            keys, params, grads = [], [], []
            self.optimizer.zero_grad()
            for key, param in online_model.named_parameters():
                grad = torch.autograd.grad(
                    outputs=rl_loss, inputs=param, 
                    create_graph=True, retain_graph=True, allow_unused=True
                )[0]
                if grad is not None:
                    keys.append(key)
                    params.append(param)
                    grads.append(grad)
                
            hessian = Hessian(online_model, keys, params, grads, self.device)
            eigenvalues, eigenvectors = hessian.get_topk_eigenvalues(top_k=2)
            max_eigenvalue = eigenvalues[0]
        else:
            max_eigenvalue = 0

        ##############################
        # Dead neuron analysis
        self.optimizer.zero_grad()
        rl_loss.backward()

        # dead neuron w.r.t activation
        num_zero_activation_params, num_activation_params = {}, {}
        for layer_name, activations in layer_wise_outputs.items():
            num_zero_activation_params[layer_name] = torch.sum(activations == 0).item()
            num_activation_params[layer_name] = activations.numel()

        zero_activation_ratio = get_param_cnt_ratio(
            num_zero_activation_params, num_activation_params, 'zero_activation_ratio', activation = True)

        ratios = {}        
        ratios.update(zero_activation_ratio)

        # log evaluation metrics
        eval_logs = {
            'pred_var': pred_var.item(),
            'target_var': target_var.item(),
            'exp_var': exp_var.item(),
            'rl_loss': rl_loss.item(),
            'max_eigenvalue': max_eigenvalue,
        }

        eval_logs.update(ratios)

        return eval_logs 

    def rollout(self):
        if self.cfg.rollout_model == 'online':
            rollout_model = self.model
        else:
            rollout_model = self.target_model        

        exp_noise = 0.0
        for _ in tqdm.tqdm(range(self.cfg.num_eval_trajectories)):
            obs = self.eval_env.reset()
            while True:
                # encode last observation to torch.tensor()
                obs_tensor = self.buffer.encode_obs(obs, prediction=True)

                # evaluation is based on greedy prediction
                with torch.no_grad():
                    action = self.predict(rollout_model, obs_tensor, self.cfg.eval_eps, exp_noise)

                # step
                next_obs, reward, done, info = self.eval_env.step(action)

                # logger
                self.logger.step(obs, reward, done, info, mode='eval')

                # move on
                if info.traj_done:
                    break
                else:
                    obs = next_obs
        
