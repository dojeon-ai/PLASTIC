from .base import BaseAgent
from .buffers import *
from dotmap import DotMap
from omegaconf import OmegaConf
from src.common.augmentation import Augmentation

from src.common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseAgent)

AGENTS = {subclass.get_name():subclass
          for subclass in all_subclasses(BaseAgent)}

BUFFERS = {subclass.get_name():subclass
           for subclass in all_subclasses(BaseBuffer)}


def build_agent(cfg,
                device,
                train_env,
                eval_env,
                logger,
                model):
    
    cfg = DotMap(OmegaConf.to_container(cfg))
    cfg.total_optimize_steps = int((cfg.num_timesteps - cfg.min_buffer_size) * cfg.optimize_per_env_step)

    # augemntation
    if len(cfg.aug_types) == 0:
        cfg.aug_types = []
    aug_func = Augmentation(obs_shape=cfg.obs_shape, 
                            aug_types=cfg.aug_types)

    # buffer
    buffer_cfg = cfg.pop('buffer')
    buffer_type = buffer_cfg.pop('type')
    
    if buffer_type != str(None):
        buffer = BUFFERS[buffer_type]
        if 'prior_weight_scheduler' in buffer_cfg:
            buffer_cfg.prior_weight_scheduler.step_size = cfg.total_optimize_steps
        buffer = buffer(device=device, gamma=cfg['gamma'], **buffer_cfg)
    else:
        buffer = None

    agent_type = cfg.pop('type')
    agent = AGENTS[agent_type]
    return agent(cfg=cfg,
                 device=device,
                 train_env=train_env,
                 eval_env=eval_env,
                 logger=logger,
                 buffer=buffer,
                 aug_func=aug_func,
                 model=model)
