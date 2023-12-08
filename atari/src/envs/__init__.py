from .base import BaseEnv
from .atari import AtariEnv
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses

ENVS = {subclass.get_name():subclass
        for subclass in all_subclasses(BaseEnv)}

def build_env(cfg):     
    cfg = OmegaConf.to_container(cfg)
    env_type = cfg.pop('type')
    env = ENVS[env_type]  
    train_env = env(**cfg)
    eval_env = env(**cfg)
    
    return train_env, eval_env
