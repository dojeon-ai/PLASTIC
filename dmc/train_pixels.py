import os
import random
import wandb
import pickle

import jax
import flax
import numpy as np
import jax.numpy as jnp
import optax
import tqdm
from absl import app, flags
from ml_collections import config_flags

from continuous_control.agents import DrQLearner
from continuous_control.datasets import ReplayBuffer
from continuous_control.evaluation import evaluate
from continuous_control.utils import make_env
from continuous_control.logger import WandbAgentLogger

from flax import serialization

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', '', 'Experiment description (not actually used).')
flags.DEFINE_string('env_name', 'quadruped-run', 'Environment name.')
flags.DEFINE_string('project_name', 'DMC_reset', 'Wandb project name.')
flags.DEFINE_string('entity', 'test', 'Wandb entity name.')
flags.DEFINE_string('exp_name', 'test', 'Wandb exp name.')
flags.DEFINE_string('save_dir', './out/', 'Logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('device_id', '0', 'device id.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_integer('log_frequency', int(2000), 'Wandb log interval.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', True, 'Save videos during evaluation.')
flags.DEFINE_integer('updates_per_step', 1, 'Number of updates per step')

## PLASTIC config
# Reset config
flags.DEFINE_string('reset_type', 'llf', 'Reset strategy.')
flags.DEFINE_integer('reset_interval', 25000, 'Periodicity of resets.')
flags.DEFINE_boolean('resets', True, 'Periodically reset last actor / critic layers.')

# SAM config
flags.DEFINE_float('rho', 0.01, 'perturbation value in sam update.')
flags.DEFINE_boolean('use_sam', True, 'Use sam optimizer.')
flags.DEFINE_boolean('only_enc', True, 'use sam only at encoder gradients.')

# CReLU
flags.DEFINE_boolean('use_CReLU', True, 'Use CReLU.')
# LayerNorm
flags.DEFINE_boolean('use_LN', True, 'Use LayerNormalization.')


config_flags.DEFINE_config_file(
    'config',
    'configs/drq.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}

MAX_STEP_PER_ENV = {
    'cartpole-balance':500000,
    'cartpole-balance_sparse':500000,
    'cartpole-swingup':500000,
    'finger-spin':500000,
    'walker-stand':500000,
    'walker-walk':500000,
    'hopper-stand':1000000,
    'pendulum-swingup':1000000,
    'acrobot-swingup':2000000,
    'cartpole-swingup_sparse':2000000,
    'cheetah-run':2000000,
    'finger-turn_easy':2000000,
    'finger-turn_hard':2000000,
    'hopper-hop':2000000,
    'quadruped-run':2000000,
    'quadruped-walk':2000000,
    'reacher-easy':2000000,
    'reacher-hard':2000000,
    'walker-run':2000000,
    'reach-duplo':2000000,
    'ball_in_cup-catch': 500000,
}
    
RESET_INTERVAL_PER_ENV = {
    'cartpole-balance':25000,
    'cartpole-balance_sparse':25000,
    'cartpole-swingup':25000,
    'finger-spin':25000,
    'walker-stand':25000,
    'walker-walk':25000,
    'hopper-stand':50000,
    'pendulum-swingup':50000,
    'acrobot-swingup':100000,
    'cartpole-swingup_sparse':100000,
    'cheetah-run':100000,
    'finger-turn_easy':100000,
    'finger-turn_hard':100000,
    'hopper-hop':100000,
    'quadruped-run':100000,
    'quadruped-walk':100000,
    'reacher-easy':100000,
    'reacher-hard':100000,
    'walker-run':100000,
    'reach-duplo':100000,
    'ball_in_cup-catch': 25000,
}


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    ## Different max_steps and reset_interval for each envs 
    FLAGS.max_steps = MAX_STEP_PER_ENV[FLAGS.env_name]
    FLAGS.reset_interval = RESET_INTERVAL_PER_ENV[FLAGS.env_name]

    kwargs = dict(FLAGS.config)
    if FLAGS.env_name == 'cheetah-run': FLAGS.rho = FLAGS.rho / 5
    kwargs.update({'use_sam': FLAGS.use_sam, 
                   'rho': FLAGS.rho, 
                   'only_enc': FLAGS.only_enc,
                   'use_CReLU': FLAGS.use_CReLU,
                   'use_LN': FLAGS.use_LN})

    logger = WandbAgentLogger(project_name = FLAGS.project_name,
                              entity = FLAGS.entity,
                              cfg = kwargs,
                              exp_name = FLAGS.exp_name,
                              action_repeat=action_repeat)

    gray_scale = kwargs.pop('gray_scale')
    image_size = kwargs.pop('image_size')

    def make_pixel_env(seed, video_folder, save_video = False):
        return make_env(FLAGS.env_name,
                        seed,
                        video_folder,
                        action_repeat=action_repeat,
                        image_size=image_size,
                        frame_stack=3,
                        from_pixels=True,
                        gray_scale=False,
                        save_video = save_video)

    env = make_pixel_env(FLAGS.seed, video_train_folder)
    eval_env = make_pixel_env(FLAGS.seed + 42, video_eval_folder, save_video = FLAGS.save_video)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    assert kwargs.pop('algo') == 'drq'
    updates_per_step = kwargs.pop('updates_per_step')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    rho = kwargs.pop('rho')

    obs_demo = env.observation_space.sample()
    action_demo = env.action_space.sample()
    agent = DrQLearner(FLAGS.seed,
                       obs_demo[np.newaxis],
                       action_demo[np.newaxis], 
                       rho, **kwargs)
    
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)
    
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps // action_repeat + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        logger.step(observation, reward, done, mode='train')
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        if i >= FLAGS.start_training:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                train_info = agent.update(batch)
            logger.update_log(mode='train', **train_info)

        if (i == 1) or (i % FLAGS.eval_interval) == 0:
            eval_stats, video = evaluate(agent, eval_env, FLAGS.eval_episodes)
            eval_log = {'Episode Return': eval_stats['return'],
                        'Video': wandb.Video(video[::8, :, ::2, ::2], fps=6, format='gif')}
            logger.update_log(mode='eval', **eval_log)

        if i % FLAGS.log_frequency == 0:
            logger.write_log(mode='train')
            logger.write_log(mode='eval')
        
    
        if FLAGS.resets and (i % FLAGS.reset_interval == 0) and (i != (FLAGS.max_steps // action_repeat)):
            # shared enc params: 388416
            # critic head(s) params: 366232
            # actor head params: 286882
            # total params: 1041530
            # so we reset roughtly half of the agent (both layer and param wise)
            
            # save encoder parameters
            old_critic_enc = agent.critic.params['SharedEncoder']
            # target critic has its own copy of encoder
            old_target_critic_enc = agent.target_critic.params['SharedEncoder']
            # save encoder optimizer statistics
            old_critic_enc_opt = agent.critic.opt_state_enc
            
            # create new agent: note that the temperature is new as well
            agent = DrQLearner(FLAGS.seed + i,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis],
                            rho, **kwargs)
            
            # resetting critic: copy encoder parameters and optimizer statistics
            new_critic_params = agent.critic.params.copy(
                add_or_replace={'SharedEncoder': old_critic_enc})
            agent.critic = agent.critic.replace(params=new_critic_params, 
                                                opt_state_enc=old_critic_enc_opt)
            
            # resetting actor: actor in DrQ uses critic's encoder
            # note we could have copied enc optimizer here but actor does not affect enc
            new_actor_params = agent.actor.params.copy(
                add_or_replace={'SharedEncoder': old_critic_enc})
            agent.actor = agent.actor.replace(params=new_actor_params)
            
            # resetting target critic
            new_target_critic_params = agent.target_critic.params.copy(
                add_or_replace={'SharedEncoder': old_target_critic_enc})
            agent.target_critic = agent.target_critic.replace(
                params=new_target_critic_params)
    
    ## Final eval
    eval_stats, video = evaluate(agent, eval_env, FLAGS.eval_episodes)
    eval_log = {'Episode Return': eval_stats['return'],
                'Video': wandb.Video(video[::8, :, ::2, ::2], fps=6, format='gif')}
    logger.update_log(mode='eval', **eval_log)
    logger.write_log(mode='eval')
            
if __name__ == '__main__':
    app.run(main)
