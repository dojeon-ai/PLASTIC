"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Union

from optax._src import base
import jax
import jax.numpy as jnp
import numpy as np
import optax

from continuous_control.agents.drq.augmentations import batched_random_crop
from continuous_control.agents.drq.networks import DrQDoubleCritic, DrQPolicy
from continuous_control.agents.sac import temperature
from continuous_control.agents.sac.actor import update as update_actor
from continuous_control.agents.sac.critic import target_update
from continuous_control.agents.sac.critic import update as update_critic
from continuous_control.datasets import Batch
from continuous_control.networks import policies
from continuous_control.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOpt

from flax.core.frozen_dict import unfreeze

eps = 1e-4

@functools.partial(jax.jit, static_argnames=('update_target', 'use_sam', 'rho', 'only_enc'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, update_target: bool, use_sam: bool, rho: float, only_enc: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    next_observations = batched_random_crop(key, batch.next_observations)

    batch = batch._replace(observations=observations,
                           next_observations=next_observations)
    rng, key = jax.random.split(rng)
    new_critic, critic_info, critic_norm = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            soft_critic = True,
                                            use_sam = use_sam,
                                            rho = rho,
                                            only_enc = only_enc)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
        new_actor_params = actor.params.copy(
            add_or_replace={'SharedEncoder': new_critic.params['SharedEncoder']})
        actor = actor.replace(params=new_actor_params)

        rng, key = jax.random.split(rng)
        new_actor, actor_info, actor_norm = update_actor(key, actor, new_critic, temp, batch, 
                                            use_sam = use_sam, rho = rho, only_enc = only_enc)
        new_temp, alpha_info, tem_norm = temperature.update(temp, actor_info['entropy'],
                                                target_entropy, use_sam = use_sam,
                                                rho=rho, only_enc = only_enc)
    else:
        new_target_critic = target_critic
        new_actor = actor
        new_temp = temp
        actor_info, alpha_info = {}, {}

    return rng, new_actor, new_critic, new_target_critic, new_temp, critic_norm, actor_norm, tem_norm, {
        **critic_info,
        **actor_info,
        **alpha_info
    }

class DrQLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 rho: float,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 0.1,
                 use_sam: bool = True,
                 only_enc:bool = True,
                 use_CReLU: bool = True,
                 use_LN: bool = True):

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = DrQPolicy(hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim, use_LN)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                     cnn_padding, latent_dim, use_LN)

        critic = ModelDecoupleOpt.create(critic_def,
                                         inputs=[critic_key, observations, actions],
                                         tx=optax.adam(learning_rate=critic_lr),
                                         tx_enc=optax.adam(learning_rate=critic_lr))
            
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        self.step = 0
        self.rho = rho
        self.use_sam = use_sam
        self.only_enc = only_enc
        self.prev_actions = None
        self.use_LN = use_LN
        self.use_CReLU = use_CReLU

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        
        if np.isnan(actions).any():
            actions = self.prev_actions
        else:
            self.prev_actions = actions

        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, critic_norm, actor_norm, tem_norm, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0, 
            use_sam = self.use_sam, rho=self.rho, only_enc = self.only_enc)

        critic_norm_np = np.asarray(critic_norm)
        actor_norm_np = np.asarray(actor_norm)
        tem_norm_np = np.asarray(tem_norm)

        # If get nan value, do not update
        if np.isnan(critic_norm_np).any():
            return info
        if np.isnan(actor_norm_np).any():
            return info
        if np.isnan(tem_norm_np).any():
            return info
        
        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
    

