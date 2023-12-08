from typing import Tuple

import jax
import jax.numpy as jnp

from continuous_control.datasets import Batch
from continuous_control.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm

eps = 1e-4

def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch, use_sam:bool, rho:float, only_enc:bool) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, log_stds, means = actor.apply({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        # actions = jnp.clip(actions, -1.0 + eps, 1.0 - eps)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_params),
            'actor_action_mean': jnp.mean(jnp.abs(actions)) * 1e7,
            'actor_action_max': jnp.max(jnp.abs(actions)) * 1e7,
            'actor_log_std_mean': log_stds.mean(),
            'actor_log_std_min': log_stds.min(),
            'actor_log_std_max': log_stds.max(),
            'actor_mean_mean': means.mean(),
            'actor_mean_min': means.min(),
            'actor_mean_max': means.max()
        }
    if only_enc:
        use_sam = False

    new_actor, info, norm = actor.apply_gradient(actor_loss_fn, use_sam, rho, only_enc)

    return new_actor, info, norm

