from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from continuous_control.networks.common import InfoDict, Model


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(temp: Model, entropy: float,
           target_entropy: float, use_sam:bool, rho:float, only_enc:bool) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    if only_enc:
        use_sam = False
    new_temp, info, norm= temp.apply_gradient(temperature_loss_fn, use_sam, rho, only_enc)

    return new_temp, info, norm
