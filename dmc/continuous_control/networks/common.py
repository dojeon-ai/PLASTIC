import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
    """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.

    Args:
        y: A pytree of numpy ndarray, vector y in the equation above.
    """
    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
    normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)  
    return normalized_gradient

def normalize_second_grad(second: jnp.ndarray, first_norm:float, second_norm: float) -> jnp.ndarray:
    """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.

    Args:
        y: A pytree of numpy ndarray, vector y in the equation above.
    """
    normalized_gradient = jax.tree_map(lambda x: x * (first_norm / second_norm), second)  
    return normalized_gradient

def print_traced_value(x):
    if isinstance(x, jnp.ndarray):  
        print(f'ShapedArray({x.shape}, dtype={x.dtype}), value={x}')
    elif isinstance(x, tuple):
        print(f'({", ".join(str(e) for e in x)})')
    else:
        print(str(x))


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  
InfoDict = Dict[str, float]

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    use_CReLU: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                
                # Use CReLU
                if self.use_CReLU:
                    x = jnp.concatenate((self.activations(x), self.activations(-x)), axis = -1)
                else:
                    x = self.activations(x)
                
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, use_sam:bool, rho:float, only_enc:bool) -> Tuple[Any, 'Model']:
        
        def get_sam_gradient(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            first_grads, info = grad_fn(model_params)
            dual_grads = dual_vector(first_grads)
            def update_fn(p, n):
                return p + rho * n
            noised_model = jax.tree_map(update_fn, 
                                        model_params,
                                        dual_grads)
            second_grads, info = grad_fn(noised_model)

            return first_grads, second_grads, info

        if use_sam:
            first_grads, grads, info = get_sam_gradient(self.params)
        else:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, info = grad_fn(self.params)

        grad_norms = tree_norm(grads)
        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info, grad_norms
    
    def return_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        return grads, info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


def split_tree(tree, key):
    tree_head = tree.unfreeze()
    tree_enc = tree_head.pop(key)
    tree_head = flax.core.FrozenDict(tree_head)
    tree_enc = flax.core.FrozenDict(tree_enc)
    return tree_enc, tree_head


# to separate opt_state for encoder and other layers
@flax.struct.dataclass
class ModelDecoupleOpt:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    tx_enc: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state_enc: Optional[optax.OptState] = None
    opt_state_head: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None,
               tx_enc: Optional[optax.GradientTransformation] = None) -> 'ModelDecoupleOpt':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            if tx_enc is None:
                tx_enc = tx
            params_enc, params_head = split_tree(params, 'SharedEncoder')
            opt_state_enc = tx_enc.init(params_enc)
            opt_state_head = tx.init(params_head)
        else:
            opt_state_enc, opt_state_head = None, None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   tx_enc=tx_enc,
                   opt_state_enc=opt_state_enc,
                   opt_state_head=opt_state_head)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, use_sam:bool, rho:float, only_enc:bool) -> Tuple[Any, 'Model']:

        def get_sam_gradient(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            first_grads, info = grad_fn(model_params)
            dual_grads = dual_vector(first_grads)
            def update_fn(p, n):
                return p + rho * n
            noised_model = jax.tree_map(update_fn, 
                                        model_params,
                                        dual_grads)
            second_grads, info = grad_fn(noised_model)

            return first_grads, second_grads, info

        if use_sam:
            first_grads, grads, info = get_sam_gradient(self.params)
        else:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, info = grad_fn(self.params)

        grad_norms = tree_norm(grads)
        params_enc, params_head = split_tree(self.params, 'SharedEncoder')

        if only_enc:
            grads_enc, _ = split_tree(grads, 'SharedEncoder')
            _, grads_head = split_tree(first_grads, 'SharedEncoder')
        else:
            grads_enc, grads_head = split_tree(grads, 'SharedEncoder')
        
        updates_enc, new_opt_state_enc = self.tx_enc.update(grads_enc, self.opt_state_enc,
                                                            params_enc)
        new_params_enc = optax.apply_updates(params_enc, updates_enc)

        updates_head, new_opt_state_head = self.tx.update(grads_head, self.opt_state_head,
                                                        params_head)
        new_params_head = optax.apply_updates(params_head, updates_head)

        new_params = flax.core.FrozenDict({**new_params_head, 'SharedEncoder': new_params_enc})
        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state_enc=new_opt_state_enc,
                            opt_state_head=new_opt_state_head), info, grad_norms

    def return_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        return grads, info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
