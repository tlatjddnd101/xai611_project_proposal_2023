from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net

from common import Batch, InfoDict, Model, PRNGKey

from critic import update_cost_oil

@jax.jit
def _update_jit_discriminator(
    rng: PRNGKey, cost: Model, 
    batch: Batch, discount: float,  
    alpha: float, cost_grad_coeff: float, grad_coeff: float
) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    
    key, rng = jax.random.split(rng)
    new_cost, cost_info = update_cost(key, cost, batch, cost_grad_coeff)
    
    return rng, new_cost, {
        **cost_info,
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 cost_grad_coeff: float = 10.0):

        self.cost_grad_coeff = cost_grad_coeff

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, cost_key = jax.random.split(rng, 5)

        cost_def = value_net.Cost(hidden_dims)
        cost_init_input = jnp.concatenate([observations, actions], axis=-1)
        cost = Model.create(cost_def,
                            inputs=[cost_key, cost_init_input],
                            tx=optax.adam(learning_rate=critic_lr))

        self.cost = cost
        self.rng = rng


    def update(self, batch: Batch) -> InfoDict:
        # type <class 'str'> is not a valid JAX type.

        new_rng, new_cost, info = _update_jit_discriminator(
            self.rng, self.cost, batch, self.discount, 
            self.alpha, self.cost_grad_coeff, self.grad_coeff)

        self.rng = new_rng
        self.cost = new_cost

        return info
