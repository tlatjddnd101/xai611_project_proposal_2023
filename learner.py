from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import network

from common import Batch, InfoDict, Model, PRNGKey

from cost import update_cost

@jax.jit
def _update_jit_discriminator(
    rng: PRNGKey, cost: Model, batch: Batch
) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    
    key, rng = jax.random.split(rng)
    new_cost, cost_info = update_cost(key, cost, batch)
    
    return rng, new_cost, {
        **cost_info,
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256)):

        rng = jax.random.PRNGKey(seed)
        rng, cost_key = jax.random.split(rng, 2)

        cost_def = network.Cost(hidden_dims)
        cost_input = jnp.concatenate([observations, actions], axis=-1)
        cost = Model.create(cost_def,
                            inputs=[cost_key, cost_input],
                            tx=optax.adam(learning_rate=lr))

        self.cost = cost
        self.rng = rng


    def update(self, batch: Batch) -> InfoDict:
        # type <class 'str'> is not a valid JAX type.

        new_rng, new_cost, info = _update_jit_discriminator(
            self.rng, self.cost, batch)

        self.rng = new_rng
        self.cost = new_cost

        return info
