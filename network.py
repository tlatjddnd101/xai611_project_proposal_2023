from typing import Callable, Sequence, Tuple, Optional

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class Cost(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:

        cost = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(cost, -1)
