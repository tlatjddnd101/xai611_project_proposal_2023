from typing import Callable, Sequence, Tuple, Optional

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class Cost(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:

        critic = MLP((*self.hidden_dims, 1),
                     layer_norm=self.layer_norm,
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)
