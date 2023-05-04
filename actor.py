from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey

EPS2 = 1e-3

def update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model,
                 batch: Batch, alpha: float, alg: str) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    if alg == 'SQL':
        weight = q - v
        weight = jnp.maximum(weight, 0)
    elif alg == 'EQL':
        weight = jnp.exp(10 * (q - v) / alpha)

    weight = jnp.clip(weight, 0, 100.)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(weight * log_probs).mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_actor_oil(key: PRNGKey, actor: Model, critic: Model, value: Model, cost: Model,
                 batch: Batch, alpha: float, alg: str) -> Tuple[Model, InfoDict]:
    v = value(batch.union_observations)
    inputs = jnp.concatenate([batch.union_observations, batch.union_actions], axis=-1)
    q1, q2 = critic(inputs)
    q = jnp.minimum(q1, q2)
    
    cost_inputs = jnp.concatenate([batch.union_observations, batch.union_actions], axis=-1)
    cost_val = cost(cost_inputs)
    r = -jnp.log(1 / (jax.nn.sigmoid(cost_val) + EPS2) - 1 + EPS2)

    if alg == 'zril':
        weight = jnp.exp(10 * (r + q - v) / alpha)
    elif alg in ['sqla1', 'drdemo']:
        weight = jnp.exp(10 * (q - v) / alpha)
    else:
        raise NotImplementedError
    
    weight = jnp.clip(weight, 0, 100.)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.union_observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.union_actions)
        actor_loss = -(weight * log_probs).mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_actor_demodice(key: PRNGKey, actor: Model, value: Model, cost: Model,
                 batch: Batch, discount:float, alpha: float) -> Tuple[Model, InfoDict]:
    
    cost_inputs = jnp.concatenate([batch.union_observations, batch.union_actions], axis=-1)
    cost_val = cost(cost_inputs)
    r = -jnp.log(1 / (jax.nn.sigmoid(cost_val) + EPS2) - 1 + EPS2)
    
    v = value(batch.union_observations)
    next_v = value(batch.union_next_observations)
    adv_v = r + discount * next_v - v
    
    weight = jnp.exp(adv_v / alpha)
    weight /= weight.mean()

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.union_observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.union_actions)
        actor_loss = -(weight * log_probs).mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info