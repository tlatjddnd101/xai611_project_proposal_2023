from typing import Tuple
import jax.numpy as jnp
from common import PRNGKey
import policy
import jax

from common import Batch, InfoDict, Model, Params

EPS = jnp.finfo(np.float32).eps
EPS2 = 1e-3

def update_v(critic: Model, value: Model, batch: Batch,
             alpha: float, alg: str) -> Tuple[Model, InfoDict]:

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        if alg == 'SQL':
            sp_term = (q - v) / (2 * alpha) + 1.0
            sp_weight = jnp.where(sp_term > 0, 1., 0.)
            value_loss = (sp_weight * (sp_term**2) + v / alpha).mean()
        elif alg == 'EQL':
            sp_term = (q - v) / alpha
            sp_term = jnp.minimum(sp_term, 5.0)
            max_sp_term = jnp.max(sp_term, axis=0)
            max_sp_term = jnp.where(max_sp_term < -1.0, -1.0, max_sp_term)
            max_sp_term = jax.lax.stop_gradient(max_sp_term)
            value_loss = (jnp.exp(sp_term - max_sp_term) + jnp.exp(-max_sp_term) * v / alpha).mean()
        else:
            raise NotImplementedError('please choose SQL or EQL')
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'q-v': (q - v).mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, value: Model,
             batch: Batch,  discount: float) -> Tuple[Model, InfoDict]:
    next_v = value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_cost(key: PRNGKey, cost: Model, batch: Batch, cost_grad_coeff: float) -> Tuple[Model, InfoDict]:
    
    expert_cost_val = cost(batch.expert_observations, batch.expert_actions)
    union_cost_val = cost(batch.union_observations, batch.union_actions)
    
    def cost_loss_fn(cost_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        expert_cost = cost.apply({'params': cost_params}, batch.expert_observations, batch.expert_actions)
        union_cost = cost.apply({'params': cost_params}, batch.union_observations, batch.union_actions)
        cost_loss = discriminator_loss(expert_cost, union_cost).mean()
        
        gradient_penalty = cost_gradient_penalty(key, cost, batch)
        cost_loss += cost_grad_coeff * gradient_penalty
        return cost_loss, {
            'cost_loss': cost_loss,
            'expert_cost': expert_cost.mean(),
            'union_cost': union_cost.mean()
        }
    
    new_cost, info = cost.apply_gradient(cost_loss_fn)
    
    return new_cost, info
        

def discriminator_loss(real_outputs, gen_outputs) -> jnp.ndarray:
    
    real_labels = jnp.ones_like(real_outputs)
    gen_labels = jnp.zeros_like(gen_outputs)
    
    def sigmoid_cross_entropy(logits, labels):
        return jnp.maximum(logitis, 0.) - logits * labels + jnp.log(1. + jnp.exp(-jnp.abs(logits)))
    
    real_loss = sigmoid_cross_entropy(real_outputs, real_labels)
    gen_loss = sigmoid_cross_entropy(gen_outputs, gen_labels)
    
    return real_loss + gen_loss


def cost_gradient_penalty(key, cost, batch):
    
    unif_rand = jax.random.uniform(key=key, shape=(batch.expert_observations.shape[0], 1), minval=0., maxval=1.)
    mixed_obs1 = unif_rand * batch.expert_observations + (1. - unif_rand) * batch.union_observations
    mixed_actions1 = unif_rand * batch.expert_actions + (1. - unif_rand) * batch.union_actions
    mixed_obs2 = unif_rand * jax.random.shuffle(key, batch.union_observations) + (1. - unif_rand) * batch.union_observations
    mixed_actions2 = unif_rand * jax.random.shuffle(key, batch.union_actions) + (1. - unif_rand) * batch.union_actions
    
    mixed_observations = jnp.concatenate([mixed_obs1, mixed_obs2], axis=0)
    mixed_actions = jnp.concatenate([mixed_actions1, mixed_actions2], axis=0)
    
    def cost_output(cost_params: Params):
        cost_outputs = cost.apply({'params', cost_params}, mixed_observations, mixed_actions)
        cost_outputs = jnp.log(1 / (jax.nn.sigmoid(cost_outputs) + EPS2) - 1 + EPS2)
        return cost_outputs
    
    grad_fn = jax.grad(cost_output)
    gradient = grad_fn(cost.params) + EPS
    
    gradient_penalty = (jnp.square(jnp.linalg.norm(gradient, keepdims=True) - 1.)).mean()
    return gradient_penalty