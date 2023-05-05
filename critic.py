from typing import Tuple
import jax.numpy as jnp
from common import PRNGKey
import policy
import jax

from common import Batch, InfoDict, Model, Params

EPS = jnp.finfo(jnp.float32).eps
EPS2 = 1e-3


def update_cost(key: PRNGKey, cost: Model, batch: Batch, cost_grad_coeff: float) -> Tuple[Model, InfoDict]:
    
    expert_inputs = jnp.concatenate([batch.expert_observations, batch.expert_actions], axis=-1)
    union_inputs = jnp.concatenate([batch.union_observations, batch.union_actions], axis=-1)
    
    def cost_loss_fn(cost_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        expert_cost = cost.apply({'params': cost_params}, expert_inputs)
        union_cost = cost.apply({'params': cost_params}, union_inputs)
        cost_loss = minimax_discriminator_loss(expert_cost, union_cost)
        
        gradient_penalty = cost_gradient_penalty(key, cost, cost_params, expert_inputs, union_inputs)
        cost_loss += cost_grad_coeff * gradient_penalty
        return cost_loss, {
            'cost_loss': cost_loss,
            'expert_cost': expert_cost.mean(),
            'union_cost': union_cost.mean()
        }
    
    new_cost, info = cost.apply_gradient(cost_loss_fn)

    return new_cost, info
        

def minimax_discriminator_loss(real_outputs, gen_outputs) -> jnp.ndarray:
    
    real_labels = jnp.ones_like(real_outputs)
    gen_labels = jnp.zeros_like(gen_outputs)

    def sigmoid_cross_entropy(logits, labels):
        zeros = jnp.zeros_like(logits)
        cond = (logits >= zeros)
        relu_logits = jnp.where(cond, logits, zeros)
        neg_abs_logits = jnp.where(cond, -logits, logits)
        return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))
    
    real_loss = sigmoid_cross_entropy(real_outputs, real_labels).mean()
    gen_loss = sigmoid_cross_entropy(gen_outputs, gen_labels).mean()
    
    return real_loss + gen_loss


def cost_gradient_penalty(key: PRNGKey, cost: Model, cost_params: Params
                          , expert_inputs: jnp.ndarray, union_inputs: jnp.ndarray):
    
    unif_rand = jax.random.uniform(key=key, shape=(expert_inputs.shape[0], 1), minval=0., maxval=1.)
    mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
    mixed_inputs2 = unif_rand * jax.random.permutation(key, union_inputs) + (1 - unif_rand) * union_inputs
    mixed_inputs = jnp.concatenate([mixed_inputs1, mixed_inputs2], axis=0)

    def cost_output(inputs) -> jnp.ndarray:
        # cost_outputs = cost(inputs)
        cost_outputs = cost.apply({'params': cost_params}, inputs)
        cost_outputs = jnp.log(1 / (jax.nn.sigmoid(cost_outputs) + EPS2) - 1 + EPS2)
        return cost_outputs
    
    y, g_vjp = jax.vjp(cost_output, mixed_inputs)
    gradient, = g_vjp(jnp.ones_like(y))
    gradient += EPS

    gradient_penalty = (jnp.square(jnp.linalg.norm(gradient, axis=-1, keepdims=True) - 1.)).mean()
    return gradient_penalty