from typing import Tuple
import jax.numpy as jnp
from common import PRNGKey
import policy
import jax

from common import Batch, InfoDict, Model, Params

EPS = jnp.finfo(jnp.float32).eps
EPS2 = 1e-3

def update_v(critic: Model, value: Model, batch: Batch,
             alpha: float, alg: str) -> Tuple[Model, InfoDict]:

    critic_inputs = jnp.concatenate([batch.observation, batch.actions], axis=-1)
    q1, q2 = critic(critic_inputs)
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
        critic_inputs = jnp.concatenate([batch.observation, batch.actions], axis=-1)
        q1, q2 = critic.apply({'params': critic_params}, critic_inputs)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_v_oil(key: PRNGKey, critic: Model, value: Model, cost: Model, batch: Batch,
             alpha: float, grad_coeff: float, alg: str) -> Tuple[Model, InfoDict]:

    inputs = jnp.concatenate([batch.union_observations, batch.union_actions], axis=-1)
    q1, q2 = critic(inputs)
    q = jnp.minimum(q1, q2)
    
    cost_val = cost(inputs)
    r = -jnp.log(1 / (jax.nn.sigmoid(cost_val) + EPS2) - 1 + EPS2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.union_observations)

        if alg == 'zril':
            target_v = r + q
        elif alg in ['sqla1', 'drdemo']:
            target_v = q
        else:
            raise NotImplementedError
        
        sp_term = (target_v - v) / alpha
        sp_term = jnp.minimum(sp_term, 5.0)
        max_sp_term = jnp.max(sp_term, axis=0)
        max_sp_term = jnp.where(max_sp_term < -1.0, -1.0, max_sp_term)
        max_sp_term = jax.lax.stop_gradient(max_sp_term)
        value_loss = (jnp.exp(sp_term - max_sp_term) + jnp.exp(-max_sp_term) * v / alpha).mean()

        gradient_penalty = network_gradient_penalty(key, value, value_params, batch.expert_observations, batch.union_observations,
                                                    batch.expert_next_observations, batch.union_next_observations)
        value_loss += grad_coeff * gradient_penalty
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'adv_v': (target_v - v).mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q_oil(key: PRNGKey, critic: Model, value: Model, cost: Model,
             batch: Batch, discount: float, grad_coeff: float, alg: str) -> Tuple[Model, InfoDict]:
    next_v = value(batch.union_next_observations)
    
    union_inputs = jnp.concatenate([batch.union_observations, batch.union_actions], axis=-1)
    cost_val = cost(union_inputs)
    r = -jnp.log(1 / (jax.nn.sigmoid(cost_val) + EPS2) - 1 + EPS2)
    
    if alg == 'zril':
        target_q = discount * (1. - batch.union_dones) * next_v
    elif alg in ['sqla1', 'drdemo']:
        target_q = r + discount * (1. - batch.union_dones) * next_v

    expert_inputs = jnp.concatenate([batch.expert_observations, batch.expert_actions], axis=-1)
    union_next_inputs = jnp.concatenate([batch.union_next_observations, batch.union_next_actions], axis=-1)
    expert_next_inputs = jnp.concatenate([batch.expert_next_observations, batch.expert_next_actions], axis=-1)
    
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, union_inputs)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        
        # gradient_penalty = network_gradient_penalty(key, critic, critic_params, expert_inputs, union_inputs,
        #                                             expert_next_inputs, union_next_inputs)
        # critic_loss += grad_coeff * gradient_penalty
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_cost_oil(key: PRNGKey, cost: Model, batch: Batch, cost_grad_coeff: float) -> Tuple[Model, InfoDict]:
    
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


def network_gradient_penalty(key: PRNGKey, network: Model, net_params: Params, expert_inputs: jnp.ndarray, union_inputs: jnp.ndarray
                             , expert_next_inputs: jnp.ndarray, union_next_inputs: jnp.ndarray):
    
    unif_rand = jax.random.uniform(key=key, shape=(expert_inputs.shape[0], 1), minval=0., maxval=1.)
    net_inter = unif_rand * expert_inputs + (1. - unif_rand) * union_inputs
    net_next_inter = unif_rand * expert_next_inputs + (1. - unif_rand) * union_next_inputs
    net_inter = jnp.concatenate([union_inputs, net_inter, net_next_inter], axis=0)
    
    def net_output(inputs) -> jnp.ndarray:
        net_outputs = network.apply({'params': net_params}, inputs)
        return net_outputs
    
    y, g_vjp = jax.vjp(net_output, net_inter)
    gradient, = g_vjp(jnp.ones_like(y))
    gradient += EPS
    
    gradient_penalty = (jnp.square(jnp.linalg.norm(gradient, axis=-1, keepdims=True))).mean()
    return gradient_penalty


def update_v_demodice(key: PRNGKey, value: Model, cost: Model, batch: Batch,
             discount: float, alpha: float, grad_coeff: float) -> Tuple[Model, InfoDict]:
    
    cost_inputs = jnp.concatenate([batch.union_observations, batch.union_actions], axis=-1)
    cost_val = cost(cost_inputs)
    r = -jnp.log(1 / (jax.nn.sigmoid(cost_val) + EPS2) - 1 + EPS2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        init_v = value.apply({'params': value_params}, batch.union_init_observations)
        v = value.apply({'params': value_params}, batch.union_observations)
        next_v = value.apply({'params': value_params}, batch.union_next_observations)

        adv_v = r + discount * next_v - v
        non_linear_loss = alpha * jax.scipy.special.logsumexp(adv_v / alpha)
        linear_loss = (1. - discount) * init_v.mean()
        value_loss = linear_loss + non_linear_loss

        gradient_penalty = network_gradient_penalty(key, value, value_params, batch.expert_observations, batch.union_observations,
                                                    batch.expert_next_observations, batch.union_next_observations)
        value_loss += grad_coeff * gradient_penalty
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'adv_v': adv_v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info