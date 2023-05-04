"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net

from common import Batch, InfoDict, Model, PRNGKey

from actor import update_actor, update_actor_oil, update_actor_demodice
from critic import update_q, update_v, update_q_oil, update_v_oil, update_cost_oil, update_v_demodice


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)

@jax.jit
def _update_jit_sql(
    rng: PRNGKey, actor: Model, critic: Model,
    value: Model, target_critic: Model, batch: Batch, discount: float, tau: float, 
    alpha: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = update_v(target_critic, value, batch, alpha, alg='SQL')
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic,
                                             new_value, batch, alpha, alg='SQL')
    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }

@jax.jit
def _update_jit_eql(
    rng: PRNGKey, actor: Model, critic: Model,
    value: Model, target_critic: Model, batch: Batch, discount: float, tau: float, 
    alpha: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    
    new_value, value_info = update_v(target_critic, value, batch, alpha, alg='EQL')
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic,
                                             new_value, batch, alpha, alg='EQL')
    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }
    
@jax.jit
def _update_jit_zril(
    rng: PRNGKey, actor: Model, critic: Model,
    value: Model, target_critic: Model, cost: Model, batch: Batch, discount: float, tau: float, 
    alpha: float, cost_grad_coeff: float, grad_coeff: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    
    key, rng = jax.random.split(rng)
    new_cost, cost_info = update_cost_oil(key, cost, batch, cost_grad_coeff)
    
    new_value, value_info = update_v_oil(key, target_critic, value, new_cost, batch, alpha, grad_coeff, alg='zril')
    
    new_actor, actor_info = update_actor_oil(key, actor, target_critic,
                                             new_value, new_cost, batch, alpha, alg='zril')
    new_critic, critic_info = update_q_oil(key, critic, new_value, new_cost, batch, discount, grad_coeff, alg='zril')

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, new_cost, {
        **cost_info,
        **critic_info,
        **value_info,
        **actor_info
    }
    
@jax.jit
def _update_jit_sqla1(
    rng: PRNGKey, actor: Model, critic: Model,
    value: Model, target_critic: Model, cost: Model, batch: Batch, discount: float, tau: float, 
    alpha: float, cost_grad_coeff: float, grad_coeff: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    
    key, rng = jax.random.split(rng)
    new_cost, cost_info = update_cost_oil(key, cost, batch, cost_grad_coeff)
    
    new_value, value_info = update_v_oil(key, target_critic, value, new_cost, batch, alpha, grad_coeff, alg='sqla1')
    
    new_actor, actor_info = update_actor_oil(key, actor, target_critic,
                                             new_value, new_cost, batch, alpha, alg='sqla1')
    new_critic, critic_info = update_q_oil(key, critic, new_value, new_cost, batch, discount, grad_coeff, alg='sqla1')

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, new_cost, {
        **cost_info,
        **critic_info,
        **value_info,
        **actor_info
    }
    
@jax.jit
def _update_jit_drdemo(
    rng: PRNGKey, actor: Model, critic: Model,
    value: Model, target_critic: Model, cost: Model, batch: Batch, discount: float, tau: float, 
    alpha: float, cost_grad_coeff: float, grad_coeff: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    
    key, rng = jax.random.split(rng)
    new_cost, cost_info = update_cost_oil(key, cost, batch, cost_grad_coeff)
    
    new_value, value_info = update_v_oil(key, target_critic, value, new_cost, batch, alpha, grad_coeff, alg='drdemo')
    
    new_actor, actor_info = update_actor_oil(key, actor, target_critic,
                                             new_value, new_cost, batch, alpha, alg='drdemo')
    new_critic, critic_info = update_q_oil(key, critic, new_value, new_cost, batch, discount, grad_coeff, alg='drdemo')

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, new_cost, {
        **cost_info,
        **critic_info,
        **value_info,
        **actor_info
    }
    
@jax.jit
def _update_jit_demodice(
    rng: PRNGKey, actor: Model, value: Model, cost: Model, 
    batch: Batch, discount: float,  
    alpha: float, cost_grad_coeff: float, grad_coeff: float
) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    
    key, rng = jax.random.split(rng)
    new_cost, cost_info = update_cost_oil(key, cost, batch, cost_grad_coeff)
    
    new_value, value_info = update_v_demodice(key, value, new_cost, batch, discount, alpha, grad_coeff)
    
    new_actor, actor_info = update_actor_demodice(key, actor, new_value, new_cost, 
                                                  batch, discount, alpha)

    return rng, new_actor, new_value, new_cost, {
        **cost_info,
        **value_info,
        **actor_info
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.1,
                 cost_grad_coeff: float = 10.0,
                 grad_coeff: float = 1e-4,
                 dropout_rate: Optional[float] = None,
                 value_dropout_rate: Optional[float] = None,
                 layernorm: bool = False,
                 max_steps: Optional[int] = None,
                 max_clip: Optional[int] = None,
                 mix_dataset: Optional[str] = None,
                 alg: Optional[str] = None,
                 opt_decay_schedule: str = "cosine"):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        # self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.alpha = alpha
        self.cost_grad_coeff = cost_grad_coeff
        self.grad_coeff = grad_coeff
        self.max_clip = max_clip
        self.alg = alg

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, cost_key = jax.random.split(rng, 5)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)
        critic_init_input = jnp.concatenate([observations, actions], axis=-1)
        critic = Model.create(critic_def,
                              inputs=[critic_key, critic_init_input],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims, layer_norm=layernorm, dropout_rate=value_dropout_rate)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, critic_init_input])
        
        cost_def = value_net.Cost(hidden_dims)
        cost_init_input = jnp.concatenate([observations, actions], axis=-1)
        cost = Model.create(cost_def,
                            inputs=[cost_key, cost_init_input],
                            tx=optax.adam(learning_rate=critic_lr))

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.cost = cost
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        # type <class 'str'> is not a valid JAX type.
        if self.alg == 'SQL':
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit_sql(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch, self.discount, self.tau, self.alpha)
        elif self.alg == 'EQL':
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit_eql(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch, self.discount, self.tau, self.alpha)
        elif self.alg == 'zril':
            new_rng, new_actor, new_critic, new_value, new_target_critic, new_cost, info = _update_jit_zril(
                self.rng, self.actor, self.critic, self.value, self.target_critic, self.cost,
                batch, self.discount, self.tau, self.alpha, self.cost_grad_coeff, self.grad_coeff)
        elif self.alg == 'sqla1':
            new_rng, new_actor, new_critic, new_value, new_target_critic, new_cost, info = _update_jit_sqla1(
                self.rng, self.actor, self.critic, self.value, self.target_critic, self.cost,
                batch, self.discount, self.tau, self.alpha, self.cost_grad_coeff, self.grad_coeff)
        elif self.alg == 'drdemo':
            new_rng, new_actor, new_critic, new_value, new_target_critic, new_cost, info = _update_jit_drdemo(
                self.rng, self.actor, self.critic, self.value, self.target_critic, self.cost,
                batch, self.discount, self.tau, self.alpha, self.cost_grad_coeff, self.grad_coeff)
        elif self.alg == 'demodice':
            new_rng, new_actor, new_value, new_cost, info = _update_jit_demodice(
                self.rng, self.actor, self.value, self.cost, batch, self.discount, 
                self.alpha, self.cost_grad_coeff, self.grad_coeff)
        else:
            raise NotImplementedError

        self.rng = new_rng
        self.actor = new_actor
        if self.alg == 'demodice':
            new_critic = None
            new_target_critic = None
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.value = new_value
        if self.alg not in ['SQL', 'EQL']:
            self.cost = new_cost

        return info
