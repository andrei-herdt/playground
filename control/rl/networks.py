import jax
from brax.training.agents.ppo import networks as ppo_networks
import functools

def get_isaac_network():
    return functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 64, 32),
    value_hidden_layer_sizes=(128, 64, 32),
    activation=jax.nn.elu,)
