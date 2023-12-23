from datetime import datetime
import functools
import numpy as np

from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from brax import envs
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
import mujoco

from absl import logging

from environments import BarkourEnv, domain_randomize

import mujoco
from mujoco import mjx

import jax
from jax import numpy as jp

from typing import Any, Dict, Tuple, Union

from brax.envs.base import Env, State
from brax.base import Base, Motion, Transform
from brax import math

from etils import epath

from ml_collections import config_dict

import numpy as np


logging.set_verbosity(logging.INFO)


envs.register_environment("barkour", BarkourEnv)
env_name = "barkour"
env = envs.get_environment(env_name)

# re-instantiate the renderer
renderer = mujoco.Renderer(env.model)

# Train policy

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks, 
    policy_hidden_layer_sizes=(128, 64, 32),
    value_hidden_layer_sizes=(128, 64, 32),
)
train_fn = functools.partial(
    ppo.train,
    num_timesteps=60_000_000,
    num_evals=3,
    reward_scaling=1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=8,
    gae_lambda=0.95,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=1024,
    network_factory=make_networks_factory,
    num_resets_per_eval=10,
    randomization_fn=domain_randomize,
    seed=0,
)


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 30, 0


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])
    print("num_steps: ", num_steps)
    print("eval/episode_reward: ", metrics["eval/episode_reward"])
    print("eval/episode_reward_std: ", metrics["eval/episode_reward_std"])


# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=progress, eval_env=eval_env
)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")


# save and reload params.
model_path = "/tmp/mjx_brax_quadruped_policy"
model.save_params(model_path, params)
