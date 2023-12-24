from brax.io import model
from brax.training.agents.ppo.networks import make_inference_fn
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax import envs

from environments import BarkourEnv, State

import jax
from jax import numpy as jp

import functools

import numpy as np

import mujoco
from mujoco import mj_step, mj_resetDataKeyframe, mjx
import mujoco.viewer
import time
import networks as nw


# load params.
model_path = "/workdir/mjx_brax_quadruped_policy"
params = model.load_params(model_path)

normalize = running_statistics.normalize
make_networks_factory = nw.get_isaac_network()
ppo_network = make_networks_factory(465, 12, preprocess_observations_fn=normalize)
make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# visualize policy
#
#
envs.register_environment("barkour", BarkourEnv)
env_name = "barkour"
eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

model = eval_env.model
data = mujoco.MjData(model)
mj_resetDataKeyframe(model, data, 0)

# @markdown Commands **only used for Barkour Env**:
x_vel = 1.0  # @param {type: "number"}
y_vel = 0.0  # @param {type: "number"}
ang_vel = -0.5  # @param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)

# grab a trajectory
n_steps = 500
render_every = 1
ctrl = jp.zeros(model.nu)

with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        act_rng, rng = jax.random.split(rng)
        obs = eval_env._get_obs(data.qpos, ctrl)
        ctrl, _ = jit_inference_fn(obs, act_rng)
        data.ctrl = ctrl

        mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
