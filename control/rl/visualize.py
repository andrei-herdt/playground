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
from mujoco import mjx

import mediapy as media
import networks as nw


def get_image(state: State, camera: str, env) -> np.ndarray:
    """Renders the environment state."""
    d = mujoco.MjData(env.model)
    # write the mjx.Data into an mjData object
    mjx.device_get_into(d, state.pipeline_state)
    mujoco.mj_forward(env.model, d)
    # use the mjData object to update the renderer
    renderer.update_scene(d, camera=camera)
    return renderer.render()


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

renderer = mujoco.Renderer(eval_env.model)

# @markdown Commands **only used for Barkour Env**:
x_vel = 1.0  # @param {type: "number"}
y_vel = 0.0  # @param {type: "number"}
ang_vel = -0.5  # @param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info["command"] = the_command
rollout = [state]
images = [get_image(state, camera="track", env=eval_env)]

# grab a trajectory
n_steps = 500
render_every = 1

for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    if i % render_every == 0:
        images.append(get_image(state, camera="track", env=eval_env))

media.write_video(
    images=images, path="/workdir/quadruped.mp4", fps=1.0 / eval_env.dt / render_every
)
