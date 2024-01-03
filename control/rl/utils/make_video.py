from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax import envs

from environments import BarkourEnv, BarkourEnvHutter, State

import jax
from jax import numpy as jp

import numpy as np

import mujoco
from mujoco import mjx

import mediapy as media
import networks as nw

import argparse
import sys

parser = argparse.ArgumentParser(description="Generate video from policy")
parser.add_argument("--file", "-f", type=str, help="Path to the file", required=True)
args = parser.parse_args()


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


# Environment
envs.register_environment("barkour", BarkourEnv)
envs.register_environment("barkour_hutter", BarkourEnvHutter)
env_name = "barkour_hutter"
env = envs.get_environment(env_name)

# Inference function
normalize = running_statistics.normalize
make_networks_factory = nw.get_isaac_network()
# TODO: Get from env instead of hard-coding
ppo_network = make_networks_factory(env.dim_obs, 12, preprocess_observations_fn=normalize)
make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

renderer = mujoco.Renderer(env.model)

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
images = [get_image(state, camera="track", env=env)]

# grab a trajectory
n_steps = 500

render_every = 1
print("generate images")
for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    if i % render_every == 0:
        images.append(get_image(state, camera="track", env=env))

print("write video")
fps = 1.0 / env.dt / render_every
media.write_video(images=images, path=args.file, codec="gif", fps=fps)
