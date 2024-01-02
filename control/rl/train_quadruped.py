from datetime import datetime
import functools

from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax import envs
import mujoco

from environments import BarkourEnv, BarkourEnvHutter, domain_randomize
import networks as nw

from typing import Dict, List, Any

import pandas as pd


envs.register_environment("barkour", BarkourEnv)
envs.register_environment("barkour_hutter", BarkourEnvHutter)
env_name = "barkour_hutter"
env = envs.get_environment(env_name)

renderer = mujoco.Renderer(env.model)

# Train policy
make_networks_factory = nw.get_isaac_network()

train_fn = functools.partial(
    ppo.train,
    num_timesteps=60_000_000,
    num_evals=3,
    reward_scaling=1,
    episode_length=1000,  # TODO: How many seconds is that? 20?
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    gae_lambda=0.95,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=1024,
    batch_size=1024,
    num_minibatches=4,
    network_factory=make_networks_factory,
    num_resets_per_eval=10,
    randomization_fn=None,
    seed=0,
)


data: Dict[str, List[Any]] = {
    "times": [],
    "num_steps": [],
    "reward": [],
    "reward_std": [],
}
start = datetime.now()


def progress(num_steps, metrics):
    global data
    data["times"].append(datetime.now())
    data["num_steps"].append(num_steps)
    data["reward"].append(metrics["eval/episode_reward"])
    data["reward_std"].append(metrics["eval/episode_reward_std"])


# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=progress, eval_env=eval_env
)

df = pd.DataFrame.from_dict(data)
print(df.to_markdown(index=False))
print(f"time to jit: {df['times'][1] - start}")
print(f"time to train: {df['times'][-1] - df['times'][1]}")


# save and reload params.
policy_id = ""
model_path = "/workdir/mjx_brax_quadruped_policy" + policy_id
model.save_params(model_path, params)
