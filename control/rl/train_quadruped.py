from datetime import datetime
import functools

from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax import envs
import mujoco

from environments import BarkourEnv, BarkourEnvHutter, domain_randomize
import networks as nw

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
    num_evals=6,
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
    num_envs=512,
    batch_size=16384,
    num_minibatches=8,
    network_factory=make_networks_factory,
    num_resets_per_eval=10,
    randomization_fn=domain_randomize,
    seed=0,
)


start = datetime.now()
data_df = pd.DataFrame()


def progress(num_steps, metrics):
    global data_df

    metrics["num_steps"] = num_steps
    metrics["times"] = datetime.now()
    idx = data_df.index.size
    data_df = pd.concat([data_df, pd.DataFrame(metrics, index=[idx])])
    print(data_df[["eval/episode_reward_std", "eval/episode_reward"]].to_markdown())


# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=progress, eval_env=eval_env
)


# save policy
policy_id = ""
model_path = "/workdir/mjx_brax_quadruped_policy" + policy_id
model.save_params(model_path, params)

f = open("/workdir/metrics.md", "w")
f.write(data_df.to_markdown())
f.close()
print(f"time to jit: {data_df['times'].iloc[1] - start}")
print(f"time to train: {data_df['times'].iloc[-1] - data_df['times'].iloc[1]}")
