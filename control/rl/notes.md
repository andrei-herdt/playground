TODO:
- [ ] Examine individual episodes
- [ ] Plot learning curve

Flat terrain params
```python
# PPO
actor_hidden_dims = [128, 64, 32]
critic_hidden_dims = [128, 64, 32]
activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

num_envs = 4096
max_contact_force = 350.
num_observations = 48
episode_length_s = 20 # episode length in seconds
class scales ( AnymalCRoughCfg.rewards.scales ):
    orientation = -5.0
    torques = -0.000025
    feet_air_time = 2.
```

changenum=1
num_minibatches = 4
num_steps:  0 eval/episode_reward:  0.5760557 eval/episode_reward_std:  0.9477362
num_steps:  30310400 eval/episode_reward:  1.806519 eval/episode_reward_std:  4.0989275
num_steps:  60620800 eval/episode_reward:  8.016773 eval/episode_reward_std:  7.0157056
time to jit: 0:00:48.443359 time to train: 0:16:33.615585

changenum=1
num_envs=1024,
num_steps:  0 eval/episode_reward:  0.56519926 eval/episode_reward_std:  0.94614327
num_steps:  30310400 eval/episode_reward:  8.624402 eval/episode_reward_std:  6.6967926
num_steps:  60620800 eval/episode_reward:  11.163882 eval/episode_reward_std:  6.147504
time to jit: 0:00:46.034939 time to train: 0:25:47.985451

ff9ffb95f64a821fc80553d6aa9e756ca97ce7db
num_steps:  0 eval/episode_reward:  0.5935141 eval/episode_reward_std:  1.0470779
num_steps:  32768000 eval/episode_reward:  7.8386593 eval/episode_reward_std:  5.9317703
num_steps:  65536000 eval/episode_reward:  9.442217 eval/episode_reward_std:  5.8058133
time to jit: 0:00:47.062260 time to train: 0:27:53.307440
