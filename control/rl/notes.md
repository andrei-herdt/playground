TODO:
- [ ] Examine individual episodes
- [ ] Plot learning curve


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
increase batch_size
num_steps:  0 eval/episode_reward:  0.5935141 eval/episode_reward_std:  1.0470779
num_steps:  32768000 eval/episode_reward:  7.8386593 eval/episode_reward_std:  5.9317703
num_steps:  65536000 eval/episode_reward:  9.442217 eval/episode_reward_std:  5.8058133
time to jit: 0:00:47.062260 time to train: 0:27:53.307440


d200d204
![](videos/d200d204.mp4)
decrease batch_size, increase torque penalty
num_steps:  0 eval/episode_reward:  0.587013 eval/episode_reward_std:  1.0336502
num_steps:  30310400 eval/episode_reward:  6.013794 eval/episode_reward_std:  7.0553393
num_steps:  60620800 eval/episode_reward:  9.640033 eval/episode_reward_std:  9.364621
time to jit: 0:00:46.774209 time to train: 0:25:50.067106

d24e43d7
increase feet_air_time
num_steps:  0 eval/episode_reward:  0.5909057 eval/episode_reward_std:  1.0505813
num_steps:  30310400 eval/episode_reward:  3.4921775 eval/episode_reward_std:  6.37509
num_steps:  60620800 eval/episode_reward:  13.843228 eval/episode_reward_std:  7.590749
time to jit: 0:00:47.117407 time to train: 0:25:49.084794

70697c88
increase episode_length
num_steps:  0 eval/episode_reward:  0.57269526 eval/episode_reward_std:  0.9965664
num_steps:  30310400 eval/episode_reward:  30.21885 eval/episode_reward_std:  27.981796
num_steps:  60620800 eval/episode_reward:  19.369843 eval/episode_reward_std:  24.694622
time to jit: 0:01:29.635887 time to train: 0:27:17.197335

84a2f6a3
increase torque penalty
num_steps:  0 eval/episode_reward:  0.60052776 eval/episode_reward_std:  1.0443081
num_steps:  30310400 eval/episode_reward:  0.014213366 eval/episode_reward_std:  0.013499956
num_steps:  60620800 eval/episode_reward:  0.017713964 eval/episode_reward_std:  0.020368503
time to jit: 0:01:29.584691 time to train: 0:27:17.806234

42ee1b36
decrease torque penalty
num_steps:  0 eval/episode_reward:  0.60416865 eval/episode_reward_std:  1.0367799
num_steps:  30310400 eval/episode_reward:  9.292147 eval/episode_reward_std:  19.73348
num_steps:  60620800 eval/episode_reward:  31.627762 eval/episode_reward_std:  28.004442
time to jit: 0:01:28.993436 time to train: 0:27:16.385898
