TODO:
- [ ] 
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

Questions:
- [ ] What is entropy cost
