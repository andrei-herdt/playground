- [ ] Turn the sparse and dense formulations into a test

Inequality constrain foot
- [ ] Solve same problem using proxqp
- [ ] Try to express dynamics as equality constraints
- [ ] Double check solutions are correct, using qp and linalg.lstsq
  - [ ] Add inequalities
- [ ] Can we visualize cones?

- [ ] Why does the bias term of the joint control task make joint task less robust?
  - [ ] Is the commanded joint acceleration being realised?
  - [ ] In zero-gravity, zero torque command results in non-zero joint accelerations... Is it because of gravity?
    - [ ] Why is gravity not deactivated by model.opt.gravity = zero
    - [ ] Do we need to use the whole mass matrix?

Equality constraints
- [ ] Understand constraints mechanics. Weld seems to work but destabilises control
  - [ ] Read chapter on solver parameters
  - [ ] meaning of model.eq_*
  - [ ] meaning of data.qfrc_*
  - [ ] Why are torques much larger when activating equality constraints
    - [ ] Relax solver impedance?


- [ ] Create identic models of humanoid in mujoco and pinocchio

- [ ] Add plots

General:
  - [ ] Why does the joint task make the conditioning worse? It is the mass matrix.
```python
from robot_descriptions.loaders.mujoco import load_robot_description
robot_mj = load_robot_description("jvrc_mj_description")
from robot_descriptions.loaders.pinocchio import load_robot_description
robot_pin = load_robot_description("jvrc_description")
```
