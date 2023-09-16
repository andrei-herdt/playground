- [x] Implement analytical solution to qp-based problem
  - [x] Get mass matrix
  - [x] Build task jacobians
  - [x] Build reference vectors
  - [ ] Robot is unstable, solutions differ
    - [x] Build weight matrices
    - [x] Add joint task
    - [ ] Why are torque signs different?
    - [ ] Relax solver impedance


- [ ] Solve same problem using proxqp

- [ ] Create identic models of humanoid in mujoco and pinocchio
```python
from robot_descriptions.loaders.mujoco import load_robot_description
robot_mj = load_robot_description("jvrc_mj_description")
from robot_descriptions.loaders.pinocchio import load_robot_description
robot_pin = load_robot_description("jvrc_description")
```

- [ ] git public key not available inside docker
- [ ] dubious access rights error message in docker

