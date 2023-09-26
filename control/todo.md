- [x] Implement analytical solution to qp-based problem
  - [x] Get mass matrix
  - [x] Build task jacobians
  - [x] Build reference vectors
  - [ ] Robot is unstable, solutions differ
    - [x] Build weight matrices
    - [x] Add joint task
    - [x] When using weld, where are the parameters in mjModel? We want to change the init position
          anchor set to 0,0,0 should set constraint to center of the first body
    - [ ] Why are torque signs different?

Equality constrain foot
- [wip] Validate using manipulator model
  - [ ] why is computed com acceleration non-zero the first step? qacc, and qvel non-zero?? 
    - [ ] Understand constraints mechanics. Weld seems to work but destabilises control
      - [ ] Read chapter on solver parameters
      - [ ] meaning of model.eq_*
      - [ ] meaning of data.qfrc_*
      - [ ] Why are torques much larger when activating equality constraints
        - [ ] Relax solver impedance?

Inequality constrain foot
- [ ] Solve same problem using proxqp
  - [ ] Add inequalities
- [ ] Can we visualize cones?

- [ ] Create identic models of humanoid in mujoco and pinocchio
```python
from robot_descriptions.loaders.mujoco import load_robot_description
robot_mj = load_robot_description("jvrc_mj_description")
from robot_descriptions.loaders.pinocchio import load_robot_description
robot_pin = load_robot_description("jvrc_description")
```
