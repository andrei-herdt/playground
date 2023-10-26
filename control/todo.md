- [ ] Make all the old models work
- [ ] Refactor
- [ ] Turn to sparse formulation and compare computation times

- [ ] Do balancing on one leg
- [ ] hierarchical (multiQP) formulation

- [ ] refactor old functions and make sure they still work (CI?)
- [ ] Limit contact forces or base acceleration to understand why base is tilting 
- [ ] simulation time-dependent, i.e. results differ when pc slows down
- [ ] how to assess stability

- [ ] Why does the bias term of the joint control task make joint task less robust?
  - [ ] Is the commanded joint acceleration being realised?

Equality constraints
- [ ] Understand constraints mechanics. Weld seems to work but destabilises control
  - [ ] Read chapter on solver parameters
  - [ ] meaning of model.eq_*
  - [ ] meaning of data.qfrc_*
  - [ ] Why are torques much larger when activating equality constraints
    - [ ] Relax solver impedance?

- [ ] Turn the sparse and dense formulations into a test

- [ ] Create identic models of humanoid in mujoco and pinocchio

- [ ] Add plots

General:
  - [ ] Why does the joint task make the conditioning worse? It is the mass matrix.


## Ideas for tests
- add a humanoid in space model and a test
- 3dof orientation error
- manipulator with root stability of tasks
- manipulator with root tracking accuracy
- compare force vs no force results
- compare sparse with dense

## Roadmap
- NMPC
- RL
