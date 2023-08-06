# LQR based joint-space control
We start with LQR in the joint space from the example [here](https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/LQRipynb) 
Here, we are purely controlling the positions of the base and the joints. 
There is no consideration of contact forces or dynamics. 

Let's see what the limitations of this controller are.
We apply an impulse to various parts of the body.

Let's start considering the dynamics.
We want to be able to compensate for disturbances by generating angular momenta
We exploit the coupling between the linear and angular momentum.

Let's start considering contact forces to further improve robustness.
We add a penalty on the contact forces.

Now, we add constraints.

