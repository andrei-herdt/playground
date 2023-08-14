# LQR based joint-space control
We start with LQR in the joint space from the example [here](https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/LQRipynb) 
Here, we are purely controlling the positions of the base and the joints. 
There is no consideration of contact forces or dynamics. 

Let's see what the limitations of this controller are.
We apply an increasing set of impulses to the root.
- TODO: plot root positions

So far, the control is purely position-based.
Let's start considering the dynamics.
We want to be able to compensate for disturbances by generating angular momenta
We exploit the coupling between the linear and angular momentum.
We create a CoM position task:
$$
\ddot c_d = K_p(c_d-c) + K_d(\dot c_d - \dot c)
$$

$$
\begin{align}
\dot c = J_c \dot q \\
\ddot c = J_c \ddot q + \dot J_c \dot q
\end{align}
$$

This leads to:

$$
\ddot q_d = J^{*}_c (\ddot c - \dot J_c \dot q_d)
$$
Then, we obtain the desired torque $\tau_d$ via:
$$
M\ddot q + N\dot q + G = \tau
$$
to obtain:
$$
\tau_d = M\ddot q_d + N\dot q + G 
$$

For the joint-space task, we have directly the PD controller representing the complete task:
$$
\ddot q_d = K_p(q_d - q) + K_d(\dot q_d - \dot q)
$$

And a joint position task:
$$
\ddot q_d = K_p(q_d - q) + K_d(\dot q_d - \dot q)
$$

Let's start considering contact forces to further improve robustness.
We add a penalty on the contact forces.

Now, we add constraints.

