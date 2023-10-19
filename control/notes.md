github: https://github.com/andrei-herdt/playground/blob/master/control/notes.md

# LQR based joint-space control
We start with LQR in the joint space from the example [here](https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/LQRipynb) 
- [ ] If our task is expressed at acceleration level, and we are missing dynamic expressions, would an LQR formulation make sense?

Here, we are purely controlling the positions of the base and the joints. 
There is no consideration of contact forces or dynamics. 

Let's see what the limitations of this controller are.
We apply an increasing set of impulses to the root.
- TODO: plot root positions

### Considering dynamics
So far, the control is purely position-based.
Let's start considering the dynamics.
We want to be able to compensate for disturbances by generating angular momenta
We exploit the coupling between the linear and angular momentum.

We create a CoM position task:
$$\ddot c_d = K_p(c_d-c) + K_d(\dot c_d - \dot c)$$

$$
\begin{align}
\dot c = J_c \dot q_2 \\
\ddot c = J_c \ddot q_2 + \dot J_c \dot q_2
\end{align}
$$

This leads to:
$$\ddot q_d = J^{*}_c (\ddot c - \dot J_c \dot q_d),$$
where $^{+}$ denotes a pseudo-inverse.

### Considering dynamics
We use the Lagrangian:
$$\begin{align}
M_1\ddot q + N_1\dot q + G_1 &= J_c \lambda \\
M_2\ddot q + N_2\dot q + G_2 &= J_c \lambda + \tau
\end{align}$$
to obtain the desired torque $\tau_d$:
$$ \tau_d = M_2\ddot q_{2,d} + N_2\dot q_{2,d} + G_2 $$

For the joint-space task, we have directly the PD controller representing the complete task:
$$ \ddot q_d = K_p(q_d - q) + K_d(\dot q_d - \dot q) $$

When expressed on the same hierarchy level, the task becomes a trade-off between two tasks:
$$ \ddot q_d = \alpha_1 \ddot q_{d,1} + \alpha_2 \ddot q_{d,2}, $$
with $\alpha_1 + \alpha_2 = 1$.

### Quadratic Programming

The above tasks are fulfilled when $\ddot q = \ddot q_d$. For reasons that we shall see later, let's express this as a QP problem.
Our optimisation variable shall be the joint torque.
$$
\begin{align}
\underset{\tau}{\text{minimise}} & \space (\ddot q_d - \ddot q)^TQ(\ddot q_d - \ddot q) + \tau^TR\tau \\
\text{subject to} \space & \nonumber \\
& M\ddot q + N\dot q + G = \tau
\end{align}
$$

Let's first take a simplified version of the dynamics and kinematics:
$$
\begin{align}
\underset{\tau}{\text{minimise}} & \space \|\ddot x_1 - \ddot x_{1,d}\|^2_{W_1} +  \|\ddot x_2 - \ddot x_{2,d}\|^2_{W_2}  \\
\text{subject to} \space & \nonumber \\
& M_2\ddot q + h_2= \tau \\
& J_1\ddot q + \dot J_1 \dot q= \ddot x_1 \\
& J_2\ddot q + \dot J_2 \dot q= \ddot x_3
\end{align}
$$
We limit ourselves to two tasks and simplify the dynamics.
After substitution of equality constraints, we get:
$$
\begin{align}
\underset{\tau}{\text{minimise}} & \space\|J_1M_2^{-1}(\tau - h_2) - \ddot x_{1,d}\|^2_{W_1} +  \|J_2M_2^{-1}(\tau - h_2) - \ddot x_{2,d}\|^2_{W_2}  \\
\end{align}
$$
The solution of which is:
$$
\tau^* = (H_1+H_2)^+(r_1+r_2), 
$$
where $H_j = M^{-T}J_j^TW_jJ_jM^{-1}$ and $r_j = (J_jM^{-1}h+\ddot x_{j,d})W_jJ_jM^{-1}$

### Constrain end effector forces
For a rigid contact at an end effector, we have:
$$
M_2\ddot q + h_2 = \tau + J_e^T\lambda,
$$
where $J_e$ is the end effector Jacobian.

Then the expression for the force is:
$$
\lambda = J_e^{T*}(M_2\ddot q + h_2 - \tau).
$$
The dense formulation then becomes:
$$
\begin{align}
\underset{\tau}{\text{minimise}} & \space \sum_i\|J_iM_2^{-1}(\tau - h_2) - \ddot x_{i,d}\|^2_{W_i} +  \|\lambda - \lambda_d\|^2_{W_{\lambda}}  \\
\end{align}
$$

## Sparse formulation
We can avoid inverting the mass matrix when setting up the program if we express the mappings as equality constraints:
$$
\begin{align}
\underset{\tau,\ddot q}{\text{minimise}} & \sum_i\|\ddot x_i - \ddot x_{i,d}\|^2_{W_i}  \\ \text{subject to} \space & \nonumber \\
& M_2\ddot q + h_2= \tau \\
& J_i\ddot q + \dot J_i \dot q= \ddot x_i,
\end{align}
$$
although, we still want to substitute the cartesian acceleration and set $\dot J = 0$
$$
\begin{align}
\underset{\tau,\ddot q}{\text{minimise}} & \sum_i\|J_i\ddot q - \ddot x_{i,d}\|^2_{W_i}  \\ \text{subject to} \space & \nonumber \\
& - \tau + M_2\ddot q  = -h_2
\end{align}
$$
Let's add contact forces:
$$
\begin{align}
\underset{\tau,\ddot q, \lambda}{\text{minimise}} & \sum_i\|J_i\ddot q - \ddot x_{i,d}\|^2_{W_i}  \\ \text{subject to} \space & \nonumber \\
& M_1\ddot q -C^T\lambda = -h_1 \\
& - \tau + M_2\ddot q -C^T\lambda = -h_2
\end{align}
$$


### Notes
Single com task with $\ddot x_d$ set to 0, produces following torque command

```
array([ -0.   ,   0.   , 400.68 ,   0.   ,  -0.   ,   0.   ,  -0.   ,  -0.   ,  -0.   ,   0.   ,
        -0.   ,   0.   ,   0.   ,   0.   ,   0.   ,  -0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
        -0.   ,  -0.   ,  -0.   ,   0.   ,   0.   ,  -0.   ,  -0.001])
```

The PD task controller is missing at this stage. The desired acceleration should consider the gravity.


## Contact forces

Let's start considering contact forces to further improve robustness.
We add a penalty on the contact forces.

Now, we add polyhedral constraints.

