import time

import mujoco
import mujoco.viewer
import numpy as np
import scipy

from helpers import Perturbations, get_perturbation, calculateCoMAcc

np.set_printoptions(precision=3, suppress=True, linewidth=100)

pert = Perturbations([(2, 0.05), (5, 0.05)], 0)

model = mujoco.MjModel.from_xml_path(
    '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml')

#tmp
model.opt.gravity = np.zeros(3)
data = mujoco.MjData(model)

nu = model.nu  # Alias for the number of actuators.
nv = model.nv  # Shortcut for the number of DoFs.
pmapu = [*range(7,7+nu, 1)]
vmapu = [*range(6,6+nu, 1)]
ugrid = np.ix_(vmapu,vmapu)

data = mujoco.MjData(model)

#tmp
best_offset = -0.0005
best_offset = +0.1

mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += best_offset
qpos0 = data.qpos.copy()  # Save the position setpoint.
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()

ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.

data.ctrl = ctrl0
mujoco.mj_forward(model, data)

# Allocate
Jc = np.zeros((3, nv))
M = np.zeros((nv, nv))
Minv = np.zeros((nv, nv))

# Task weights
W1 = 0*np.identity(3)
W2 = 1*np.identity(nu)
W3 = 1*np.identity(nu)

# Constants
x_c_d = data.subtree_com[0].copy()
q_d = data.qpos[7:].copy()

# Task function
Kp_c = 1000
Kd_c = 100
Kp_q = 1
Kd_q = .1

def ddotx_c_d(p, v): 
    print("e_p: ",p-x_c_d)
    return -Kp_c * (p - x_c_d) - Kd_c * (v - np.zeros(3))

def ddotq_d(p, v): 
    print("e_p: ",p-q_d)
    return -Kp_q * (p - q_d) - Kd_q * (v - np.zeros(nu)) 

mujoco.mj_fullM(model, M, data.qM)

#tmp
data.qacc = 0

mujoco.mj_forward(model, data)

#tmp
__import__('pdb').set_trace()

sim_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Get state
        x_c = data.subtree_com[0]
        dx_c = data.subtree_linvel[0]

        # Get the mass matrix and the bias term
        mujoco.mj_fullM(model, M, data.qM)
        h = data.qfrc_bias

        M1 = M[ugrid]
        h1 = h[vmapu]
        Minv = np.linalg.inv(M1)
        mujoco.mj_jacSubtreeCom(model, data, Jc, model.body('torso').id)
        J1 = Jc[:,vmapu]
        J2 = np.eye(nu,nu)
        H1 = Minv.T@J1.T@W1@J1@Minv
        H2 = Minv.T@J2.T@W2@J2@Minv
        H = H1+H2+W3
        print("cond(H)", np.linalg.cond(H))
        Hpinv = np.linalg.pinv(H)

        print(J2@Minv@h1)
        r1 = (J1@Minv@h1+ddotx_c_d(x_c, dx_c))@W1@J1@Minv
        # r2 = (J2@Minv@h1+ddotq_d(data.qpos[pmapu], data.qvel[vmapu]))@W2@J2@Minv
        r2 = (ddotq_d(data.qpos[pmapu], data.qvel[vmapu]))@W2@J2@Minv

        tau_d = Hpinv@(r1 + r2)
        print("data.ctrl", data.ctrl)
        print("tau_d", tau_d)
        data.ctrl = tau_d

        ddotc = calculateCoMAcc(model, data)

        print("ddotq_d: ", ddotq_d(data.qpos[pmapu], data.qvel[vmapu]))

        mujoco.mj_step(model, data)
        print("qacc: ", data.qacc)
        # input()
        print('\n')

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
