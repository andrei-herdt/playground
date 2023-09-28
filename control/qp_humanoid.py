import time

import mujoco
import mujoco.viewer
import numpy as np
import scipy

from helpers import Perturbations, get_perturbation

np.set_printoptions(precision=3, suppress=True, linewidth=100)

pert = Perturbations([(2, 0.05), (5, 0.05)], 0)

model = mujoco.MjModel.from_xml_path(
    '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml')
data = mujoco.MjData(model)

model.opt.gravity = np.zeros(3)

nu = model.nu  # Alias for the number of actuators.
nv = model.nv  # Shortcut for the number of DoFs.

data = mujoco.MjData(model)

best_offset = -0.0005

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

# Get the Jacobian for the root body (torso) CoM.
Jc = np.zeros((3, nv))

M = np.zeros((model.nv, model.nv))
Minv = np.zeros((model.nv, model.nv))

# Task weights
W1 = 0*np.identity(3)
W2 = 1*np.identity(nu)
W3 = 100*np.identity(nu)

# Constants
x_c_d = data.subtree_com[0].copy()
q_d = data.qpos[7:].copy()

g = np.array([0, 0, 9.81])

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

sim_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Get state
        dx_c = data.subtree_linvel[0]
        x_c = data.subtree_com[0]

        # Get the mass matrix and the bias term
        mujoco.mj_fullM(model, M, data.qM)
        h = data.qfrc_bias

        M1 = M[6:,6:]
        h1 = h[6:]
        Minv = np.linalg.inv(M1)
        mujoco.mj_jacSubtreeCom(model, data, Jc, model.body('torso').id)
        J1 = Jc[:,6:]
        J2 = np.eye(nu,nu)
        H1 = Minv.T@J1.T@W1@J1@Minv
        H2 = Minv.T@J2.T@W2@J2@Minv
        Hpinv = np.linalg.pinv(H1 + H2 + W3)

        r1 = (J1@Minv@h1+ddotx_c_d(x_c, dx_c))@W1@J1@Minv
        r2 = (J2@Minv@h1+ddotq_d(data.qpos[7:], data.qvel[6:]))@W2@J2@Minv

        tau_d = Hpinv@(r1 + r2)
        print("data.ctrl", data.ctrl)
        print("tau_d", tau_d)
        data.ctrl = tau_d

        # Calculate 'dot J'
        # Given some qpos we need to first update the internal datastructures. I'm assuming all this happens within a rollout, so qpos has just been updated by mj_step but derived quantities have not. mj_forward will do this for us but it's enough to call mj_kinematics and mj_comPos.
        #
        # Call the relevant jac function and save the Jacobian in J.
        # Choose a very small positive number h. Anything in the range 
        # should give identical results.
        delta_t = 1e-3
        # Call mj_integratePos(m, d->qpos, d->qvel, h). This will integrate qpos in-place with the timestep h.
        qpos_bkp = data.qpos
        mujoco.mj_integratePos(model, data.qpos, data.qvel, delta_t)
        # Do step 1 again to update mjData kinematic quantities.
        mujoco.mj_kinematics(model,data)
        mujoco.mj_comPos(model,data)
        # Get the new Jacobian as in step 2, call it Jh.
        Jc_plus = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, Jc_plus, model.body('torso').id)
        # The quantity we want is Jdot = (Jh-J)/h.
        Jdot = (Jc_plus - Jc)/delta_t
        # Reset d->qpos to the original value, continue with the simulation. Kinematic quantities will be overwritten, no need to call kinematics and comPos again.
        data.qpos = qpos_bkp
        mujoco.mj_kinematics(model,data)
        mujoco.mj_comPos(model,data)

        # Compute com acceleration via:
        # \ddot c = J_c \ddot q_2 + \dot J_c \dot q_2
        ddot_c = Jc@data.qacc + Jdot@data.qvel

        print("ddotq_d: ", ddotq_d(data.qpos[7:], data.qvel[6:]))
        # input()
        print('\n')

        mujoco.mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
