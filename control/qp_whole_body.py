import time

import mujoco
import mujoco.viewer
import numpy as np
import scipy

from helpers import Perturbations, get_perturbation

np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = mujoco.MjModel.from_xml_path(
    '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml')
data = mujoco.MjData(model)

height_offsets = np.linspace(-0.001, 0.001, 2001)
vertical_forces = []
for offset in height_offsets:
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    # Offset the height by `offset`.
    data.qpos[2] += offset
    mujoco.mj_inverse(model, data)
    vertical_forces.append(data.qfrc_inverse[2])

# Find the height-offset at which the vertical force is smallest.
idx = np.argmin(np.abs(vertical_forces))
best_offset = height_offsets[idx]

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

nu = model.nu  # Alias for the number of actuators.
R = np.eye(nu)

nv = model.nv  # Shortcut for the number of DoFs.

# Get the Jacobian for the root body (torso) CoM.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
mujoco.mj_forward(model, data)
Jc = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, Jc, model.body('torso').id)

# Get the Jacobian for the left foot.
Jfl = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, Jfl, None, model.body('foot_left').id)
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
jac_diff = Jc - Jfl
Qbalance = jac_diff.T @ jac_diff

# Get all joint names.
joint_names = [model.joint(i).name for i in range(model.njnt)]

# Get indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
abdomen_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'abdomen' in name
    and 'z' not in name
]
left_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'left' in name
    and ('hip' in name or 'knee' in name or 'ankle' in name)
    and 'z' not in name
]
balance_dofs = abdomen_dofs + left_leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

# Cost coefficients.
BALANCE_COST = 1000  # Balancing.
BALANCE_JOINT_COST = 3     # Joints required for balancing.
OTHER_JOINT_COST = .3    # Other joints.

# Construct the Qjoint matrix.
Qjoint = np.eye(nv)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
Qpos = BALANCE_COST * Qbalance + Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])


# Set the initial state and control.
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0

# Allocate the A and B matrices, compute them.
A = np.zeros((2*nv, 2*nv))
B = np.zeros((2*nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

# Allocate position difference dq.
dq = np.zeros(model.nv)

pert = Perturbations([(2, 0.05), (5, 0.1)], 0)

M = np.zeros((model.nv, model.nv))
Minv = np.zeros((model.nv, model.nv))

# Get the mass matrix and the bias term
mujoco.mj_fullM(model, M, data.qM)
M2 = M[6:,6:]
M2inv = np.linalg.inv(M2)
h2 = data.qfrc_bias[6:]

J1 = Jc[:,6:]
J2 = Jfl[:,6:]

H1 = M2inv.T@J1.T@J1@M2inv
H2 = M2inv.T@J2.T@J2@M2inv
Hpinv = np.linalg.pinv(H1 + H2)

# Define task acceleration as pd control output
Kp_c = 1
Kd_c = 0.1
x_c_init = data.subtree_com[0]
ddotx_c_d = lambda p, v : Kp_c * (p - x_c_init) + Kd_c * (v - np.zeros(3))

dx_c = data.subtree_linvel[0]
r1 = (J1@M2inv@h2+ddotx_c_d(x_c_init, dx_c))@J1@M2inv

id_fl = model.body('foot_left').id
x_fl_init = data.subtree_com[id_fl]
dx_fl = data.subtree_linvel[id_fl]
r2 = (J2@M2inv@h2+ddotx_c_d(x_fl_init, dx_fl))@J2@M2inv

tau_d = Hpinv@(r1 + r2)


sim_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # Get state difference dx.
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T


        # LQR control law.
        data.ctrl = ctrl0 - K @ dx

        data.qvel[0] += get_perturbation(pert, step_start-sim_start)

        mujoco.mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
