import time

import mujoco
import mujoco.viewer
import numpy as np
import scipy

import proxsuite

from helpers import Perturbations, get_perturbation, calculateCoMAcc

np.set_printoptions(precision=3, suppress=True, linewidth=100)

pert = Perturbations([(2, 0.05), (5, 0.05)], 0)

model = mujoco.MjModel.from_xml_path(
    '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml')
data = mujoco.MjData(model)

from robot_descriptions.loaders.mujoco import load_robot_description
model = load_robot_description("gen3_mj_description")
nu = model.nu  # Alias for the number of actuators.
nv = model.nv  # Shortcut for the number of DoFs.

data = mujoco.MjData(model)

for i in range(nu):
    data.qpos[i] = 0.5
data.qpos[1] = np.pi/2
mujoco.mj_kinematics(model, data)
mujoco.mj_comPos(model, data)

# Get the Jacobian for the root body (torso) CoM.
Jc = np.zeros((3, nv))

M = np.zeros((model.nv, model.nv))
Minv = np.zeros((model.nv, model.nv))

# Task weights
W1 = 1*np.identity(3)
W2 = 1*np.identity(6)
W3 = .01*np.identity(nu)

# Constants
x_c_d = data.subtree_com[0].copy()
q_d = data.qpos[:6].copy()

g = np.array([0, 0, 9.81])

# Task function
Kp_c = 1000
Kd_c = 10
Kp_q = 0
Kd_q = 10

def ddotx_c_d(p, v): 
    return -Kp_c * (p - x_c_d) - Kd_c * (v - np.zeros(3))

def ddotq_d(p, v): 
    return -Kp_q * (p - q_d) - Kd_q * (v - np.zeros(6)) 

mujoco.mj_fullM(model, M, data.qM)

n = nu-1
n_eq = 0
n_in = 0
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
A = None
b = None
C = None
u = None
l = None
l_box = -np.ones(n) * 1.0e2
u_box = np.ones(n) * 1.0e2

__import__('pdb').set_trace()

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

        M1 = M[:6,:6]
        h1 = h[:6]
        Minv = np.linalg.inv(M1)
        mujoco.mj_jacSubtreeCom(model, data, Jc, model.body('base_link').id)
        J1 = Jc[:,:6]
        J2 = np.eye(6,6)
        H1 = Minv.T@J1.T@W1@J1@Minv
        H2 = Minv.T@J2.T@W2@J2@Minv
        H = H1 + H2 + W3[:6,:6]
        Hpinv = np.linalg.pinv(H)

        r1 = (J1@Minv@h1+ddotx_c_d(x_c, dx_c))@W1@J1@Minv
        r2 = (J2@Minv@h1+ddotq_d(data.qpos[:6], data.qvel[:6]))@W2@J2@Minv
        g = r1 + r2

        qp.init(H, -g, A, b, C, l, u, l_box, u_box)
        qp.solve()

        tau_d = qp.results.x

        data.ctrl[:6] = tau_d

        print(tau_d / qp.results.x)
        print(tau_d)
        print(qp.results.x)
        print('\n')

        mujoco.mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
