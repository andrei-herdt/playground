import time

import mujoco
import mujoco.viewer

import numpy as np
import scipy

import proxsuite
from robot_descriptions.loaders.mujoco import load_robot_description

from helpers import Perturbations, get_perturbation, calculateCoMAcc


np.set_printoptions(precision=3, suppress=True, linewidth=100)

pert = Perturbations([(2, 0.05), (5, 0.05)], 0)

# model = load_robot_description("gen3_mj_description")
model = mujoco.MjModel.from_xml_path(
    '/workdir/kinova_mj_description/xml/gen3_7dof_mujoco.xml')
# model = mujoco.MjModel.from_xml_path(
#     '3dof.xml')
data = mujoco.MjData(model)
#
# Get the center of mass of the body
ee_id = model.body('bracelet_link').id
ee_com = data.subtree_com[ee_id]

nu = model.nu  # Alias for the number of actuators.
nv = model.nv  # Shortcut for the number of DoFs.
nq0 = model.nq - model.nu
nv0 = model.nv - model.nu
pmapu = [*range(nq0,nq0+nu, 1)]
vmapu = [*range(nv0,nv0+nu, 1)]
udof = np.ix_(vmapu,vmapu) # Controlled DoFs

# for i in range(nu):
#     data.qpos[i] = 0.5
# data.qpos[1] = np.pi/2
mujoco.mj_kinematics(model, data)
mujoco.mj_comPos(model, data)

# Get the Jacobian for the root body (torso) CoM.
Je = np.zeros((3, nv))
Jebt = np.zeros((3, nv))
Jebr = np.zeros((3, nv))

M = np.zeros((model.nv, model.nv))
Minv = np.zeros((model.nv, model.nv))

nforce = 0
A1 = np.zeros((3, nu+nforce))
A2 = np.zeros((nu, nu+nforce))
A4 = np.zeros((3, nu+nforce))

# Task weights
W1 = 10*np.identity(3)
W2 = 1*np.identity(nu)
W3 = .01*np.identity(nu+nforce)
W4 = 10*np.identity(3)

# References
x_c_d = data.subtree_com[0].copy()
x_c_d[0] = x_c_d[0]+0.5
x_c_d[2] = 0.1
dx_c_d = np.zeros(3)
q_d = data.qpos[:nu].copy()
quat_d_ee = np.array([ 1, 0, 0, 0])

r = 0.3
f = 5
def circular_motion(t):
    return np.array([r*np.cos(f*t), r*np.sin(f*t), 0])

# Task function
Kp_c = 1000
Kd_c = 100
Kp_q = 0
Kd_q = 100
Kp_r = 1000
Kd_r = 100
 
def ddotx_c_d(p, v): 
    return -Kp_c * (p - x_c_d) - Kd_c * (v - dx_c_d)

def ddotq_d(p, v): 
    return -Kp_q * (p - q_d) - Kd_q * (v - np.zeros(nu)) 

def ddotR_d(p, v): 
    angerr = np.zeros(3)
    mujoco.mju_subQuat(angerr, p, quat_d_ee)
    return -Kp_r * angerr - Kd_r * (v - np.zeros(3)) 

mujoco.mj_fullM(model, M, data.qM)

n = nu + nforce
n_eq = 0
n_in = 0
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
A = None
b = None
C = None
u = None
l = None
l_box = -np.ones(n) * 1.0e3
u_box = np.ones(n) * 1.0e3

sim_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Get Jacobians
        mujoco.mj_jacSubtreeCom(model, data, Je, ee_id)
        mujoco.mj_jacBody(model, data, Jebt, Jebr, ee_id)

        # Get state
        dx_c = data.subtree_linvel[ee_id]
        x_c = data.subtree_com[ee_id]
        angvel = Jebr@data.qvel

        # Get the mass matrix and the bias term
        mujoco.mj_fullM(model, M, data.qM)
        h = data.qfrc_bias

        M1 = M[nq0:nq0+nu,nq0:nq0+nu]
        h1 = h[nq0:nq0+nu]
        Minv = np.linalg.inv(M1)
        J1 = Je[:,:nu]
        J2 = np.eye(nu,nu)
        J4 = Jebr[:,:nu]
        # todo: double check J1.T
        A1[:,:nu] = J1@Minv
        A2[:,:nu] = J2@Minv
        # A1[:,6:] = J1@Minv@J1.T
        # A2[:,6:] = J2@Minv@J1.T
        A4[:,:nu] = J4@Minv
        # A4[:,6:] = J4@Minv@J1.T
        H1 = A1.T@W1@A1
        H2 = A2.T@W2@A2
        H4 = A4.T@W4@A4
        H = H1 + H2 + W3[:nu+nforce,:nu+nforce] + H4 
        Hpinv = np.linalg.pinv(H)


        dx_d = circular_motion(time.time()-start)
        r1 = (A1[:,:nu]@h1 + ddotx_c_d(x_c, dx_c+dx_d))@W1@A1
        r2 = (A2[:,:nu]@h1 + ddotq_d(data.qpos[nq0:nq0+nu], data.qvel[nv0:nv0+nu]))@W2@A2
        r4 = (A4[:,:nu]@h1 + ddotR_d(data.body(ee_id).xquat, angvel))@W4@A4

        g = r1 + r2 + r4

        qp.init(H, -g, A, b, C, l, u, l_box, u_box)
        qp.solve()

        tau_d = qp.results.x[:nu]
        force = qp.results.x[nu:nu+nforce]

        print("quat",data.body(ee_id).xquat)
        print("angvel",angvel)
        print("taud_d",tau_d)

        data.ctrl = tau_d

        mujoco.mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
