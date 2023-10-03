import time

import mujoco
import mujoco.viewer

import numpy as np
import scipy

from robot_descriptions.loaders.mujoco import load_robot_description

from helpers import *


np.set_printoptions(precision=3, suppress=True, linewidth=100)

pert = Perturbations([(2, 0.05), (5, 0.05)], 0)

# model = load_robot_description("gen3_mj_description")
model = mujoco.MjModel.from_xml_path(
    '/workdir/kinova_mj_description/xml/gen3_7dof_mujoco.xml')
# model = mujoco.MjModel.from_xml_path(
#     '3dof.xml')
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
#
# Get the center of mass of the body
ee_id = model.body('bracelet_link').id

nu = model.nu  # Alias for the number of actuators.
nv = model.nv  # Shortcut for the number of DoFs.
nq0 = model.nq - model.nu
nv0 = model.nv - model.nu
pmapu = [*range(nq0,nq0+nu, 1)]
vmapu = [*range(nv0,nv0+nu, 1)]
udof = np.ix_(vmapu,vmapu) # Controlled DoFs

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
x_c_d = data.subtree_com[ee_id].copy()
x_c_d[2] = 0.04
dx_c_d = np.zeros(3)
q_d = data.qpos[:nu].copy()
quat_d_ee = np.array([ 1, 0, 0, 0])

p0 = x_c_d
r = .1
f = 1
def circular_motion(t):
    w = 2*np.pi*f
    p_d = np.array([p0[0]+r*np.cos(w*t),p0[1]+ r*np.sin(w*t), p0[2]])
    v_d = np.array([-w*r*np.sin(w*t),w*r*np.cos(w*t),0])
    return (p_d, v_d)

# Task function
Kp_c = 10000
Kd_c = 1000
Kp_q = 0
Kd_q = 100
Kp_r = 1000
Kd_r = 100
 
def ddotx_c_d(p, v, p_d, v_d): 
    return -Kp_c * (p - p_d) - Kd_c * (v - v_d)

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
qp1 = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
qpproblem1 = QPProblem()

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
        J1 = Je[:,:nu]
        J2 = np.eye(nu,nu)
        J4 = Jebr[:,:nu]

        # Define References
        (x_d, v_d) = circular_motion(time.time()-start)
        ref1 = ddotx_c_d(x_c, dx_c, x_d, v_d)
        ref2 = ddotq_d(data.qpos[nq0:nq0+nu], data.qvel[nv0:nv0+nu])
        ref4 = ddotR_d(data.body(ee_id).xquat, angvel)

        setupQPDense(M1, J1, J2, J4, W1, W2, W3, W4, h1, ref1, ref2, ref4, nu, nforce, qp1, qpproblem1)
        qp1.solve()

        tau_d = qp1.results.x[:nu]
        force = qp1.results.x[nu:nu+nforce]

        data.ctrl = tau_d

        mujoco.mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
