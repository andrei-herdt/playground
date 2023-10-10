import time

import mujoco
import mujoco.viewer

import numpy as np
import scipy

from robot_descriptions.loaders.mujoco import load_robot_description

from helpers import *

import two_manip_wheel_base as tf

@dataclass
class Task:
    id: int
    name: str
    Kp: float
    Kd: float
    ref: np.ndarray
    W: np.ndarray
    J: np.ndarray

np.set_printoptions(precision=3, suppress=True, linewidth=100)

pert = Perturbations([(2, 0.05), (5, 0.05)], 0)

# model = load_robot_description("gen3_mj_description")
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/gen3_7dof_mujoco.xml')
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/manipulator_on_wheels.xml')
model = mujoco.MjModel.from_xml_path(
    '/workdir/playground/3rdparty/kinova_mj_description/xml/two_manipulator_on_wheels.xml')
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml')
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/wheel_base_with_deck.xml')
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/wheel_base.xml')
# model = mujoco.MjModel.from_xml_path('3dof.xml')
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

# Alias for model properties
nu: int = model.nu
nv: int = model.nv
nq0: int = model.nq - model.nu
nv0: int = model.nv - model.nu

# Generate actuator mappings
qmapu: List[int] = [*range(nq0, nq0 + nu)]
vmapu: List[int] = [*range(nv0, nv0 + nu)]
udof = np.ix_(vmapu, vmapu)

mujoco.mj_kinematics(model, data)
mujoco.mj_comPos(model, data)

# Jacobians
Je, Je_left, Jebt, Jebr, Jebt_left, Jebr_left = (initialize_zero_array((3, nv)) for _ in range(6))

contacts = tf.get_list_of_contacts()
ncontacts = len(contacts)
Ct = initialize_zero_array((3 * ncontacts, nv))


M = initialize_zero_array((nv, nv))

# Initialize task matrices
A1, A2, A4 = (initialize_zero_array((3, nu)) for _ in range(3))

# specific
#
weights = tf.create_weights(nv0, nu)

ee_names = tf.get_end_effector_names()
ee_ids = get_ee_body_ids(ee_names, model)

# References
x_c_d: np.ndarray = data.subtree_com[ee_ids['ee']].copy()
x_c_d_left: np.ndarray = data.subtree_com[ee_ids['ee_left']].copy()
dx_c_d: np.ndarray = np.zeros(3)
dx_c_d_left: np.ndarray = np.zeros(3)
q2_d: np.ndarray = data.qpos[qmapu].copy()
R_d_ee: np.ndarray = data.body(ee_ids['ee']).xquat.copy()
R_d_ee_left: np.ndarray = data.body(ee_ids['ee_left']).xquat.copy()
root_id: np.ndarray = model.body('wheel_base').id
p_d_root: np.ndarray = data.body(root_id).xpos.copy()
R_d_root: np.ndarray = data.body(root_id).xquat.copy()

# Task functio0000n
Kp_c: float = 10000
Kd_c: float = 1000
Kp_q: float = 0
Kd_q: float = 100
Kp_r: float = 1000
Kd_r: float = 100
#
# /specific
 
for idx, name in enumerate(contacts):
    id: int = model.site(name).id
    Cflt, Cflr = (initialize_zero_array((3, nv)) for _ in range(2))
    mujoco.mj_jacSite(model, data, Cflt, Cflr, id)
    Ct[3 * idx:3 * (idx + 1), :] = Cflt

mujoco.mj_fullM(model, M, data.qM)

n = nu 
n_eq: int = 0
n_in: int = 0
qpproblem1 = QPProblem()
qpproblem2 = QPProblem()
qpproblemfull = QPProblem()
qpproblemfullfulljac = QPProblem()

qp1 = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
qp2 = proxsuite.proxqp.dense.QP(2*nu, nu, n_in, True)
qpfull = proxsuite.proxqp.dense.QP(nv0+2*nu+3*ncontacts, nv0+nu+nv0, n_in, True)
qpfullfulljac = proxsuite.proxqp.dense.QP(nv0+2*nu+3*ncontacts, nv0+nu, n_in, True)

# Init box constraints
l_box, u_box = initialize_box_constraints(nv0 + 2*nu + 3*ncontacts)
qpproblemfullfulljac.l_box = l_box
qpproblemfullfulljac.u_box = u_box
qpproblemfull.l_box = l_box
qpproblemfull.u_box = u_box

# Specific
#
# Avoid tilting
idx_fz = [nu + nv0 + nu + i for i in [2, 5, 8, 11]]
for idx in idx_fz:
    l_box[idx] = 0
#
# /specific

qpproblemfull.l_box = l_box
qpproblemfullfulljac.l_box = l_box

# set acc to zero for z,roll,pitch
for i in range(2, 5):
    qpproblemfullfulljac.l_box[nu + i] = 0
    qpproblemfullfulljac.u_box[nu + i] = 0

sim_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Get Jacobians
        mujoco.mj_jacSubtreeCom(model, data, Je, ee_ids['ee'])
        mujoco.mj_jacSubtreeCom(model, data, Je_left, ee_ids['ee_left'])
        mujoco.mj_jacBody(model, data, Jebt, Jebr, ee_ids['ee'])
        mujoco.mj_jacBody(model, data, Jebt_left, Jebr_left, ee_ids['ee_left'])

        # Get state
        x_c = data.subtree_com[ee_ids['ee']]
        dx_c = data.subtree_linvel[ee_ids['ee']]
        angvel = Jebr@data.qvel

        x_c_left = data.subtree_com[ee_ids['ee_left']]
        dx_c_left = data.subtree_linvel[ee_ids['ee_left']]
        angvel_left = Jebr_left@data.qvel

        # Get the mass matrix and the bias term
        mujoco.mj_fullM(model, M, data.qM)
        h = data.qfrc_bias

        M2 = M[udof]
        h2 = h[vmapu]
        M1full = M[:nv0,:]
        h1full = h[:nv0]
        M2full = M[nv0:,:]
        h2full = h[nv0:]

        # Specific
        #
        J1 = Je[:,vmapu]
        J1_left = Je_left[:,vmapu]
        J2 = np.eye(nu,nu)
        J4 = Jebr[:,vmapu]
        J4_left = Jebr_left[:,vmapu]
        J1full = Je
        J1full_left = Je_left
        J4full = Jebr
        J4full_left = Jebr_left
        J2full = np.eye(nu+nv0,nu+nv0)

        # Define References
        r = 0.1
        f = 0.3
        (x_d, v_d) = circular_motion(time.time()-start, x_c_d, r, f)
        ref1 = ddotx_c_d(x_c, dx_c, x_d, v_d, Kp_c, Kd_c)
        (x_d, v_d) = circular_motion(time.time()-start, x_c_d_left, r, f, -np.pi)
        ref1_left = ddotx_c_d(x_c_left, dx_c_left, x_d, v_d, Kp_c, Kd_c)
        ref2 = ddotq_d(data.qpos[qmapu], data.qvel[vmapu], q2_d, np.zeros(nu), Kp_q, Kd_q)
        ref4 = ddotR_d(data.body(ee_ids['ee']).xquat, angvel, R_d_ee, np.zeros(3), Kp_r, Kd_r)
        ref4_left = ddotR_d(data.body(ee_ids['ee_left']).xquat, angvel_left, R_d_ee_left, np.zeros(3), Kp_r, Kd_r)
        r = .0
        f = .0
        (x_d, v_d) = circular_motion(time.time()-start, np.zeros(3), r, f)
        ref2full = ddotq_d_full(data.qpos, data.qvel, x_d, v_d, p_d_root, R_d_root, q2_d, np.zeros(nu+nv0), Kp_q, Kd_q)
        #
        # Specific

        setupQPDense(M2, J1, J2, J4, weights['W1'], weights['W2'], weights['W3'], weights['W4'], h2, ref1, ref2, ref4, nu, 0, qp1, qpproblem1)
        setupQPSparse(M2, J1, J2, J4, weights['W1'], weights['W2'], weights['W3'], weights['W4'], h2, ref1, ref2, ref4, nu, 0, qp2, qpproblem2)
        setupQPSparseFull(M1full, M2full, h1full, h2full, Ct, J1, J2, J4, weights['W1'], weights['W2'], weights['W3'], weights['W4'], ref1, ref2, ref4, nv0, nu, 3*ncontacts, qpfull, qpproblemfull)
        # setupQPSparseFullFullJac(M1full, M2full, h1full, h2full, Ct, J1full, J2full, J4full, W1, W2full, W3, W4, ref1, ref2full, ref4, nv0, nu, 3*ncontacts, qpfullfulljac, qpproblemfullfulljac)
        setupQPSparseFullFullJacTwoArms(M1full, M2full, h1full, h2full, Ct, J1full, J1full_left, J2full, J4full, J4full_left, weights['W1'], weights['W1_left'], weights['W2full'], weights['W3'], weights['W4'], weights['W4_left'], ref1, ref1_left, ref2full, ref4, ref4_left, nv0, nu, 3*ncontacts, qpfullfulljac, qpproblemfullfulljac)
        # qp1.solve()
        # qp2.solve()
        # qpfull.solve()
        qpfullfulljac.solve()

        tau_d = qpfullfulljac.results.x[:nu]

        data.ctrl = tau_d

        mujoco.mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
