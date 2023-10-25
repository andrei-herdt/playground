import time

import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData, \
    mj_resetDataKeyframe, mj_kinematics, \
    mj_comPos, mj_jacSite, mj_fullM, mj_step
import numpy as np
from helpers import initialize_zero_array, \
    get_ee_body_ids, QPProblem, initialize_box_constraints, \
    create_jacobians_dict, fill_jacobians_dict, Perturbations, \
    get_dynamics, create_figure, draw_vectors
from proxsuite import proxqp
from typing import List

# import two_manip_wheel_base as tf
# import humanoid as tf
# import quadruped as tf
import humanoid2 as tf

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/gen3_7dof_mujoco.xml')
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/manipulator_on_wheels.xml')
model = MjModel.from_xml_path(tf.xml_model_path)
# model.opt.gravity[2] = 0
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml')
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/wheel_base_with_deck.xml')
# model = mujoco.MjModel.from_xml_path(
#     '/workdir/playground/3rdparty/kinova_mj_description/xml/wheel_base.xml')
# model = mujoco.MjModel.from_xml_path('3dof.xml')
data = MjData(model)

mj_resetDataKeyframe(model, data, tf.key_frame_id)

# Alias for model properties
nu: int = model.nu
nv: int = model.nv
nq0 = tf.nq0
nv1 = tf.nv1
qpnv = nv1+nu

contacts = tf.get_list_of_contacts()
ncontacts = len(contacts)

# Generate actuator mappings
act_j_names = tf.get_actuated_names()
vmapu = tf.get_vmapu(act_j_names, model)
qmapu = tf.get_qmapu(act_j_names, model)
udof = np.ix_(vmapu, vmapu)

# maps from qp to physical
# todo: move to qpproblem
qpmapf: List[int] = [*range(nu+nv1+nu, nu+nv1+nu+3*ncontacts)]
qpmapq: List[int] = [*range(nu, nu+qpnv)]
qpmaptau: List[int] = [*range(0, nu)]

mj_kinematics(model, data)
mj_comPos(model, data)

# Jacobians
Jc = initialize_zero_array((3 * ncontacts, nv))

M = initialize_zero_array((nv, nv))

# Initialize task matrices
A1, A2, A4 = (initialize_zero_array((3, nu)) for _ in range(3))

weights = tf.create_weights(nv1, nu, ncontacts)
ee_names = tf.get_end_effector_names()
ee_ids = get_ee_body_ids(ee_names, model)
ref = tf.create_references_dict(data, ee_ids, qmapu)
gains = tf.create_gains_dict()

# Move to fill_jacobians_dict
for idx, name in enumerate(contacts):
    id: int = model.site(name).id
    Cflt, Cflr = (initialize_zero_array((3, nv)) for _ in range(2))
    mj_jacSite(model, data, Cflt, Cflr, id)
    Jc[3 * idx:3 * (idx + 1), :] = Cflt

mj_fullM(model, M, data.qM)

n = nu 
n_eq: int = 0
n_in: int = 0
qpp = QPProblem()

qp1 = proxqp.dense.QP(n, n_eq, n_in, True)
qp2 = proxqp.dense.QP(2*nu, nu, n_in, True)
nvar = nu+qpnv+3*ncontacts

# qpfull = proxqp.dense.QP(nvar, nv1+nu+nv1, n_in, True)
# qp = proxqp.dense.QP(nv1+2*nu+3*ncontacts, nv1+nu, n_in, True)
# Init box constraints
l_box, u_box = initialize_box_constraints(nvar)
# qpproblemfull.l_box = l_box
# qpproblemfull.u_box = u_box

# Avoid tilting
# tmp

idx_fx = [nu + qpnv + 3*i+0 for i in range(ncontacts)]
idx_fy = [nu + qpnv + 3*i+1 for i in range(ncontacts)]
idx_fz = [nu + qpnv + 3*i+2 for i in range(ncontacts)]
for idx in idx_fz:
    l_box[idx] = 0
# for idx in idx_fx:
#     l_box[idx] = -3
#     u_box[idx] = 3
# for idx in idx_fy:
#     l_box[idx] = -3
#     u_box[idx] = 3

nineq = len(idx_fx)
# ninex = 0
mu = 0.5
qp = proxqp.dense.QP(nvar, qpnv + 3*ncontacts, nineq, True)
qpp.l_box = l_box
qpp.u_box = u_box

qpp.C = np.zeros((nineq, nvar))
for i in range(nineq):
    qpp.C[i,idx_fx[i]] = 1
    qpp.C[i,idx_fz[i]] = -mu
qpp.l = -np.ones(nineq) * 1e8
qpp.u = np.zeros(nineq)

# set acc to zero for z,roll,pitch
# for i in range(2, 5):
#     qpp.l_box[nu + i] = 0
#     qpp.u_box[nu + i] = 0

Jebt, Jebr, Jebt_left, Jebr_left = (initialize_zero_array((3, nv)) for _ in range(4))

jacs = create_jacobians_dict(ee_ids, (3,nv))

sim_start = time.time()
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        fill_jacobians_dict(jacs, model, data)
        state = tf.get_state(data, ee_ids, jacs, qmapu, vmapu)
        dyn = get_dynamics(model, data, M, udof, vmapu, nv1)

        # Specific
        # J1 = jacs[ee_ids['ee']]['t'][:,vmapu]
        # J1_left = jacs[ee_ids['ee_left']]['t'][:,vmapu]
        # J4 = jacs[ee_ids['ee']]['r'][:,vmapu]
        # J4_left = jacs[ee_ids['ee_left']]['r'][:,vmapu]
        # J2 = np.eye(nu, nu)

        # Define References
        t = time.time() - start
        des_acc = tf.compute_des_acc(t, ref, gains, state, data, nu, nv1, vmapu)

        tf.setupQPSparseFullFullJacTwoArms(dyn['M1full'], dyn['M2full'], dyn['h1full'], dyn['h2full'], Jc, jacs, ee_ids, vmapu, weights, des_acc, nv1, nu, 3*ncontacts, qp, qpp)
        qp.solve()

        tau_d = qp.results.x[qpmaptau]
        forces = qp.results.x[qpmapf]
        ddq = qp.results.x[qpmapq]
        print("fx: ", qp.results.x[idx_fx])
        print("fy: ", qp.results.x[idx_fy])
        print("fz: ", qp.results.x[idx_fz])

        data.ctrl = tau_d

        mj_step(model, data)

        __import__('pdb').set_trace()
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
