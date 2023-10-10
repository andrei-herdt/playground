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
contacts = tf.get_list_of_contacts()
ncontacts = len(contacts)
Ct = initialize_zero_array((3 * ncontacts, nv))

M = initialize_zero_array((nv, nv))

# Initialize task matrices
A1, A2, A4 = (initialize_zero_array((3, nu)) for _ in range(3))

weights = tf.create_weights(nv0, nu)
ee_names = tf.get_end_effector_names()
ee_ids = get_ee_body_ids(ee_names, model)
ref = tf.create_references_dict(data, ee_ids, qmapu)
gains = tf.create_gains_dict()

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

Jebt, Jebr, Jebt_left, Jebr_left = (initialize_zero_array((3, nv)) for _ in range(4))

def create_jacobians_dict(ee_ids: Dict[str, int], shape) -> Dict[str, Dict[str, Any]]:
    jacobians = {}
    for _, id in ee_ids.items():
        jacobians[id] = {
            't': np.zeros(shape),
            'r': np.zeros(shape)
        }
    return jacobians

def fill_jacobians_dict(jacobians: Dict[str, Dict[str, Any]]):
    for id, jac in jacobians.items():
        mujoco.mj_jacBody(model, data, jac['t'], jac['r'], id)

jacs = create_jacobians_dict(ee_ids, (3,nv))

refs: Dict[str, np.ndarray] = {}

sim_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        fill_jacobians_dict(jacs)

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
        J1 = jacs[ee_ids['ee']]['t'][:,vmapu]
        J1_left = jacs[ee_ids['ee_left']]['t'][:,vmapu]
        J4 = jacs[ee_ids['ee']]['r'][:,vmapu]
        J4_left = jacs[ee_ids['ee_left']]['r'][:,vmapu]
        J2 = np.eye(nu,nu)
        J2full = np.eye(nu+nv0,nu+nv0)

        # Define References
        r = 0.1
        f = 0.3
        (x_d, v_d) = circular_motion(time.time()-start, ref['x_c_d'], r, f)
        refs['ee'] = ddotx_c_d(x_c, dx_c, x_d, v_d, gains['Kp_c'], gains['Kd_c'])
        (x_d, v_d) = circular_motion(time.time()-start, ref['x_c_d_left'], r, f, -np.pi)
        refs['ee_left'] = ddotx_c_d(x_c_left, dx_c_left, x_d, v_d, gains['Kp_c'], gains['Kd_c'])
        refs['joints'] = ddotq_d(data.qpos[qmapu], data.qvel[vmapu], ref['q2_d'], np.zeros(nu), gains['Kp_q'], gains['Kd_q'])
        refs['ee_R'] = ddotR_d(data.body(ee_ids['ee']).xquat, angvel, ref['R_d_ee'], np.zeros(3), gains['Kp_r'], gains['Kd_r'])
        refs['ee_R_left'] = ddotR_d(data.body(ee_ids['ee_left']).xquat, angvel_left, ref['R_d_ee_left'], np.zeros(3), gains['Kp_r'], gains['Kd_r'])
        r = .0
        f = .0
        (x_d, v_d) = circular_motion(time.time()-start, np.zeros(3), r, f)
        refs['joints_full'] = ddotq_d_full(data.qpos, data.qvel, x_d, v_d, ref['p_d_root'], ref['R_d_root'], ref['q2_d'], np.zeros(nu+nv0), gains['Kp_q'], gains['Kd_q'])
        #
        # Specific

        setupQPDense(M2, J1, J2, J4, weights['W1'], weights['W2'], weights['W3'], weights['W4'], h2, refs['ee'], refs['joints'], refs['ee_R'], nu, 0, qp1, qpproblem1)
        setupQPSparse(M2, J1, J2, J4, weights['W1'], weights['W2'], weights['W3'], weights['W4'], h2, refs['ee'], refs['joints'], refs['ee_R'], nu, 0, qp2, qpproblem2)
        setupQPSparseFull(M1full, M2full, h1full, h2full, Ct, J1, J2, J4, weights['W1'], weights['W2'], weights['W3'], weights['W4'], refs['ee'], refs['joints'], refs['ee_R'], nv0, nu, 3*ncontacts, qpfull, qpproblemfull)
        # setupQPSparseFullFullJac(M1full, M2full, h1full, h2full, Ct, Jebt, J2full, Jebr, W1, W2full, W3, W4, refs['ee'], refs['joints_full'], refs['ee_R'], nv0, nu, 3*ncontacts, qpfullfulljac, qpproblemfullfulljac)
        setupQPSparseFullFullJacTwoArms(M1full, M2full, h1full, h2full, Ct, jacs, ee_ids, vmapu, J2full, weights, refs, nv0, nu, 3*ncontacts, qpfullfulljac, qpproblemfullfulljac)
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
