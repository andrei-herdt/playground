import numpy as np
import scipy.sparse as spa
from typing import Dict, Any, List
from helpers import circular_motion, ddotR_d, ddotx_c_d, ddotq_d, ddotq_d_full

def get_vmapu(joint_names: List[str], model) -> List[int]:
    vmapu: List[int] = [0 for _ in range(len(joint_names))]
    for i in range(len(joint_names)):
        vmapu[i] = model.joint(joint_names[i]).dofadr[0]
    return vmapu

def get_qmapu(joint_names: List[str], model) -> List[int]:
    vmapu: List[int] = [0 for _ in range(len(joint_names))]
    for i in range(len(joint_names)):
        vmapu[i] = model.joint(joint_names[i]).qposadr[0]
    return vmapu

def create_gains_dict() -> Dict[str, float]:
    """
    Factory function to generate a gains dictionary.

    Returns:
    - Dictionary containing the gains.
    """
    # Define gains
    Kp_c: float = 1000
    Kd_c: float = 100
    Kp_q: float = 100
    Kd_q: float = 10
    Kp_r: float = 100
    Kd_r: float = 10
    # Store in a dictionary
    gains_dict = {
        "Kp_c": Kp_c,
        "Kd_c": Kd_c,
        "Kp_q": Kp_q,
        "Kd_q": Kd_q,
        "Kp_r": Kp_r,
        "Kd_r": Kd_r,
    }
    return gains_dict

def create_references_dict(data, ee_ids, qmapu, root_name) -> Dict[str, Any]:
    """
    Factory function to generate a references dictionary.

    Parameters:
    - data: The data source (needs to have methods like subtree_com, qpos, and body)
    - ee_ids: end effector ids
    - qmapu: Index or slice for the qpos method of the data source.

    Returns:
    - Dictionary containing the references.
    """

    # Create references
    x_c_d: np.ndarray = data.subtree_com[ee_ids[root_name]].copy()
    dx_c_d: np.ndarray = data.subtree_linvel[ee_ids[root_name]].copy()
    q2_d: np.ndarray = data.qpos[qmapu].copy()
    p_d_root: np.ndarray = data.body(ee_ids[root_name]).xpos.copy()
    R_d_root: np.ndarray = data.body(ee_ids[root_name]).xquat.copy()
    p_d_ee: np.ndarray = data.body(ee_ids["ee"]).xpos.copy()
    R_d_ee: np.ndarray = data.body(ee_ids["ee"]).xquat.copy()
    p_d_ee_left: np.ndarray = data.body(ee_ids["ee_left"]).xpos.copy()
    R_d_ee_left: np.ndarray = data.body(ee_ids["ee_left"]).xquat.copy()

    # Store in a dictionary
    references_dict = {
        "com": x_c_d,
        "dx_c_d": dx_c_d,
        "q2_d": q2_d,
        "p_d_root": p_d_root,
        "R_d_root": R_d_root,
        "p_d_ee": p_d_ee,
        "R_d_ee": R_d_ee,
        "p_d_ee_left": p_d_ee_left,
        "R_d_ee_left": R_d_ee_left,
    }

    return references_dict

def create_weights(nv1: int, nu: int, nc: int, root_name: str) -> dict:
    """
    Factory function to generate weights dictionary.

    Parameters:
    - nv1: int
    - nu: int
    - nc: int

    Returns:
    - Dictionary containing the weight arrays.
    """
    # Task weights
    com: np.ndarray = 1 * np.identity(3)  # EE pos task
    wq2: float = 0
    q2: np.ndarray = wq2 * np.identity(nu)  # ddq1,ddq2
    q: np.ndarray = np.zeros((nv1 + nu, nv1 + nu))  # ddq1,ddq2
    q[nv1:, nv1:] = 0 * np.identity(nu)  # ddq2
    tau: np.ndarray = 0.1 * np.identity(nu)  # tau
    trunk: np.ndarray = 1 * np.identity(3)  # EE orientation task
    forces: np.ndarray = 0.001 * np.identity(3 * nc)

    # Create and return the dictionary
    weights_dict = {
        "com": com,
        "q2": q2,
        "q": q,
        "tau": tau,
        root_name: trunk,
        "forces": forces,
    }
    return weights_dict

def get_task_states(
    data,
    ee_ids: Dict[str, int],
    jacs: Dict[int, Dict[str, np.ndarray]],
    qmapu: np.ndarray,
    vmapu: np.ndarray,
    root_name: str
) -> Dict[str, Dict[str, Any]]:
    task_states: Dict[str, Dict[str, Any]] = {}
    body_name = root_name
    state = {
        "p": data.subtree_com[ee_ids[body_name]],
        "dp": data.subtree_linvel[ee_ids[body_name]],
        "R": data.body(ee_ids[body_name]).xquat,
        "dR": jacs[ee_ids[body_name]]["r"] @ data.qvel,
    }
    task_states["com"] = state
    # ee
    body_name = "ee"
    state = {
        "p": data.body(ee_ids[body_name]).xpos,
        "dp": jacs[ee_ids[body_name]]["t"] @ data.qvel,
        "R": data.body(ee_ids[body_name]).xquat,
        "dR": jacs[ee_ids[body_name]]["r"] @ data.qvel,
    }
    task_states[body_name] = state
    # ee_left
    body_name = "ee_left"
    state = {
        "p": data.body(ee_ids[body_name]).xpos,
        "dp": jacs[ee_ids[body_name]]["t"] @ data.qvel,
        "R": data.body(ee_ids[body_name]).xquat,
        "dR": jacs[ee_ids[body_name]]["r"] @ data.qvel,
    }
    task_states[body_name] = state
    # posture
    state = {
        "q2": data.qpos[qmapu],
        "v2": data.qvel[vmapu],
    }
    task_states["posture"] = state

    return task_states

def compute_des_acc(t, ref, gains, task_states, data, nu, nv1, vmapu):
    des_acc: Dict[str, np.ndarray] = {}

    r = 0.0
    f = 0.0
    (x_d, v_d) = circular_motion(t, ref["com"], r, f)
    des_acc["com"] = ddotx_c_d(
        task_states["com"]["p"], 
        task_states["com"]["dp"], 
        x_d, v_d, gains["Kp_c"], gains["Kd_c"])
    des_acc['ee'] = ddotx_c_d(
        task_states["ee"]["p"], 
        task_states["ee"]["dp"], 
        x_d, v_d, gains['Kp_c'], gains['Kd_c'])
    des_acc['ee_left'] = ddotx_c_d(
        task_states["ee_left"]["p"], 
        task_states["ee_left"]["dp"], 
        x_d, v_d, gains['Kp_c'], gains['Kd_c'])
    des_acc['ee_R'] = ddotR_d(
        task_states["ee"]["R"], 
        task_states["ee"]["dR"], 
        ref['R_d_ee'], np.zeros(3), gains['Kp_r'], gains['Kd_r'])
    des_acc['ee_R_left'] = ddotR_d(
        task_states["ee_left"]["R"], 
        task_states["ee_left"]["dR"], 
        ref['R_d_ee_left'], np.zeros(3), gains['Kp_r'], gains['Kd_r'])

    des_acc["joints"] = ddotq_d(
        task_states["posture"]["q2"],
        task_states["posture"]["v2"],
        ref["q2_d"],
        np.zeros(nu),
        gains["Kp_q"],
        gains["Kd_q"],
    )
    # r = .0
    # f = .0
    # (x_d, v_d) = circular_motion(t, np.zeros(3), r, f)
    # des_acc['joints_full'] = ddotq_d_full(data.qpos, data.qvel, x_d, v_d, ref['p_d_root'], ref['R_d_root'], ref['q2_d'], np.zeros(nu+nv1), gains['Kp_q'], gains['Kd_q'], vmapu)
    des_acc["R_root"] = ddotR_d(
        data.qpos[3:7],
        data.qvel[3:6],
        ref["R_d_root"],
        np.zeros(3),
        gains["Kp_q"],
        gains["Kd_q"],
    )
    return des_acc


def setupQPDenseFullFullJacTwoArms(
    M1,
    M2,
    h1,
    h2,
    Jc,
    jacs,
    ee_ids,
    vmapu,
    weights,
    des_acc,
    nv1,
    nu,
    nforce,
    qp,
    qpproblem,
    root_name
):
    vmapv1 = [0, 1, 2, 3, 4, 5]
    ntau = nu
    # nv = M1.shape[1]
    nv = nu + nv1
    nc = Jc.shape[0]
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda]
    qpproblem.H = np.zeros((ntau + nv + nforce, ntau + nv + nforce))
    g = np.zeros(ntau + nv + nforce)

    vmap = vmapv1 + vmapu

    J1 = jacs[ee_ids[root_name]]["t"][:, vmap]
    # J2 = np.eye(nu+nv1,nu+nv1)[:,vmap]
    J4 = jacs[ee_ids[root_name]]["r"][:, vmap]
    J5 = np.eye(nu, nu)
    Jc = Jc[:, vmap]

    W1 = weights["com"]
    # W2 = weights['q']
    W3 = weights["tau"]
    W4 = weights[root_name]
    W5 = weights["q2"]
    W6 = weights["forces"]

    ref1 = des_acc["com"]
    # ref2 = des_acc['joints_full']
    ref4 = des_acc["R_root"]
    ref5 = des_acc["joints"]

    # [tau]
    qpproblem.H[:nu, :nu] += W3  # tau
    # [ddq1,ddq2]
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J1.T @ W1 @ J1  # ddq_2
    # qpqpproblem.H[ntau:ntau+nv, ntau:ntau+nv] += J2.T@W2@J2 # ddq
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J4.T @ W4 @ J4  # ddq
    # [ddq2]
    # ddq_2 = [x + nu for x in vmapu]
    # udof = np.ix_(ddq_2, ddq_2)
    # qpproblem.H[udof] += J5.T@W5@J5 # ddq_2
    qpproblem.H[ntau + nv1 : ntau + nv1 + nu, ntau + nv1 : ntau + nv1 + nu] += (
        J5.T @ W5 @ J5
    )  # ddq_2
    # [forces]
    qpproblem.H[ntau + nv : ntau + nv + nforce, ntau + nv : ntau + nv + nforce] += W6

    # tmp
    # qpproblem.H[nu:nu+6, nu:nu+6] += 0.001 * np.identity(6) # tau

    r1 = ref1 @ W1 @ J1
    # r2 = ref2@W2@J2
    r4 = ref4 @ W4 @ J4
    r5 = ref5 @ W5 @ J5

    g[ntau : ntau + nv] += r1  # ddq
    # g[ntau:ntau+nv] += r2 # ddq
    g[ntau : ntau + nv] += r4  # ddq
    # g[vmapu] += r5 # ddq
    g[ntau + nv1 : ntau + nv1 + nu] += r5  # ddq

    qpproblem.A = np.zeros((nv + nc, ntau + nv + nforce))
    qpproblem.b = np.zeros(nv + nc)

    # qpproblem.A[vmapu,0:ntau] += -np.eye(ntau,ntau) # tau
    # q1
    qpproblem.A[0:nv1, ntau : ntau + nv] += M1[:, vmap]  # ddq
    qpproblem.b[0:nv1] += -h1
    # q2
    qpproblem.A[nv1:nv, 0:ntau] += -np.eye(ntau, ntau)  # tau
    rows = [x - nv1 for x in vmapu]
    udof = np.ix_(rows, vmap)
    qpproblem.A[nv1:nv, ntau : ntau + nv] += M2[udof]  # ddq
    qpproblem.b[nv1:nv] += -h2[rows]
    # forces
    qpproblem.A[0:nv, ntau + nv :] += -Jc.T  # lambda
    # non-penetration
    qpproblem.A[nv : nv + nc, ntau : ntau + nv] += Jc  # lambda

    qp.init(
        qpproblem.H,
        -g,
        qpproblem.A,
        qpproblem.b,
        qpproblem.C,
        qpproblem.l,
        qpproblem.u,
        qpproblem.l_box,
        qpproblem.u_box,
    )

def setupQPSparseFullFullJacTwoArms(
    M1,
    M2,
    h1,
    h2,
    Jc,
    jacs,
    ee_ids,
    vmapu,
    weights,
    des_acc,
    nv1,
    nu,
    ncontacts,
    qp,
    qpproblem,
    root_name
):
    vmapv1 = [0, 1, 2, 3, 4, 5]
    ntau = nu
    # nv = M1.shape[1]
    nv = nu + nv1
    nc = Jc.shape[0]
    nforces = 3*ncontacts
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda]
    qpproblem.H = np.zeros((ntau + nv + nforces, ntau + nv + nforces))
    g = np.zeros(ntau + nv + nforces)

    vmap = vmapv1 + vmapu

    J1 = jacs[ee_ids[root_name]]["t"][:, vmap]
    J4 = jacs[ee_ids[root_name]]["r"][:, vmap]
    J5 = np.eye(nu, nu)
    Jc = Jc[:, vmap]

    J10 = jacs[ee_ids['ee']]['t']
    J11 = jacs[ee_ids['ee_left']]['t']
    J12 = jacs[ee_ids['ee']]['r']
    J13 = jacs[ee_ids['ee_left']]['r']

    W1 = weights["com"]
    W3 = weights["tau"]
    W4 = weights[root_name]
    W5 = weights["q2"]
    W6 = weights["forces"]

    W10 = W1
    W11 = W1
    W12 = W1
    W13 = W1

    ref1 = des_acc["com"]
    # ref2 = des_acc['joints_full']
    ref4 = des_acc["R_root"]
    ref5 = des_acc["joints"]

    ref10 = des_acc['ee']
    ref11 = des_acc['ee_left']
    ref12 = des_acc['ee_R']
    ref13 = des_acc['ee_R_left']

    # [tau]
    qpproblem.H[:nu, :nu] += W3  # tau
    # [ddq1,ddq2]
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J1.T @ W1 @ J1  # ddq_2
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J4.T @ W4 @ J4  # ddq

    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J10.T @ W10 @ J10  # ddq_2
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J11.T @ W11 @ J11  # ddq
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J12.T @ W12 @ J12  # ddq_2
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J13.T @ W13 @ J13  # ddq
    # [ddq2]
    qpproblem.H[ntau + nv1 : ntau + nv1 + nu, ntau + nv1 : ntau + nv1 + nu] += (
        J5.T @ W5 @ J5
    )  # ddq_2
    # [forces]
    qpproblem.H[ntau + nv : ntau + nv + nforces, ntau + nv : ntau + nv + nforces] += W6

    r1 = ref1 @ W1 @ J1
    r4 = ref4 @ W4 @ J4
    r5 = ref5 @ W5 @ J5

    r10 = ref10 @ W10 @ J10
    r11 = ref11 @ W11 @ J11
    r12 = ref12 @ W12 @ J12
    r13 = ref13 @ W13 @ J13

    g[ntau : ntau + nv] += r1  # ddq
    g[ntau : ntau + nv] += r4  # ddq
    g[ntau + nv1 : ntau + nv1 + nu] += r5  # ddq

    g[ntau : ntau + nv] += r10  # ddq
    g[ntau : ntau + nv] += r11  # ddq
    g[ntau : ntau + nv] += r12  # ddq
    g[ntau : ntau + nv] += r13  # ddq

    qpproblem.A = np.zeros((nv + nc, ntau + nv + nforces))
    qpproblem.b = np.zeros(nv + nc)

    # qpproblem.A[vmapu,0:ntau] += -np.eye(ntau,ntau) # tau
    # q1
    qpproblem.A[0:nv1, ntau : ntau + nv] += M1[:, vmap]  # ddq
    qpproblem.b[0:nv1] += -h1
    # q2
    qpproblem.A[nv1:nv, 0:ntau] += -np.eye(ntau, ntau)  # tau
    rows = [x - nv1 for x in vmapu]
    udof = np.ix_(rows, vmap)
    qpproblem.A[nv1:nv, ntau : ntau + nv] += M2[udof]  # ddq
    qpproblem.b[nv1:nv] += -h2[rows]
    # forces
    qpproblem.A[0:nv, ntau + nv :] += -Jc.T  # lambda
    # non-penetration
    qpproblem.A[nv : nv + nc, ntau : ntau + nv] += Jc  # lambda

    # Inequalities
    qpnv = nv1 + nu
    idx_fx = [nu + qpnv + 3 * i + 0 for i in range(ncontacts)]
    idx_fy = [nu + qpnv + 3 * i + 1 for i in range(ncontacts)]
    idx_fz = [nu + qpnv + 3 * i + 2 for i in range(ncontacts)]

    nineq = 3*len(idx_fx)
    mu = 0.5 # friction coefficient

    nvar = qp.model.dim
    qpproblem.C = np.zeros((nineq, nvar))
    qpproblem.l = np.zeros(nineq)
    qpproblem.u = np.zeros(nineq)
    for i in range(ncontacts):
        #x
        qpproblem.C[i, idx_fx[i]] = 1
        qpproblem.C[i, idx_fz[i]] = -mu
        qpproblem.l[i] = -1e8
        qpproblem.u[i] = 0
        #y
        qpproblem.C[2*i, idx_fy[i]] = 1
        qpproblem.C[2*i, idx_fz[i]] = -mu
        qpproblem.l[2*i] = -1e8
        qpproblem.u[2*i] = 0
        #z
        qpproblem.C[3*i, idx_fz[i]] = 1
        qpproblem.l[3*i] = 0
        qpproblem.u[3*i] = 1e8

    qp.init(
        spa.csc_matrix(qpproblem.H),
        -g,
        spa.csc_matrix(qpproblem.A),
        qpproblem.b,
        spa.csc_matrix(qpproblem.C),
        qpproblem.l,
        qpproblem.u,
    )
