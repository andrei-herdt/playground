import numpy as np
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

    # Store in a dictionary
    references_dict = {
        "com": x_c_d,
        "dx_c_d": dx_c_d,
        "q2_d": q2_d,
        "p_d_root": p_d_root,
        "R_d_root": R_d_root,
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

def get_end_effector_names(root_name):
    names: List[str] = [root_name]
    return names


def get_state(
    data,
    ee_ids: Dict[str, int],
    jacs: Dict[int, Dict[str, np.ndarray]],
    qmapu: np.ndarray,
    vmapu: np.ndarray,
    root_name: str
) -> Dict[str, Any]:
    """Retrieve the states for all end effectors and return in a single dictionary.

    Args:
        data: The simulation data.
        ee_ids (Dict[str, int]): A dictionary mapping end effector names to their IDs.
        jacs (Dict[int, Dict[str, np.ndarray]]): A dictionary containing Jacobians for the end effectors.

    Returns:
        Dict[str, Any]: A dictionary containing state information for all end effectors.
    """
    state = {
        "com": data.subtree_com[ee_ids[root_name]],
        "dcom": data.subtree_linvel[ee_ids[root_name]],
        "angvel": jacs[ee_ids[root_name]]["r"] @ data.qvel,
        "R_root": data.body(ee_ids[root_name]).xquat,
        "q2": data.qpos[qmapu],
        "v2": data.qvel[vmapu],
    }

    return state


def compute_des_acc(t, ref, gains, state, data, nu, nv1, vmapu):
    des_acc: Dict[str, np.ndarray] = {}

    r = 0.0
    f = 0.0
    (x_d, v_d) = circular_motion(t, ref["com"], r, f)
    des_acc["com"] = ddotx_c_d(
        state["com"], state["dcom"], x_d, v_d, gains["Kp_c"], gains["Kd_c"]
    )

    des_acc["joints"] = ddotq_d(
        state["q2"],
        state["v2"],
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
