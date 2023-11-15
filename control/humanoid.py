import numpy as np
import scipy.sparse as spa
from typing import Dict, Any, List

# from control.robotis_op3 import get_end_effector_names
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


def create_references_dict(data, ee_ids, qmapu, robot) -> Dict[str, Any]:
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
    com: np.ndarray = data.subtree_com[ee_ids[robot.root_name]].copy()
    q2_d: np.ndarray = data.qpos[qmapu].copy()

    # Store in a dictionary
    references_dict = {
        "com": com,
        "q2_d": q2_d,
    }

    # SE3 references
    ees = robot.get_end_effector_names()
    for ee_name in ees:
        references_dict[ee_name + "_p"] = data.body(ee_ids[ee_name]).xpos.copy()
        references_dict[ee_name + "_R"] = data.body(ee_ids[ee_name]).xquat.copy()

    return references_dict


# todo: Move
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
    ee_p: np.ndarray = 1 * np.identity(3)  # EE pos task
    ee_R: np.ndarray = .1 * np.identity(3)  # EE pos task
    ee_left_p: np.ndarray = 1 * np.identity(3)  # EE pos task
    ee_left_R: np.ndarray = .1 * np.identity(3)  # EE pos task
    root_name_p: np.ndarray = 0 * np.identity(3)  # EE orientation task
    root_name_R: np.ndarray = 0 * np.identity(3)  # EE orientation task

    com: np.ndarray = 0 * np.identity(3)  # EE pos task
    q2: np.ndarray = .1 * np.identity(nu)  # ddq1,ddq2
    q: np.ndarray = np.zeros((nv1 + nu, nv1 + nu))  # ddq1,ddq2
    q[nv1:, nv1:] = 0 * np.identity(nu)  # ddq2
    tau: np.ndarray = 0.001 * np.identity(nu)  # tau
    forces: np.ndarray = 0.001 * np.identity(3 * nc)

    # Create and return the dictionary
    weights_dict = {
        "com": com,
        "ee_p": ee_p,
        "ee_R": ee_R,
        "ee_left_p": ee_left_p,
        "ee_left_R": ee_left_R,
        "r_el_link_p": ee_p,
        "r_el_link_R": ee_R,
        "l_el_link_p": ee_left_p,
        "l_el_link_R": ee_left_R,
        root_name + "_p": root_name_p,
        root_name + "_R": root_name_R,
        "q2": q2,
        "q": q,
        "tau": tau,
        "forces": forces,
    }
    return weights_dict


def get_task_states(
    data,
    ee_ids: Dict[str, int],
    jacs: Dict[int, Dict[str, np.ndarray]],
    qmapu: np.ndarray,
    vmapu: np.ndarray,
    robot: Any,
) -> Dict[str, Dict[str, Any]]:
    task_states: Dict[str, Dict[str, Any]] = {}

    for body_name in robot.get_end_effector_names():
        state = {
            "p": data.body(ee_ids[body_name]).xpos,
            "dp": jacs[body_name]["t"] @ data.qvel,
            "R": data.body(ee_ids[body_name]).xquat,
            "dR": jacs[body_name]["r"] @ data.qvel,
        }
        task_states[body_name] = state

    # com
    state = {
        "p": data.subtree_com[ee_ids[robot.root_name]],
        "dp": data.subtree_linvel[ee_ids[robot.root_name]],
    }
    task_states["com"] = state
    # posture
    state = {
        "q2": data.qpos[qmapu],
        "v2": data.qvel[vmapu],
    }
    task_states["posture"] = state

    return task_states


def compute_des_acc(t, ref, gains, task_states, data, nu, nv1, vmapu, robot):
    des_acc: Dict[str, np.ndarray] = {}

    # com
    # TODO: Get the reference generation out of here
    r = 0.0
    f = 0.0
    (x_d, v_d) = circular_motion(t, ref["com"], r, f)
    des_acc["com"] = ddotx_c_d(
        task_states["com"]["p"],
        task_states["com"]["dp"],
        x_d,
        v_d,
        gains["Kp_c"],
        gains["Kd_c"],
    )

    # end effectors
    for body_name in robot.get_end_effector_names():
        r = 0.0
        f = 0.0
        if not body_name == robot.root_name:
            r = 0.0
            f = 0.0
        (x_d, v_d) = circular_motion(t, ref[body_name + "_p"], r, f)
        des_acc[body_name + "_p"] = ddotx_c_d(
            task_states[body_name]["p"],
            task_states[body_name]["dp"],
            x_d,
            v_d,
            gains["Kp_c"],
            gains["Kd_c"],
        )
        des_acc[body_name + "_R"] = ddotR_d(
            task_states[body_name]["R"],
            task_states[body_name]["dR"],
            ref[body_name + "_R"],
            np.zeros(3),
            gains["Kp_r"],
            gains["Kd_r"],
        )

    # posture
    des_acc["q2"] = ddotq_d(
        task_states["posture"]["q2"],
        task_states["posture"]["v2"],
        ref["q2_d"],
        np.zeros(nu),
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
    ncontacts,
    qp,
    qpproblem,
    robot,
):
    vmapv1 = [0, 1, 2, 3, 4, 5]
    ntau = nu
    # nv = M1.shape[1]
    nv = nu + nv1
    nc = Jc.shape[0]
    nforces = 3 * ncontacts
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda]
    qpproblem.H = np.zeros((ntau + nv + nforces, ntau + nv + nforces))
    g = np.zeros(ntau + nv + nforces)

    vmap = vmapv1 + vmapu

    J1 = jacs["com"]["t"][:, vmap]
    J5 = np.eye(nu, nu)
    Jc = Jc[:, vmap]

    W1 = weights["com"]
    W3 = weights["tau"]
    W5 = weights["q2"]
    W6 = weights["forces"]

    ref1 = des_acc["com"]
    ref5 = des_acc["q2"]

    # [tau]
    qpproblem.H[:nu, :nu] += W3  # tau
    # [ddq1,ddq2]
    qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += J1.T @ W1 @ J1  # ddq_2

    # [ddq2]
    qpproblem.H[ntau + nv1 : ntau + nv1 + nu, ntau + nv1 : ntau + nv1 + nu] += (
        J5.T @ W5 @ J5
    )  # ddq_2
    # [forces]
    qpproblem.H[ntau + nv : ntau + nv + nforces, ntau + nv : ntau + nv + nforces] += W6

    r1 = ref1 @ W1 @ J1
    r5 = ref5 @ W5 @ J5

    g[ntau : ntau + nv] += r1  # ddq
    g[ntau + nv1 : ntau + nv1 + nu] += r5  # ddq

    # SE3 tasks
    for body_name in robot.get_end_effector_names():
        Jt = jacs[body_name]["t"][:, vmap]
        Jr = jacs[body_name]["r"][:, vmap]
        Wt = weights[body_name + "_p"]
        Wr = weights[body_name + "_R"]
        qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += Jt.T @ Wt @ Jt
        qpproblem.H[ntau : ntau + nv, ntau : ntau + nv] += Jr.T @ Wr @ Jr
        reft = des_acc[body_name + "_p"]
        refr = des_acc[body_name + "_R"]
        rt = reft @ Wt @ Jt
        rr = refr @ Wr @ Jr
        g[ntau : ntau + nv] += rt  # ddq
        g[ntau : ntau + nv] += rr  # ddq

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

    nineq = 3 * len(idx_fx)
    mu = 0.5  # friction coefficient

    nvar = qp.model.dim
    qpproblem.C = np.zeros((nineq, nvar))
    qpproblem.l = np.zeros(nineq)
    qpproblem.u = np.zeros(nineq)
    for i in range(ncontacts):
        # x
        qpproblem.C[i, idx_fx[i]] = 1
        qpproblem.C[i, idx_fz[i]] = -mu
        qpproblem.l[i] = -1e8
        qpproblem.u[i] = 0
        # y
        qpproblem.C[2 * i, idx_fy[i]] = 1
        qpproblem.C[2 * i, idx_fz[i]] = -mu
        qpproblem.l[2 * i] = -1e8
        qpproblem.u[2 * i] = 0
        # z
        qpproblem.C[3 * i, idx_fz[i]] = 1
        qpproblem.l[3 * i] = 0
        qpproblem.u[3 * i] = 1e8

    qp.init(
        spa.csc_matrix(qpproblem.H),
        -g,
        spa.csc_matrix(qpproblem.A),
        qpproblem.b,
        spa.csc_matrix(qpproblem.C),
        qpproblem.l,
        qpproblem.u,
    )
