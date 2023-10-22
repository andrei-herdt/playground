import numpy as np
from typing import Dict, Any, List
from helpers import circular_motion, ddotx_c_d, ddotq_d, ddotq_d_full

# xml_model_path: str = '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml'
# key_frame_id: int = 0
xml_model_path: str = '/workdir/playground/3rdparty/mujoco_menagerie/agility_cassie/scene.xml'
key_frame_id: int = 0
nq0 = 7
nv1 = 6

def get_actuated_names() -> List[str]:
    joint_names: List[str] = ["left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-foot",
        "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-foot"]
    return joint_names

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
    Kp_c: float = 100
    Kd_c: float = 10
    Kp_q: float = 100
    Kd_q: float = 10
    Kp_r: float = 0
    Kd_r: float = 0
    # Store in a dictionary
    gains_dict = {
        'Kp_c': Kp_c,
        'Kd_c': Kd_c,
        'Kp_q': Kp_q,
        'Kd_q': Kd_q,
        'Kp_r': Kp_r,
        'Kd_r': Kd_r
    }
    return gains_dict

def create_references_dict(data, ee_ids, qmapu) -> Dict[str, Any]:
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
    x_c_d: np.ndarray = data.subtree_com[ee_ids['torso']].copy()
    dx_c_d: np.ndarray = data.subtree_linvel[ee_ids['torso']].copy()
    q2_d: np.ndarray = data.qpos[qmapu].copy()
    p_d_root: np.ndarray = data.body(ee_ids['torso']).xpos.copy()
    R_d_root: np.ndarray = data.body(ee_ids['torso']).xquat.copy()
    
    # Store in a dictionary
    references_dict = {
        'com': x_c_d,
        'dx_c_d': dx_c_d,
        'q2_d': q2_d,
        'p_d_root': p_d_root,
        'R_d_root': R_d_root
    }

    return references_dict

def create_weights(nv1: int, nu: int, nc: int) -> dict:
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
    wq2: float = 1
    q2: np.ndarray = wq2 * np.identity(nu)  # ddq1,ddq2
    q: np.ndarray =  np.zeros((nv1 + nu,nv1 + nu))  # ddq1,ddq2
    q[nv1:, nv1:] = 0 * np.identity(nu)  # ddq2
    tau: np.ndarray = .001 * np.identity(nu)  # tau
    torso: np.ndarray = 0 * np.identity(3)  # EE orientation task
    forces: np.ndarray = .000001 * np.identity(3*nc)

    # Create and return the dictionary
    weights_dict = {
        'com': com,
        'q2': q2,
        'q': q,
        'tau': tau,
        'torso': torso,
        'forces': forces
    }
    return weights_dict

def get_list_of_contacts():
    # contacts: List[str] = ["fl_site_1", "fl_site_2", "fl_site_3", "fl_site_4", \
    #     "fr_site_1", "fr_site_2", "fr_site_3", "fr_site_4"]
    contacts: List[str] = ["lfoot1", "lfoot2", "rfoot1", "rfoot2"]
    return contacts

def get_end_effector_names():
    names: List[str] = ["torso"]
    return names

def get_state(data, 
            ee_ids: Dict[str, int], 
            jacs: Dict[int, Dict[str, np.ndarray]], 
            qmapu: np.ndarray, 
            vmapu: np.ndarray) -> Dict[str, Any]:
    """Retrieve the states for all end effectors and return in a single dictionary.

    Args:
        data: The simulation data.
        ee_ids (Dict[str, int]): A dictionary mapping end effector names to their IDs.
        jacs (Dict[int, Dict[str, np.ndarray]]): A dictionary containing Jacobians for the end effectors.

    Returns:
        Dict[str, Any]: A dictionary containing state information for all end effectors.
    """
    state = {
        'com': data.subtree_com[ee_ids['torso']],
        'dcom': data.subtree_linvel[ee_ids['torso']],
        'angvel': jacs[ee_ids['torso']]['r'] @ data.qvel,
        'R_torso': data.body(ee_ids['torso']).xquat,
        'q2': data.qpos[qmapu],
        'v2': data.qvel[vmapu]
    }

    return state

def compute_des_acc(t, ref, gains, state, data, nu, nv1, vmapu):
    des_acc: Dict[str, np.ndarray] = {}

    r = 0.0                                                                                          
    f = 0.0                                                                                          
    (x_d, v_d) = circular_motion(t, ref['com'], r, f)                                              
    des_acc['com'] = ddotx_c_d(state['com'], state['dcom'], x_d, v_d, gains['Kp_c'], gains['Kd_c'])
 
    des_acc['joints'] = ddotq_d(state['q2'], state['v2'], ref['q2_d'], np.zeros(nu), gains['Kp_q'], gains['Kd_q'])
    r = .0
    f = .0
    (x_d, v_d) = circular_motion(t, np.zeros(3), r, f)
    # des_acc['joints_full'] = ddotq_d_full(data.qpos, data.qvel, x_d, v_d, ref['p_d_root'], ref['R_d_root'], ref['q2_d'], np.zeros(nu+nv1), gains['Kp_q'], gains['Kd_q'], vmapu)
    return des_acc

def setupQPSparseFullFullJacTwoArms(M1, M2, h1, h2, C1, jacs, ee_ids, vmapu, weights, refs, nv1, nu, nforce, qp, qpproblem):
    ntau = nu
    nv = M1.shape[1]
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda] 
    H = np.zeros((ntau+nv+nforce, ntau+nv+nforce))
    g = np.zeros(ntau+nv+nforce)

    J1 = jacs[ee_ids['torso']]['t']
    # J2 = np.eye(nu+nv1,nu+nv1)
    J4 = jacs[ee_ids['torso']]['r']
    J5 = np.eye(nu,nu)

    W1 = weights['com']
    # W2 = weights['q']
    W3 = weights['tau']
    W4 = weights['torso']
    W5 = weights['q2']
    W6 = weights['forces']

    ref1 = refs['com']
    # ref2 = refs['joints_full']
    ref5 = refs['joints']

    H[ntau:ntau+nv, ntau:ntau+nv] += J1.T@W1@J1 # ddq_2
    # H[ntau:ntau+nv, ntau:ntau+nv] += J2.T@W2@J2 # ddq
    H[ntau:ntau+nv, ntau:ntau+nv] += J4.T@W4@J4 # ddq
    H[:nu, :nu] += W3 # tau
    udof = np.ix_(vmapu, vmapu)
    H[udof] += J5.T@W5@J5 # ddq_2
    H[ntau+nv:ntau+nv+nforce, ntau+nv:ntau+nv+nforce] += W6

    r1 = ref1@W1@J1
    # r2 = ref2@W2@J2
    r5 = ref5@W5@J5

    g[ntau:ntau+nv] += r1 # ddq
    # g[ntau:ntau+nv] += r2 # ddq
    g[vmapu] += r5# ddq

    qpproblem.A = np.zeros((nv, ntau+nv+nforce))
    qpproblem.b = np.zeros(nv)

    qpproblem.A[vmapu,0:ntau] += -np.eye(ntau,ntau) # tau
    qpproblem.A[0:nv1,ntau:ntau+nv] += M1 # ddq
    qpproblem.A[nv1:nv,ntau:ntau+nv] += M2 # ddq
    qpproblem.b[0:nv1] += -h1
    qpproblem.b[nv1:nv] += -h2
    qpproblem.A[0:nv,ntau+nv:] += -C1.T # lambda

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)
