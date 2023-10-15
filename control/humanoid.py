import numpy as np
from typing import Dict, Any
from helpers import circular_motion, ddotx_c_d, ddotq_d, ddotR_d, ddotq_d_full

xml_model_path: str = '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml'
key_frame_id: int = 1

def create_gains_dict() -> Dict[str, float]:
    """
    Factory function to generate a gains dictionary.

    Returns:
    - Dictionary containing the gains.
    """
    
    # Define gains
    Kp_c: float = 0
    Kd_c: float = 0
    Kp_q: float = 1
    Kd_q: float = .1
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
        'x_c_d': x_c_d,
        'dx_c_d': dx_c_d,
        'q2_d': q2_d,
        'p_d_root': p_d_root,
        'R_d_root': R_d_root
    }

    return references_dict

def create_weights(nv1: int, nu: int) -> dict:
    """
    Factory function to generate weights dictionary.
    
    Parameters:
    - nv1: int 
    - nu: int

    Returns:
    - Dictionary containing the weight arrays.
    """
    # Task weights
    W1: np.ndarray = 0 * np.identity(3)  # EE pos task
    wq2: float = 1
    q2: np.ndarray = wq2 * np.identity(nu)  # ddq1,ddq2
    q: np.ndarray =  np.zeros((nv1 + nu,nv1 + nu))  # ddq1,ddq2
    q[nv1:, nv1:] = 0 * np.identity(nu)  # ddq2
    W3: np.ndarray = 1 * np.identity(nu)  # tau
    W4: np.ndarray = 0 * np.identity(3)  # EE orientation task

    # Create and return the dictionary
    weights_dict = {
        'W1': W1,
        'q2': q2,
        'q': q,
        'W3': W3,
        'W4': W4
    }
    return weights_dict

def get_list_of_contacts():
    contacts: List[str] = ["fl_site_1", "fl_site_2", "fl_site_3", "fl_site_4"]
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
        'x_c': data.subtree_com[ee_ids['torso']],
        'dx_c': data.subtree_linvel[ee_ids['torso']],
        'angvel': jacs[ee_ids['torso']]['r'] @ data.qvel,
        'R_torso': data.body(ee_ids['torso']).xquat,
        'q2': data.qpos[qmapu],
        'v2': data.qvel[vmapu]
    }

    return state

def compute_des_acc(t, ref, gains, state, data, nu, nv1):
    des_acc: Dict[str, np.ndarray] = {}

    r = 0.1
    f = 0.3
    (x_d, v_d) = circular_motion(t, ref['x_c_d'], r, f)
    des_acc['torso'] = ddotx_c_d(state['x_c'], state['dx_c'], x_d, v_d, gains['Kp_c'], gains['Kd_c'])
    des_acc['joints'] = ddotq_d(state['q2'], state['v2'], ref['q2_d'], np.zeros(nu), gains['Kp_q'], gains['Kd_q'])
    r = .0
    f = .0
    (x_d, v_d) = circular_motion(t, np.zeros(3), r, f)
    des_acc['joints_full'] = ddotq_d_full(data.qpos, data.qvel, x_d, v_d, ref['p_d_root'], ref['R_d_root'], ref['q2_d'], np.zeros(nu+nv1), gains['Kp_q'], gains['Kd_q'])
    return des_acc

def setupQPSparseFullFullJacTwoArms(M1, M2, h1, h2, C1, jacs, ee_ids, vmapu, weights, refs, nv1, nu, nforce, qp, qpproblem):
    ntau = nu
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda] 
    H = np.zeros((ntau+nu+nv1+nforce, ntau+nu+nv1+nforce))
    g = np.zeros(ntau+nu+nv1+nforce)

    J1 = jacs[ee_ids['torso']]['t']
    J2 = np.eye(nu+nv1,nu+nv1)
    J4 = jacs[ee_ids['torso']]['r']
    J5 = np.eye(nu,nu)

    W1 = weights['W1']
    W2 = weights['q']
    W3 = weights['W3']
    W4 = weights['W4']
    W5 = weights['q2']

    ref1 = refs['torso']
    ref2 = refs['joints_full']
    ref5 = refs['joints']

    H[:nu, :nu] += W3 # tau
    H[ntau:ntau+nv1+nu, ntau:ntau+nv1+nu] += J1.T@W1@J1 # ddq_2
    H[ntau:ntau+nv1+nu, ntau:ntau+nv1+nu] += J2.T@W2@J2 # ddq
    H[ntau+nv1:ntau+nv1+nu, ntau+nv1:ntau+nv1+nu] += J5.T@W5@J5 # ddq_2

    r1 = ref1@W1@J1
    r2 = ref2@W2@J2
    r5 = ref5@W5@J5

    g[ntau:ntau+nv1+nu] += r1 + r2# ddq
    g[ntau+nv1:ntau+nv1+nu] += r5# ddq

    qpproblem.A = np.zeros((nv1+nu, ntau+nu+nv1+nforce))
    qpproblem.b = np.zeros(nv1+nu)

    qpproblem.A[nv1:nv1+nu,0:ntau] += -np.eye(ntau,ntau) # tau
    qpproblem.A[0:nv1,ntau:ntau+nv1+nu] += M1 # ddq
    qpproblem.A[nv1:nv1+nu,ntau:ntau+nv1+nu] += M2 # ddq
    qpproblem.b[0:nv1] += -h1
    qpproblem.b[nv1:nv1+nu] += -h2
    qpproblem.A[0:nv1+nu,ntau+nv1+nu:] += -C1.T # lambda

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)
