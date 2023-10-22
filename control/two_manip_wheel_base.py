import numpy as np
from typing import Dict, Any
from helpers import circular_motion, ddotx_c_d, ddotq_d, ddotR_d, ddotq_d_full

xml_model_path: str = '/workdir/playground/3rdparty/kinova_mj_description/xml/two_manipulator_on_wheels.xml'
key_frame_id: int = 0
nq0 = 7
nv1 = 6

def create_gains_dict() -> Dict[str, float]:
    """
    Factory function to generate a gains dictionary.

    Returns:
    - Dictionary containing the gains.
    """
    
    # Define gains
    Kp_c: float = 10000
    Kd_c: float = 1000
    Kp_q: float = 0
    Kd_q: float = 100
    Kp_r: float = 1000
    Kd_r: float = 100
    
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
    - ee_ids: Dictionary with keys 'ee' and 'ee_left' mapping to respective IDs.
    - qmapu: Index or slice for the qpos method of the data source.

    Returns:
    - Dictionary containing the references.
    """
    
    # Create references
    x_c_d: np.ndarray = data.subtree_com[ee_ids['ee']].copy()
    x_c_d_left: np.ndarray = data.subtree_com[ee_ids['ee_left']].copy()
    dx_c_d: np.ndarray = np.zeros(3)
    dx_c_d_left: np.ndarray = np.zeros(3)
    q2_d: np.ndarray = data.qpos[qmapu].copy()
    R_d_ee: np.ndarray = data.body(ee_ids['ee']).xquat.copy()
    R_d_ee_left: np.ndarray = data.body(ee_ids['ee_left']).xquat.copy()
    p_d_root: np.ndarray = data.body(ee_ids['wheel_base']).xpos.copy()
    R_d_root: np.ndarray = data.body(ee_ids['wheel_base']).xquat.copy()
    
    # Store in a dictionary
    references_dict = {
        'x_c_d': x_c_d,
        'x_c_d_left': x_c_d_left,
        'dx_c_d': dx_c_d,
        'dx_c_d_left': dx_c_d_left,
        'q2_d': q2_d,
        'R_d_ee': R_d_ee,
        'R_d_ee_left': R_d_ee_left,
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
    w2: float = 1
    W1: np.ndarray = 10 * np.identity(3)  # EE pos task
    W1_left: np.ndarray = 10 * np.identity(3)  # EE pos task
    # todo
    W2: np.ndarray = w2 * np.identity(nu)  # ddq2
    W3: np.ndarray = 0.01 * np.identity(nu)  # tau
    W4: np.ndarray = 1 * np.identity(3)  # EE orientation task
    W4_left: np.ndarray = 1 * np.identity(3)  # EE orientation task
    W2full: np.ndarray = w2 * np.identity(nv1 + nu)  # ddq1,ddq2
    W2full[:nv1, :nv1] = 100 * np.identity(nv1)  # ddq1
    W2full[6, 6] = 10000  # deck joint

    # Create and return the dictionary
    weights_dict = {
        'W1': W1,
        'W1_left': W1_left,
        'W2': W2,
        'W3': W3,
        'W4': W4,
        'W4_left': W4_left,
        'W2full': W2full
    }
    return weights_dict

def get_list_of_contacts():
    contacts: List[str] = ["wheel_fl", "wheel_hl", "wheel_hr", "wheel_fr"]
    return contacts

def get_end_effector_names():
    names: List[str] = ["ee", "ee_left", "wheel_base"]
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
        'x_c': data.subtree_com[ee_ids['ee']],
        'dx_c': data.subtree_linvel[ee_ids['ee']],
        'angvel': jacs[ee_ids['ee']]['r'] @ data.qvel,
        'R_ee': data.body(ee_ids['ee']).xquat,
        'x_c_left': data.subtree_com[ee_ids['ee_left']],
        'dx_c_left': data.subtree_linvel[ee_ids['ee_left']],
        'angvel_left': jacs[ee_ids['ee_left']]['r']@data.qvel,
        'R_ee_left': data.body(ee_ids['ee_left']).xquat,
        'q2': data.qpos[qmapu],
        'v2': data.qvel[vmapu]
    }

    return state

def compute_des_acc(t, ref, gains, state, data, nu, nv0):
    des_acc: Dict[str, np.ndarray] = {}

    r = 0.1
    f = 0.3
    (x_d, v_d) = circular_motion(t, ref['x_c_d'], r, f)
    des_acc['ee'] = ddotx_c_d(state['x_c'], state['dx_c'], x_d, v_d, gains['Kp_c'], gains['Kd_c'])
    (x_d, v_d) = circular_motion(t, ref['x_c_d_left'], r, f, -np.pi)
    des_acc['ee_left'] = ddotx_c_d(state['x_c_left'], state['dx_c_left'], x_d, v_d, gains['Kp_c'], gains['Kd_c'])
    des_acc['joints'] = ddotq_d(state['q2'], state['v2'], ref['q2_d'], np.zeros(nu), gains['Kp_q'], gains['Kd_q'])
    des_acc['ee_R'] = ddotR_d(state['R_ee'], state['angvel'], ref['R_d_ee'], np.zeros(3), gains['Kp_r'], gains['Kd_r'])
    des_acc['ee_R_left'] = ddotR_d(state['R_ee_left'], state['angvel_left'], ref['R_d_ee_left'], np.zeros(3), gains['Kp_r'], gains['Kd_r'])
    r = .0
    f = .0
    (x_d, v_d) = circular_motion(t, np.zeros(3), r, f)
    des_acc['joints_full'] = ddotq_d_full(data.qpos, data.qvel, x_d, v_d, ref['p_d_root'], ref['R_d_root'], ref['q2_d'], np.zeros(nu+nv0), gains['Kp_q'], gains['Kd_q'])
    return des_acc

def setupQPSparseFullFullJacTwoArms(M1, M2, h1, h2, C1, jacs, ee_ids, vmapu, weights, refs, nv0, nu, nforce, qp, qpproblem):
    ntau = nu
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda] 
    H = np.zeros((ntau+nu+nv0+nforce, ntau+nu+nv0+nforce))
    g = np.zeros(ntau+nu+nv0+nforce)

    J1 = jacs[ee_ids['ee']]['t']
    J1_left = jacs[ee_ids['ee_left']]['t']
    J2 = np.eye(nu+nv0,nu+nv0)
    J4 = jacs[ee_ids['ee']]['r']
    J4_left = jacs[ee_ids['ee_left']]['r']

    W1 = weights['W1']
    W1_left = weights['W1_left']
    W2 = weights['W2full']
    W3 = weights['W3']
    W4 = weights['W4']
    W4_left = weights['W4_left']

    ref1 = refs['ee']
    ref1_left = refs['ee_left']
    ref2 = refs['joints_full']
    ref4 = refs['ee_R']
    ref4_left = refs['ee_R_left']

    H[:nu, :nu] += W3 # tau
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J1.T@W1@J1 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J1_left.T@W1_left@J1_left # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J2.T@W2@J2 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J4.T@W4@J4 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J4_left.T@W4_left@J4_left # ddq_2

    r1 = ref1@W1@J1
    r1_left = ref1_left@W1_left@J1_left
    r2 = ref2@W2@J2
    r4 = ref4@W4@J4
    r4_left = ref4_left@W4_left@J4_left

    g[ntau:ntau+nv0+nu] += r1 + r1_left + r2 + r4 + r4_left # ddq

    qpproblem.A = np.zeros((nv0+nu, ntau+nu+nv0+nforce))
    qpproblem.b = np.zeros(nv0+nu)

    qpproblem.A[nv0:nv0+nu,0:ntau] += -np.eye(ntau,ntau) # tau
    qpproblem.A[0:nv0,ntau:ntau+nv0+nu] += M1 # ddq
    qpproblem.A[nv0:nv0+nu,ntau:ntau+nv0+nu] += M2 # ddq
    qpproblem.b[0:nv0] += -h1
    qpproblem.b[nv0:nv0+nu] += -h2
    qpproblem.A[0:nv0+nu,ntau+nv0+nu:] += -C1.T # lambda

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)
