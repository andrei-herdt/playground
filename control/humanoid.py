import numpy as np
from typing import Dict, Any

xml_model_path: str = '/workdir/playground/3rdparty/mujoco/model/humanoid/humanoid.xml'
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
