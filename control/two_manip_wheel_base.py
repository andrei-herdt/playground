import numpy as np

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
    root_id: np.ndarray = model.body('wheel_base').id
    p_d_root: np.ndarray = data.body(root_id).xpos.copy()
    R_d_root: np.ndarray = data.body(root_id).xquat.copy()
    
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
    names: List[str] = ["ee", "ee_left"]
    return names
