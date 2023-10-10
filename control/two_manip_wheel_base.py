import numpy as np

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
