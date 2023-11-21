import numpy as np
from typing import Dict, Any, List
from helpers import circular_motion, ddotR_d, ddotx_c_d, ddotq_d, ddotq_d_full


xml_model_path: (
    str
) = "/workdir/playground/3rdparty/mujoco_menagerie/unitree_a1/suspended_a1.xml"
key_frame_id: int = 0
nq0 = 0
nv1 = 0


def get_actuated_names() -> List[str]:
    joint_names: List[str] = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    return joint_names


root_name = "trunk"
# todo: create struct of these
# todo: add list of contacts to struct

nu: int = 12
nv: int = 12
nq0 = 0  # TODO: rename to nq1
nv1 = 0
qpnv = nv1 + nu


# todo: make simple list attribute
def get_list_of_contacts():
    contacts: List[str] = ["frfoot", "flfoot", "rrfoot", "rlfoot"]
    return contacts


# todo: make simple list attribute
def get_end_effector_names():
    names: List[str] = [root_name]
    return names
