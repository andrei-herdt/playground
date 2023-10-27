import numpy as np
from typing import Dict, Any, List
from helpers import circular_motion, ddotR_d, ddotx_c_d, ddotq_d, ddotq_d_full

xml_model_path: (
    str
) = "/workdir/playground/3rdparty/mujoco_menagerie/robotis_op3/scene.xml"
key_frame_id: int = 0
nq0 = 7
nv1 = 6

root_name = "body_link"
# todo: create struct of these
# todo: add list of contacts to struct

def get_list_of_contacts():
    contacts: List[str] = [
        "lfoot1",
        "lfoot2",
        "lfoot3",
        "lfoot4",
        "rfoot1",
        "rfoot2",
        "rfoot3",
        "rfoot4",
    ]
    return contacts

def get_actuated_names() -> List[str]:
    joint_names: List[str] = [
        "head_pan",
        "head_tilt",
        "l_sho_pitch",
        "l_sho_roll",
        "l_el",
        "r_sho_pitch",
        "r_sho_roll",
        "r_el",
        "l_hip_yaw",
        "l_hip_roll",
        "l_hip_pitch",
        "l_knee",
        "l_ank_pitch",
        "l_ank_roll",
        "r_hip_yaw",
        "r_hip_roll",
        "r_hip_pitch",
        "r_knee",
        "r_ank_pitch",
        "r_ank_roll",
    ]
    return joint_names

def get_end_effector_names():
    names: List[str] = [root_name]
    return names
