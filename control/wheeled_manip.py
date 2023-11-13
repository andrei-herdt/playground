import numpy as np
from typing import Dict, Any, List
from helpers import circular_motion, ddotR_d, ddotx_c_d, ddotq_d, ddotq_d_full

xml_model_path: str = "/workdir/playground/3rdparty/kinova_mj_description/xml/two_manipulator_on_wheels.xml"
key_frame_id: int = 0
nq0 = 7
nv1 = 6

root_name = "base_link"
# todo: create struct of these
# todo: add list of contacts to struct

nu: int = 18
nv: int = 24
nq0 = 7  # TODO: rename to nq1
nv1 = 6
qpnv = nv1 + nu


# todo: make simple list attribute
def get_list_of_contacts():
    contacts: List[str] = ["wheel_fl", "wheel_hl", "wheel_hr", "wheel_fr"]
    return contacts


# todo: make simple list attribute
def get_actuated_names() -> List[str]:
    joint_names: List[str] = [
        # left manip
        "joint_1_left",
        "joint_2_left",
        "joint_3_left",
        "joint_4_left",
        "joint_5_left",
        "joint_6_left",
        "joint_7_left",
        # right manip
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        # wheels
        "joint_8",
        "joint_9",
        "joint_10",
        "joint_11",
    ]
    return joint_names


# todo: make simple list attribute
def get_end_effector_names():
    names: List[str] = [root_name, "ee", "ee_left"]
    return names
