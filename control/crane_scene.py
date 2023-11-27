import numpy as np
from typing import Dict, Any, List
from helpers import circular_motion, ddotR_d, ddotx_c_d, ddotq_d, ddotq_d_full


xml_model_path: (
    str
) = "/workdir/playground/3rdparty/mantisurdf/environment/crane/scene.xml"
key_frame_id: int = 0

def get_actuated_names() -> List[str]:
    joint_names: List[str] = [
        "FrontLeftLegHip",
        "FrontLeftUpperLeg",
        "FrontLeftLowerLeg",
        "FrontRightHip",
        "FrontRightUpperLeg",
        "FrontRightLowerLeg",
        "RearLeftHip",
        "RearLeftUpperLeg",
        "RearLeftLowerLeg",
        "RearRightHip",
        "RearRightUpperLeg",
        "RearRightLowerLeg",
    ]
    return joint_names


root_name = "base_link"
# todo: create struct of these
# todo: add list of contacts to struct

nu: int = 12
nv: int = 12
nq0 = 7  # TODO: rename to nq1
nv1 = 6
qpnv = nv1 + nu


# todo: make simple list attribute
def get_list_of_contacts():
    contacts: List[str] = []
    return contacts


# todo: make simple list attribute
def get_end_effector_names():
    names: List[str] = ["frfoot", "flfoot", "rrfoot", "rlfoot"]
    return names
