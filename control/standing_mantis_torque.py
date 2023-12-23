from typing import List


xml_model_path: (
    str
) = "/workdir/playground/3rdparty/mantisurdf/bodyonly/urdf/standing_mantis_torque.xml"
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
nv: int = 18
nq0 = 7
nv1 = 6
qpnv = nv1 + nu


# todo: make simple list attribute
def get_list_of_contacts():
    contacts: List[str] = ["frfoot", "flfoot", "rrfoot", "rlfoot"]
    return contacts


# todo: make simple list attribute
def get_end_effector_names():
    names: List[str] = [root_name]
    return names