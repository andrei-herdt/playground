import time

import mujoco
import mujoco.viewer
from mujoco import (
    MjModel,
    MjData,
    mj_resetDataKeyframe,
    mj_kinematics,
    mj_comPos,
    mj_jacSite,
    mj_fullM,
    mj_step,
)
import numpy as np
from helpers import (
    initialize_zero_array,
    get_ee_body_ids,
    QPProblem,
    initialize_box_constraints,
    create_jacobians_dict,
    fill_jacobians_dict,
    Perturbations,
    get_dynamics,
    create_figure,
    draw_vectors,
)
from proxsuite import proxqp
from typing import List

import robotis_op3 as tf

np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = MjModel.from_xml_path(tf.xml_model_path)
data = MjData(model)

# mj_resetDataKeyframe(model, data, tf.key_frame_id)

# Alias for model properties
nu: int = model.nu
nv: int = model.nv
nq0 = tf.nq0
nv1 = tf.nv1
qpnv = nv1 + nu

# Jcom = np.zeros(3, nv)
sim_start = time.time()
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        data.ctrl[10] = -0.5
        data.ctrl[11] = 1
        data.ctrl[12] = +0.5
        data.ctrl[16] = +0.5
        data.ctrl[17] = -1
        data.ctrl[18] = -0.5

        mj_step(model, data)

        # mujoco.mj_jacSubtreeCom(model, data, Jcom, 0)
        # __import__("pdb").set_trace()
        print(data.qpos)
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
