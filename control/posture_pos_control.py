import time

import mujoco
import mujoco.viewer
from mujoco import (
    MjModel,
    MjData,
    mj_resetDataKeyframe,
    mj_step,
)
import numpy as np
from behaviors import MoveNode, SuckNode, SequenceNode, TouchNode

import suspended_mantis as robot
# import standing_mantis as robot
# import suspended_quadruped as robot
# import wheeled_manip as robot

np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = MjModel.from_xml_path(robot.xml_model_path)
data = MjData(model)

mj_resetDataKeyframe(model, data, robot.key_frame_id)
ctrl: np.array = np.zeros(2 * robot.nu)
ctrl[: robot.nu] = data.qpos[robot.nq0 :].copy()

sim_start = time.time()
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        data.ctrl = ctrl

        mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
