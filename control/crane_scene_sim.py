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

import crane_scene as robot
import humanoid as tf

np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = MjModel.from_xml_path(robot.xml_model_path)
data = MjData(model)

act_j_names = robot.get_actuated_names()
# Generate actuator mappings
vmapu = tf.get_vmapu(act_j_names, model)
qmapu = tf.get_qmapu(act_j_names, model)
udof = np.ix_(vmapu, vmapu)


mj_resetDataKeyframe(model, data, robot.key_frame_id)
ctrl: np.ndarray = np.zeros(2 * robot.nu)
ctrl[: robot.nu] = data.qpos[qmapu].copy()

# ee reference
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
