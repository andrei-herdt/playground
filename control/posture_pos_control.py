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

from behaviors import TrajectoryNode
from typing import Dict

np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = MjModel.from_xml_path(robot.xml_model_path)
data = MjData(model)

mj_resetDataKeyframe(model, data, robot.key_frame_id)
ctrl: np.ndarray = np.zeros(2 * robot.nu)
ctrl[: robot.nu] = data.qpos[robot.nq0 :].copy()

# ee reference

state: Dict = {"time": 0}
ref: Dict = {}
jacs: Dict = {}
for ee in robot.get_end_effector_names():
    id: int = model.site(ee).id
    state[ee] = {"p": np.zeros(3)}
    ref[ee] = {
        "p": data.site(ee).xpos.copy(),
        "p_test": data.site(ee).xpos.copy(),
        "dp": np.zeros(3),
        "q": data.qpos[3 * id : 3 * (id + 1)].copy(),
        "dq": np.zeros(3),
    }
    jacs[ee] = np.zeros((3, 3))

delta_p_des = np.array([0, 0, 0.2])
children = [
    TrajectoryNode(
        ref["frfoot"]["p"],
        ref["frfoot"]["p"] + delta_p_des,
        np.zeros(3),
        np.zeros(3),
        0,
        1,
        ["frfoot", "rlfoot"],
    ),
    TrajectoryNode(
        ref["frfoot"]["p"],
        ref["frfoot"]["p"] - delta_p_des,
        np.zeros(3),
        np.zeros(3),
        1,
        2,
        ["frfoot", "rlfoot"],
    ),
    TrajectoryNode(
        ref["frfoot"]["p"],
        ref["frfoot"]["p"] + delta_p_des,
        np.zeros(3),
        np.zeros(3),
        2,
        3,
        ["flfoot", "rrfoot"],
    ),
    TrajectoryNode(
        ref["frfoot"]["p"],
        ref["frfoot"]["p"] - delta_p_des,
        np.zeros(3),
        np.zeros(3),
        3,
        4,
        ["flfoot", "rrfoot"],
    ),
]
sequence_node = SequenceNode(children)

sim_start = time.time()
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    state["time"] = 0
    while viewer.is_running():
        step_start = time.time()

        # Update state of task
        state["time"] += model.opt.timestep
        for ee in robot.get_end_effector_names():
            state[ee]["p"] = data.site(ee).xpos

        sequence_node.execute(state, ref)

        for ee in robot.get_end_effector_names():
            id: int = model.site(ee).id
            Jt: np.ndarray = np.zeros((3, model.nv))
            Jr: np.ndarray = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jt, Jr, id)
            jacs[ee] = Jt[:3, 3 * id : 3 * (id + 1)]

            ref[ee]["dq"] = np.linalg.inv(jacs[ee]) @ ref[ee]["dp"]
            ref[ee]["q"] += ref[ee]["dq"] * model.opt.timestep
            ref[ee]["p_test"] += ref[ee]["dp"] * model.opt.timestep

            data.ctrl[3 * id : 3 * (id + 1)] = ref[ee]["q"]
            data.ctrl[model.nv + 3 * id : model.nv + 3 * (id + 1)] = ref[ee]["dq"]

        print(
            state["time"],
            " ",
            data.site("frfoot").xpos,
            " ",
            ref["frfoot"]["p"],
            " ",
            ref["frfoot"]["p_test"],
        )

        mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
