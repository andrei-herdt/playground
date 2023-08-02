import time

import mujoco
import mujoco.viewer

import helpers

model = mujoco.MjModel.from_xml_path(
    '/workdir/mujoco/model/humanoid/humanoid.xml')
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 1)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        data.qacc = 0  # Assert that there is no the acceleration.
        mujoco.mj_inverse(model, data)
        print(data.qfrc_inverse)

        input("step")
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
