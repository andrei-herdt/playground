import mujoco
import mujoco.viewer
import numpy as np

# Graphics and plotting.

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

with open('/workdir/mujoco/model/humanoid/humanoid.xml', 'r') as f:
    xml = f.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

mujoco.mj_forward(model, data)
# renderer.update_scene(data)

for key in range(model.nkey):
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

mujoco.viewer.launch(model, data)
