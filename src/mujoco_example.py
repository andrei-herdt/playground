import mujoco
import mujoco.viewer
import numpy as np
from typing import Callable, Optional, Union, List
import scipy.linalg
# import mediapy as media

# Graphics and plotting.
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

with open('/workdir/mujoco/model/humanoid/humanoid.xml', 'r') as f:
  xml = f.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

mujoco.mj_forward(model, data)
renderer.update_scene(data)
# media.show_image(renderer.render())
mujoco.viewer.launch(model, data)
