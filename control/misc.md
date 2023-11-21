# The french connection
https://scaron.info/slides/humanoids-2022.pdf
Integrate https://github.com/machines-in-motion/mujoco_utils
https://github.com/robot-descriptions/robot_descriptions.py#arms

# Mujoco links
(xml reference)[https://mujoco.readthedocs.io/en/stable/XMLreference.html]
(mjModel)[https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel]
(mjData)[https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata]
(equality)[https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-connect]
(option)[https://mujoco.readthedocs.io/en/stable/XMLreference.html#option]
(solver parameters)[https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters]
(functions)[https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html]

# Mujoco binaries
https://github.com/google-deepmind/mujoco/releases/download/3.0.0/mujoco-3.0.0-linux-x86_64.tar.gz
https://github.com/google-deepmind/mujoco/releases/download/3.0.0/mujoco-3.0.0-linux-aarch64.tar.gz

# ProxQP links
https://github.com/Simple-Robotics/proxsuite/tree/main/examples/python

# Render and save images
renderer = mujoco.Renderer(model)
renderer.update_scene(data, 'fixed')
pixels_array = renderer.render().flatten()
pixels = renderer.render()
