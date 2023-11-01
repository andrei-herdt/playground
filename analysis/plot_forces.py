import matplotlib.pyplot as plt
import numpy as np


def cop(f_ext, h_ext, m) -> float:
    g: float = 9.81
    z = f_ext * h_ext / m / g
    return z


h_ext = 2.0
mass_robot = 50
f_ext_max = 100
f_ext = np.linspace(0, 100)

cop_deviation = cop(f_ext, h_ext, mass_robot)

plt.plot(f_ext, cop_deviation)
plt.xlabel("Applied force [N]")
plt.ylabel("CoM-CoP [m]")
plt.show()
