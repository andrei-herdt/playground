import numpy as np
from typing import List


class PosSequence:
    def __init__(self, pos: List[np.ndarray]):
        self.i: int = 0
        self.positions: List[np.ndarray] = pos

    def update(self, pos: np.ndarray) -> np.ndarray:
        # if (self.i == len(self.positions) - 1):
        #     return self.positions[self.i]

        dist = np.linalg.norm(self.positions[self.i] - pos)
        if dist < 0.015:
            print("dist: ", dist)
            # Loop
            if self.i == len(self.positions) - 1:
                print("restarting loop")
                self.i = 0
            else:
                print("incrementing sequence counter")
                self.i = self.i + 1
        return self.positions[self.i]
