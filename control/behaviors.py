import numpy as np
from typing import List, Dict


class BehaviorNode:
    def execute(self, *args, **kwargs):
        if self._pre_condition(*args, **kwargs):
            self._execute(*args, **kwargs)
            return self._post_condition(*args, **kwargs)
        else:
            return False

    def _pre_condition(self, *args, **kwargs):
        return True

    def _execute(self, *args, **kwargs):
        pass

    def _post_condition(self, *args, **kwargs):
        return True


class MoveNode(BehaviorNode):
    def __init__(self, pos_des: np.ndarray):
        super().__init__()
        self.pos_des = pos_des.copy()

    def execute(self, state: Dict, ref: Dict):
        pos: np.ndarray = state["ee"]["p"]
        if self._pre_condition(state, ref):
            self._execute(state, ref)
            return self._post_condition(state, ref)
        else:
            return False

    def _execute(self, state: Dict, ref: Dict):
        ref["ee_p"][:] = self.pos_des.copy()

    def _post_condition(self, state: Dict, ref: Dict):
        pos: np.ndarray = state["ee"]["p"]
        dist = np.linalg.norm(self.pos_des - pos)
        if dist < 0.015:
            return True
        else:
            return False


class TrajectoryNode(BehaviorNode):
    def __init__(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        t0: float,
        t1: float,
        keys: List[str],
    ):
        super().__init__()
        self.p1 = p1
        self.a = 2 * p0 - 2 * p1 + v0 + v1
        self.b = -3 * p0 + 3 * p1 - 2 * v0 - v1
        self.c = v0
        self.d = p0
        self.t0 = t0
        self.t1 = t1
        self.ee = keys

        M: np.ndarray = np.array(
            [
                [t0**3, t0**2, t0, 1],
                [3 * t0**2, 2 * t0, 1, 0],
                [t1**3, t1**2, t1, 1],
                [3 * t1**2, 2 * t1, 1, 0],
            ],
        )

        self.coeffs = np.linalg.inv(M) @ np.array([p0, v0, p1, v1])
        self.a = self.coeffs[0, :]
        self.b = self.coeffs[1, :]
        self.c = self.coeffs[2, :]
        self.d = self.coeffs[3, :]

    def execute(self, state: Dict, ref: Dict):
        # Add time to state
        if self._pre_condition(state, ref):
            self._execute(state, ref)
            return self._post_condition(state, ref)
        else:
            return False

    def _execute(self, state: Dict, ref: Dict):
        t: float = state["time"]
        if t < self.t0:
            raise ValueError(f"Input must be higher {self.t0}")
        if t > self.t1:
            t = self.t1
        trel = t

        p = self.a * trel**3 + self.b * trel**2 + self.c * trel + self.d
        dp = 3 * self.a * trel**2 + 2 * self.b * trel + self.c
        for ee in self.ee:
            ref[ee]["p"] = p
            ref[ee]["dp"] = dp

    def _post_condition(self, state: Dict, ref: Dict):
        if state["time"] >= self.t1:
            return True
        else:
            return False


class TouchNode(BehaviorNode):
    def __init__(self, pos_des: np.ndarray):
        super().__init__()
        self.pos_des = pos_des.copy()

    def execute(self, state: Dict, ref: Dict):
        pos: np.ndarray = state["ee"]["p"]
        if self._pre_condition(state, ref):
            self._execute(state, ref)
            return self._post_condition(state, ref)
        else:
            return False

    def _execute(self, state: Dict, ref: Dict):
        ref["ee_p"][:] = self.pos_des.copy()

    def _post_condition(self, state: Dict, ref: Dict):
        if state["ee_force"][0] > 30:
            return True
        else:
            return False


class SuckNode(BehaviorNode):
    def __init__(self):
        super().__init__()
        self.suction_value: float = 1

    def execute(self, state: Dict, ref: Dict):
        if self._pre_condition(state, ref):
            self._execute(ref)
            return self._post_condition(state, ref)
        else:
            return False

    def _execute(self, ref: Dict):
        ref["ee_suck"] = 1.0

    def _pre_condition(self, state: Dict, ref: Dict):
        if state["ee_force"][0] > 30:
            return True
        else:
            return False

    def _post_condition(self, state: Dict, ref: Dict):
        if state["ee_force"][0] > 100.0:
            return True
        else:
            return False


class SequenceNode(BehaviorNode):
    def __init__(self, children):
        super().__init__()
        self.children: List[BehaviorNode] = children
        self.child_id: int = 0
        self.status = False

    def _execute(self, *args, **kwargs) -> bool:
        if self.child_id >= len(self.children):
            self.child_id = 0
            self.status = False
            return self.status
        self.status = self.children[self.child_id].execute(*args, **kwargs)
        if self.status:
            self.status = False
            self.child_id = self.child_id + 1


# TODO: Add test cases
# sequence_node.execute(np.zeros(3), ref)
# print("ref: ", ref)
# sequence_node.execute(np.zeros(3), ref)
# print("ref: ", ref)
# sequence_node.execute(np.ones(3) / 3, ref)
# print("ref: ", ref)
# sequence_node.execute(np.ones(3) / 2, ref)
# print("ref: ", ref)
# sequence_node.execute(np.ones(3), ref)
# print("ref: ", ref)
