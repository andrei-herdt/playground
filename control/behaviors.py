import numpy as np
from typing import List


class BehaviorNode:
    def pre_condition(self, *args, **kwargs):
        return True

    def execute(self, *args, **kwargs):
        if self.pre_condition(*args, **kwargs):
            self._execute(*args, **kwargs)
            return self._post_condition(*args, **kwargs)
        else:
            return False

    def _execute(self, *args, **kwargs):
        pass

    def _post_condition(self, *args, **kwargs):
        return True


class MoveNode(BehaviorNode):
    def __init__(self, pos_des: np.ndarray):
        super().__init__()
        self.pos_des = pos_des.copy()

    def execute(self, pos: np.ndarray, ref: np.ndarray):
        if self.pre_condition(pos, ref):
            self._execute(ref)
            return self._post_condition(pos)
        else:
            return False

    def _execute(self, ref: np.ndarray):
        ref[:] = self.pos_des.copy()

    def _post_condition(self, pos: np.ndarray):
        print("post_condition")
        dist = np.linalg.norm(self.pos_des - pos)
        if dist < 0.015:
            print("dist: ", dist)
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
        print(self.status)
        self.status = self.children[self.child_id].execute(*args, **kwargs)
        print(self.status)
        if self.status:
            self.status = False
            self.child_id = self.child_id + 1
            print("incrementing")


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
