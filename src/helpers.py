from dataclasses import dataclass

from typing import List, Tuple


@dataclass
class Perturbations:
    data: List[Tuple[int, int]]
    npoint: int


def get_perturbation(pert, t):
    if pert.npoint >= len(pert.data) or t < pert.data[pert.npoint][0]:
        return 0

    pert.npoint += 1

    return pert.data[pert.npoint-1][1]
