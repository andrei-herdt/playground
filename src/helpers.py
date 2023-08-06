from dataclasses import dataclass


@dataclass
class Perturbations:
    values = [0.1]
    times = [2]
    point = list(zip(times, values))
    npoint = 0


def get_perturbation(pert, t):
    if pert.npoint >= len(pert.values) or t < pert.times[pert.npoint]:
        return 0

    np = pert.npoint
    pert.npoint += 1

    return pert.values[np]
