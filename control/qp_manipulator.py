import time

import mujoco
import mujoco.viewer
from mujoco import (
    MjModel,
    MjData,
    mj_resetDataKeyframe,
    mj_kinematics,
    mj_comPos,
    mj_jacSite,
    mj_fullM,
    mj_step,
)
import numpy as np
from helpers import (
    initialize_zero_array,
    get_ee_body_ids,
    QPProblem,
    create_jacobians_dict,
    fill_jacobians_dict,
    get_dynamics,
    PosSequence,
)
from proxsuite import proxqp
from typing import List

# import wheeled_slides_manip as robot
#
import quadruped as robot
# import wheeled_manip as robot

# import robotis_op3 as robot
import humanoid as tf


np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = MjModel.from_xml_path(robot.xml_model_path)
data = MjData(model)

mj_resetDataKeyframe(model, data, robot.key_frame_id)

# Alias for model properties
nu: int = robot.nu
nv: int = model.nv
nq0 = robot.nq0
nv1 = robot.nv1
qpnv = robot.qpnv

contacts = robot.get_list_of_contacts()
ncontacts = len(contacts)

# Generate actuator mappings
act_j_names = robot.get_actuated_names()
vmapu = tf.get_vmapu(act_j_names, model)
qmapu = tf.get_qmapu(act_j_names, model)
udof = np.ix_(vmapu, vmapu)

# maps from qp to physical
# todo: move to qpproblem
qpmapf: List[int] = [*range(nu + nv1 + nu, nu + nv1 + nu + 3 * ncontacts)]
qpmapq: List[int] = [*range(nu, nu + qpnv)]
qpmaptau: List[int] = [*range(0, nu)]

mj_kinematics(model, data)
mj_comPos(model, data)

# Jacobians
Jc = initialize_zero_array((3 * ncontacts, nv))

M = initialize_zero_array((nv, nv))

# Initialize task matrices
A1, A2, A4 = (initialize_zero_array((3, nu)) for _ in range(3))

weights = tf.create_weights(nv1, nu, ncontacts, robot.root_name)
ee_names = robot.get_end_effector_names()
ee_ids = get_ee_body_ids(ee_names, model)
ref = tf.create_references_dict(data, ee_ids, qmapu, robot)
gains = tf.create_gains_dict()

# # Make square reference
# orig_pos = ref["ee_p"].copy()
# positions = [orig_pos.copy() for _ in range(4)]
# # RT
# positions[0][2] += 0.1
# positions[0][1] -= 0.1
# # LT
# positions[1][2] += 0.1
# positions[1][1] += 0.1
# # LB
# positions[2][2] -= 0.1
# positions[2][1] += 0.1
# # RB
# positions[3][2] -= 0.1
# positions[3][1] -= 0.1
# seq = PosSequence(positions)


# Move to fill_jacobians_dict
for idx, name in enumerate(contacts):
    id: int = model.site(name).id
    Cflt, Cflr = (initialize_zero_array((3, nv)) for _ in range(2))
    mj_jacSite(model, data, Cflt, Cflr, id)
    Jc[3 * idx : 3 * (idx + 1), :] = Cflt

mj_fullM(model, M, data.qM)

qpp = QPProblem()

nvar = nu + qpnv + 3 * ncontacts

# Avoid tilting
nineq = 3 * ncontacts
qp = proxqp.sparse.QP(nvar, qpnv + 3 * ncontacts, nineq)
qp.settings.compute_timings = True

Jebt, Jebr, Jebt_left, Jebr_left = (initialize_zero_array((3, nv)) for _ in range(4))

jacs = create_jacobians_dict((3, nv), robot)

sim_start = time.time()
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        fill_jacobians_dict(jacs, model, data, robot)
        task_states = tf.get_task_states(data, ee_ids, jacs, qmapu, vmapu, robot)
        dyn = get_dynamics(model, data, M, udof, vmapu, nv1)

        # Define References
        # ref["ee_p"] = seq.update(task_states["ee"]["p"]).copy()

        t = time.time() - start
        des_acc = tf.compute_des_acc(
            t, ref, gains, task_states, data, nu, nv1, vmapu, robot
        )

        tf.setupQPSparseFullFullJacTwoArms(
            dyn["M1full"],
            dyn["M2full"],
            dyn["h1full"],
            dyn["h2full"],
            Jc,
            jacs,
            ee_ids,
            vmapu,
            weights,
            des_acc,
            nv1,
            nu,
            ncontacts,
            qp,
            qpp,
            robot,
        )
        qp.solve()

        tau_d = qp.results.x[qpmaptau]
        forces = qp.results.x[qpmapf]
        ddq = qp.results.x[qpmapq]

        data.ctrl[: len(tau_d)] = tau_d

        mj_step(model, data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
