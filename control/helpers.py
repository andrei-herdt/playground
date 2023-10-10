from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import proxsuite
import mujoco
import numpy as np

# Tasks
def get_ee_body_ids(names: List[str], model) -> Dict[str, int]:
    bodies: Dict[str,int] = {}
    for name in names:
        bodies[name] = model.body(name).id
    return bodies

def circular_motion(t, p0, r, f, offset=0):
    w = 2*np.pi*f
    p_d = np.array([p0[0]+r*np.cos(w*t+offset),p0[1]+ r*np.sin(w*t+offset), p0[2]])
    v_d = np.array([-w*r*np.sin(w*t+offset),w*r*np.cos(w*t+offset),0])
    return (p_d, v_d)

def linear_motion(t, p0, v):
    p_d = np.array(p0+t*v)
    v_d = np.array(v)
    return (p_d, v_d)

def initialize_zero_array(shape):
    """Utility function to initialize a zero array of the given shape."""
    return np.zeros(shape)

def initialize_box_constraints(size, lower_bound=-1e8, upper_bound=1e8):
    """Utility function to initialize box constraints."""
    l_box = np.full(size, lower_bound)
    u_box = np.full(size, upper_bound)
    return l_box, u_box

def ddotx_c_d(p, v, p_d, v_d, Kp_c, Kd_c): 
    return -Kp_c * (p - p_d) - Kd_c * (v - v_d)

def ddotq_d(p, v, q2_d, v2_d, Kp_q, Kd_q): 
    return -Kp_q * (p - q2_d) - Kd_q * (v - v2_d) 

def ddotq_d_full(p, v, p_delta, v_delta, p_d_root, R_d_root, q2_d, v_d, Kp_q, Kd_q): 
    angerr = np.zeros(3)
    ang = p[3:7]
    mujoco.mju_subQuat(angerr, ang, R_d_root)
    p_err = np.zeros_like(v)
    p_err[:3] = (p[:3]-p_d_root - p_delta)
    p_err[3:6] = angerr
    p_err[6:] = q2_d
    v_d[:3] = v_delta
    return -Kp_q * p_err - Kd_q * (v - v_d) 

def ddotR_d(p, v, R_d_ee, dR_d_ee, Kp_r, Kd_r): 
    angerr = np.zeros(3)
    mujoco.mju_subQuat(angerr, p, R_d_ee)
    return -Kp_r * angerr - Kd_r * (v - dR_d_ee) 

@dataclass
class Perturbations:
    data: List[Tuple[int, int]]
    npoint: int

def get_perturbation(pert, t):
    if pert.npoint >= len(pert.data) or t < pert.data[pert.npoint][0]:
        return 0

    pert.npoint += 1

    return pert.data[pert.npoint-1][1]

def calculateCoMAcc(model, data):
    # Calculate 'dot J'
    # Given some qpos we need to first update the internal datastructures. I'm assuming all this happens within a rollout, so qpos has just been updated by mj_step but derived quantities have not. mj_forward will do this for us but it's enough to call mj_kinematics and mj_comPos.
    #
    # Call the relevant jac function and save the Jacobian in J.
    # Choose a very small positive number h. Anything in the range 
    # should give identical results.
    delta_t = 1e-3
    # Call mj_integratePos(m, d->qpos, d->qvel, h). This will integrate qpos in-place with the timestep h.
    qpos_bkp = data.qpos
    mujoco.mj_integratePos(model, data.qpos, data.qvel, delta_t)

    Jc = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, Jc, model.body('torso').id)
    # Do step 1 again to update mjData kinematic quantities.
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    # Get the new Jacobian as in step 2, call it Jh.
    Jc_plus = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, Jc_plus, model.body('torso').id)
    # The quantity we want is Jdot = (Jh-J)/h.
    Jdot = (Jc_plus - Jc)/delta_t
    # Reset d->qpos to the original value, continue with the simulation. Kinematic quantities will be overwritten, no need to call kinematics and comPos again.
    data.qpos = qpos_bkp
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)

    # Compute com acceleration via:
    # \ddot c = J_c \ddot q_2 + \dot J_c \dot q_2
    return Jc@data.qacc + Jdot@data.qvel

@dataclass
class QPProblem:
    A: np.ndarray = None
    b: np.ndarray = None
    C: np.ndarray = None
    l: np.ndarray = None
    u: np.ndarray = None
    l_box: np.ndarray = None
    u_box: np.ndarray = None

def setupQPDense(M2, J1, J2, J4, W1, W2, W3, W4, h2, ref1, ref2, ref4, nu, nforce, qp, qpproblem):
    Minv = np.linalg.inv(M2)
    # todo: double check J1.T
    A1 = np.zeros_like(J1)
    A2 = np.zeros_like(J2)
    A4 = np.zeros_like(J4)

    A1[:,:nu] = J1@Minv
    A2[:,:nu] = J2@Minv
    A4[:,:nu] = J4@Minv
    H1 = A1.T@W1@A1
    H2 = A2.T@W2@A2
    H4 = A4.T@W4@A4
    H = H1 + H2 + W3[:nu+nforce,:nu+nforce] + H4

    r1 = (A1[:,:nu]@h2 + ref1)@W1@A1
    r2 = (A2[:,:nu]@h2 + ref2)@W2@A2
    r4 = (A4[:,:nu]@h2 + ref4)@W4@A4

    g = r1 + r2 + r4

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)

def setupQPSparse(M2, J1, J2, J4, W1, W2, W3, W4, h2, ref1, ref2, ref4, nu, nforce, qp, qpproblem):
    # Assume \tau,\ddot q arrangement
    H = np.zeros((2*nu, 2*nu))
    g = np.zeros(2*nu)

    H[nu:2*nu, nu:2*nu] += J1.T@W1@J1
    H[nu:2*nu, nu:2*nu] += J2.T@W2@J2
    H[:nu, :nu] += W3
    H[nu:2*nu, nu:2*nu] += J4.T@W4@J4

    r1 = ref1@W1@J1
    r2 = ref2@W2@J2
    r4 = ref4@W4@J4

    g[nu:2*nu] = r1 + r2 + r4

    qpproblem.A = np.zeros((nu, 2*nu))
    qpproblem.b = np.zeros(nu)

    qpproblem.A[:,nu:2*nu] += M2
    qpproblem.A[:nu,:nu] += -np.eye(nu,nu)
    qpproblem.b = -h2

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)

def setupQPSparseFull(M1, M2, h1, h2, C1, J1, J2, J4, W1, W2, W3, W4, ref1, ref2, ref4, nv0, nu, nforce, qp, qpproblem):
    
    ntau = nu
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda] 
    H = np.zeros((ntau+nu+nv0+nforce, ntau+nu+nv0+nforce))
    g = np.zeros(ntau+nu+nv0+nforce)

    H[:nu, :nu] += W3 # tau
    H[ntau+nv0:ntau+nv0+nu, ntau+nv0:ntau+nv0+nu] += J1.T@W1@J1 # ddq_2
    H[ntau+nv0:ntau+nv0+nu, ntau+nv0:ntau+nv0+nu] += J4.T@W4@J4 # ddq_2
    H[ntau+nv0:ntau+nv0+nu, ntau+nv0:ntau+nv0+nu] += J2.T@W2@J2 # ddq_2

    r1 = ref1@W1@J1
    r2 = ref2@W2@J2
    r4 = ref4@W4@J4

    g[ntau+nv0:ntau+nv0+nu] = r1 + r2 + r4 # ddq_2

    qpproblem.A = np.zeros((nv0+nu+nv0, ntau+nu+nv0+nforce))
    qpproblem.b = np.zeros(nv0+nu+nv0)

    qpproblem.A[nv0:nv0+nu,0:ntau] += -np.eye(ntau,ntau) # tau
    qpproblem.A[0:nv0,ntau:ntau+nv0+nu] += M1 # ddq
    qpproblem.A[nv0:nv0+nu,ntau:ntau+nv0+nu] += M2 # ddq
    qpproblem.b[0:nv0] += -h1
    qpproblem.b[nv0:nv0+nu] += -h2
    qpproblem.A[0:nv0+nu,ntau+nv0+nu:] += -C1.T # lambda
    qpproblem.A[nv0+nu:nv0+nu+nv0,ntau:ntau+nv0] += np.eye(nv0,nv0) # tau

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)

def setupQPSparseFullFullJac(M1, M2, h1, h2, C1, J1, J2, J4, W1, W2, W3, W4, ref1, ref2, ref4, nv0, nu, nforce, qp, qpproblem):
    ntau = nu
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda] 
    H = np.zeros((ntau+nu+nv0+nforce, ntau+nu+nv0+nforce))
    g = np.zeros(ntau+nu+nv0+nforce)

    H[:nu, :nu] += W3 # tau
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J1.T@W1@J1 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J2.T@W2@J2 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J4.T@W4@J4 # ddq_2

    r1 = ref1@W1@J1
    r2 = ref2@W2@J2
    r4 = ref4@W4@J4

    g[ntau:ntau+nv0+nu] += r1 + r2 + r4 # ddq

    qpproblem.A = np.zeros((nv0+nu, ntau+nu+nv0+nforce))
    qpproblem.b = np.zeros(nv0+nu)

    qpproblem.A[nv0:nv0+nu,0:ntau] += -np.eye(ntau,ntau) # tau
    qpproblem.A[0:nv0,ntau:ntau+nv0+nu] += M1 # ddq
    qpproblem.A[nv0:nv0+nu,ntau:ntau+nv0+nu] += M2 # ddq
    qpproblem.b[0:nv0] += -h1
    qpproblem.b[nv0:nv0+nu] += -h2
    qpproblem.A[0:nv0+nu,ntau+nv0+nu:] += -C1.T # lambda

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)

def setupQPSparseFullFullJacTwoArms(M1, M2, h1, h2, C1, jacs, ee_ids, vmapu, J2, weights, ref1, ref1_left, ref2, ref4, ref4_left, nv0, nu, nforce, qp, qpproblem):
    ntau = nu
    # Assume arrangement
    # [tau,ddq_1, ddq_2, lambda] 
    H = np.zeros((ntau+nu+nv0+nforce, ntau+nu+nv0+nforce))
    g = np.zeros(ntau+nu+nv0+nforce)

    J1 = jacs[ee_ids['ee']]['t']
    J1_left = jacs[ee_ids['ee_left']]['t']
    J4 = jacs[ee_ids['ee']]['r']
    J4_left = jacs[ee_ids['ee_left']]['r']

    W1 = weights['W1']
    W1_left = weights['W1_left']
    W2 = weights['W2full']
    W3 = weights['W3']
    W4 = weights['W4']
    W4_left = weights['W4_left']

    H[:nu, :nu] += W3 # tau
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J1.T@W1@J1 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J1_left.T@W1_left@J1_left # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J2.T@W2@J2 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J4.T@W4@J4 # ddq_2
    H[ntau:ntau+nv0+nu, ntau:ntau+nv0+nu] += J4_left.T@W4_left@J4_left # ddq_2

    r1 = ref1@W1@J1
    r1_left = ref1_left@W1_left@J1_left
    r2 = ref2@W2@J2
    r4 = ref4@W4@J4
    r4_left = ref4_left@W4_left@J4_left

    g[ntau:ntau+nv0+nu] += r1 + r1_left + r2 + r4 + r4_left # ddq

    qpproblem.A = np.zeros((nv0+nu, ntau+nu+nv0+nforce))
    qpproblem.b = np.zeros(nv0+nu)

    qpproblem.A[nv0:nv0+nu,0:ntau] += -np.eye(ntau,ntau) # tau
    qpproblem.A[0:nv0,ntau:ntau+nv0+nu] += M1 # ddq
    qpproblem.A[nv0:nv0+nu,ntau:ntau+nv0+nu] += M2 # ddq
    qpproblem.b[0:nv0] += -h1
    qpproblem.b[nv0:nv0+nu] += -h2
    qpproblem.A[0:nv0+nu,ntau+nv0+nu:] += -C1.T # lambda

    qp.init(H, -g, qpproblem.A, qpproblem.b, qpproblem.C, qpproblem.l, qpproblem.u, qpproblem.l_box, qpproblem.u_box)
