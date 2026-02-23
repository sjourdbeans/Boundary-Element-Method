import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Callable


def pyr_to_quat(pitch: float|int, 
                yaw:float|int, 
                roll:float|int):
    """
    Convert roll, pitch, yaw to a quaternion vector
    
    :param pitch: Description
    :param yaw: Description
    :param roll: Description
    """
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)

    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = cy*sp*cr + sy*cp*sr
    z = sy*cp*cr - cy*sp*sr

    return np.array([w, x, y, z])

# def quat_to_R(q):

#     w, x, y, z = q
#     return np.array([
#         [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
#         [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
#         [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
#     ])


def vector_to_quaternion_from_x(p):
    """
    Returns a quaternion that rotates the x-axis [1,0,0] into the direction of p.
    Uses SciPy's Rotation module.
    Output is in [w, x, y, z] convention.
    """
    p = np.array(p, dtype=float)    
    p /= np.linalg.norm(p)  # ensure it's a unit vector

    # Define reference vector (minus x-axis)
    ex = np.array([-1.0, 0.0, 0.0])
    rot, _ = R.align_vectors([p], [ex])

    q = rot.as_quat(scalar_first=True)

    return q

def RK4(RHS :Callable[[np.ndarray], np.ndarray],
        Y   :np.ndarray,
        t   :float,
        dt  :float)->np.ndarray:
    """
    Simple RK4 function that integrates the RHS of the ODE to the next iteration
    """
    k1 = RHS(t, Y)                      
    k2 = RHS(t + dt/2, Y + 0.5*dt*k1)
    k3 = RHS(t + dt/2, Y + 0.5*dt*k2)
    k4 = RHS(t + dt, Y + dt*k3)

    Y_next = Y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    Y_next[3:]/=np.linalg.norm(Y_next[3:])
    
    return Y_next

def forward_euler(RHS :Callable[[np.ndarray], np.ndarray],
                  Y   :np.ndarray,
                  t   :float,
                  dt  :float)->np.ndarray:
    """
    Simple Forward Euler function that integrates the RHS of the ODE to the next iteration
    """
    Y_next = Y + dt * RHS(t, Y)
    Y_next[3:]/=np.linalg.norm(Y_next[3:])
   
    return Y_next

# def rk2(RHS, Y, t, dt):
#     k1 = RHS(t, Y)
#     Y_predict = Y + dt * k1
#     k2 = RHS(t + dt, Y_predict)
#     Y_next = Y + dt * 0.5 * (k1 + k2)

#     # normalize quaternion
#     Y_next[3:] /= np.linalg.norm(Y_next[3:])
#     return Y_next

def rotate_BCs(Q, U, W, E):
    """
    Rotate the boundary conditions to particle frame.
    """

    U_body = Q.T @ U
    W_body = Q.T @ W
    E_body = Q.T @ E @ Q

    return U_body, W_body, E_body

# def omega_to_quat_dot(q, omega):
#     """Compute quaternion derivative dq/dt = 0.5 * Ω(ω) * q for [w,x,y,z]."""
#     wx, wy, wz = omega
#     Omega = np.array([
#         [0.0, -wx, -wy, -wz],
#         [wx,  0.0,  wz, -wy],
#         [wy, -wz,  0.0,  wx],
#         [wz,  wy, -wx,  0.0]
#     ])
#     return 0.5 * Omega @ q

def omega_to_quat_dot(q, omega):
    """dq/dt = 0.5 * q ⊗ (0, omega) in the lab frame"""
    wx, wy, wz = omega
    Omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0, -wz,  wy],
        [wy,  wz,  0.0, -wx],
        [wz, -wy,  wx,  0.0]
    ])
    return 0.5 * Omega @ q


def quat_multiply(q1, q2):
    """Quaternion multiplication q = q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_exp(omega, dt):
    """
    Quaternion exponential map from angular velocity.
    Returns delta quaternion corresponding to rotation omega*dt.
    """
    theta = np.linalg.norm(omega) * dt

    if theta < 1e-12:
        # small-angle approximation
        return np.array([1.0, *(0.5 * omega * dt)])

    axis = omega / np.linalg.norm(omega)
    half = 0.5 * theta

    return np.array([
        np.cos(half),
        *(np.sin(half) * axis)
    ])


def rk2(RHS, Y, t, dt):
    """
    RK2 (Heun) for translation + exponential map for quaternion.
    """
    # ---- unpack state
    x = Y[:3]
    q = Y[3:7]

    # ---- stage 1
    U1, omega1 = RHS(t, Y)

    x_predict = x + dt * U1

    dq1 = quat_exp(omega1, dt)
    q_predict = quat_multiply(dq1, q)

    Y_predict = np.concatenate([x_predict, q_predict])

    # ---- stage 2
    U2, omega2 = RHS(t + dt, Y_predict)

    # ---- translation RK2
    x_next = x + 0.5 * dt * (U1 + U2)

    # ---- rotation using averaged angular velocity
    omega_avg = 0.5 * (omega1 + omega2)
    dq = quat_exp(omega_avg, dt)

    q_next = quat_multiply(dq, q)
    q_next /= np.linalg.norm(q_next)  # numerical safety

    return np.concatenate([x_next, q_next])
