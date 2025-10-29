import numpy as np
import bemsolver as BEM
from scipy.linalg import expm

from scipy.linalg import lu_factor, lu_solve


# Background flow
U = np.zeros(3)

U[0] = 0
U[1] = 0
U[2] = 0

# Background vorticity
W = np.zeros(3)

gamma_dot=2



W[0] = 0
W[1] = 0
W[2] = gamma_dot/2

# Rate of strain tensor
E = gamma_dot/2*np.array([[0,1,0],
                          [1,0,0],
                          [0,0,0]])



def normalize_quaternion(q):
    """Ensure quaternion has unit length."""
    return q / np.linalg.norm(q)

def omega_to_quat_dot(q, omega):
    """Compute quaternion derivative dq/dt = 0.5 * Ω(ω) * q."""
    w = omega
    Omega = np.array([
        [0,   -w[0], -w[1], -w[2]],
        [w[0],  0,    w[2], -w[1]],
        [w[1], -w[2],  0,   w[0]],
        [w[2],  w[1], -w[0],  0 ]
    ])
    return 0.5 * Omega @ q

def quat_to_matrix(q):
    """Convert quaternion to rotation matrix."""
    q = normalize_quaternion(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])


def rk4_quaternion(q, omega_func, Q, t, dt):
    """
    RK4 integration for quaternion.

    q : np.ndarray, shape (4,)
        Current quaternion.
    omega_func : function
        Function that returns angular velocity (in body frame).
    Q : np.ndarray, shape (3,3)
        Current rotation matrix (needed by omega_func).
    """
    k1 = omega_to_quat_dot(q, omega_func(Q, t))
    k2 = omega_to_quat_dot(q + 0.5*dt*k1, omega_func(Q, t + 0.5*dt))
    k3 = omega_to_quat_dot(q + 0.5*dt*k2, omega_func(Q, t + 0.5*dt))
    k4 = omega_to_quat_dot(q + dt*k3, omega_func(Q, t + dt))

    q_new = q + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return normalize_quaternion(q_new)





# def skew(omega):
#     return np.array([[0, -omega[2], omega[1]],
#                      [omega[2], 0, -omega[0]],
#                      [-omega[1], omega[0], 0]])


# def update_rotation(Q, omega_body, dt):
#     """Integrate rotation matrix forward one step."""
#     return expm(skew(omega_body) * dt) @ Q


# def rk4_rotation(Q, omega_func, t, dt):
#     """
#     Integrate rotation matrix Q(t) using RK4.

#     Parameters
#     ----------
#     Q : np.ndarray, shape (3,3)
#         Current rotation matrix.
#     omega_func : function
#         Function omega = omega_func(Q, t) that returns angular velocity in lab frame.
#     t : float
#         Current time.
#     dt : float
#         Timestep.

#     Returns
#     -------
#     Q_new : np.ndarray, shape (3,3)
#         Updated rotation matrix at t+dt.
#     """
#     k1 = skew(omega_func(Q, t)) @ Q
#     k2 = skew(omega_func(Q + 0.5*dt*k1, t + 0.5*dt)) @ (Q + 0.5*dt*k1)
#     k3 = skew(omega_func(Q + 0.5*dt*k2, t + 0.5*dt)) @ (Q + 0.5*dt*k2)
#     k4 = skew(omega_func(Q + dt*k3, t + dt)) @ (Q + dt*k3)

#     Q_new = Q + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
#     return Q_new

def omega_func(Q, t):
    # Rotate lab-frame background flow into particle frame
    U_body = Q.T @ U
    W_body = Q.T @ W
    E_body = Q.T @ E @ Q

    # Solve BEM for current angular velocity
    U_rhs = sys.set_shear_boundary_condition(U_body, W_body, E_body)
    RHS = np.hstack((U_rhs, np.zeros(6)))
    phi = lu_solve((lu, piv), -RHS)

    omega = phi[-3:]
    return omega  



# warnings.simplefilter("once", category=UserWarning)

# path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid-variation/spheroid_mesh_b=0.1.mat"
# path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid/spheroid_05.mat"
# path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"
path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/jeffrey-orbits-fine/jeffrey_mesh_b=0.5.mat"
#======================================================================



    
mesh=BEM.Mesh(path)
mesh.plot_mesh(plot_normals=True)
sys=BEM.System(mesh)

sys.construct_mobility_matrix()

M=sys.construct_grand_mobility_matrix()

Q=np.eye(3)


lu, piv = lu_factor(M)

dt=0.01
T=80

time=np.arange(0,T+dt,dt)


Q = np.eye(3)
q = np.array([1, 0, 0, 0])   # identity rotation quaternion

p = np.zeros((len(time), 3))
p[0] = Q[:, 0]

for k, t in enumerate(time[:-1]):
    if t%5==0:
        print(f"Calculating timestep t={t} out of {T}")
    # RK4 integrate quaternion
    q = rk4_quaternion(q, omega_func, Q, t, dt)

    # Update rotation matrix from quaternion
    Q = quat_to_matrix(q)
    # if k % 10 == 0:
    Ut, _, Vt = np.linalg.svd(Q)
    Q = Ut @ Vt

    # Store orientation (major axis)
    p[k+1] = Q[:, 0]


    


    # print(v,omega)
np.savetxt(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/orientation/orientation_prolate_spheroid_b=0.5_shear={round(gamma_dot,2)}_T={T}_with_dt={round(dt,3)}.txt",
            np.vstack((time,p.T)).T)
# psi, force, torque = sys.solve(U,W)







# psi, force, torque = sys.solve(U,W)



# MATRIX, surface_matrix, torque_matrix=sys.construct_mobility_matrix()
# print(force)

