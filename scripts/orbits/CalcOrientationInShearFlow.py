import numpy as np
import bemsolver as bem

from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.transform import Rotation as R


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



def rk4_quaternion(q, omega_func, Q, dt):
    """
    RK4 integration for quaternion.

    q : np.ndarray, shape (4,)
        Current quaternion.
    omega_func : function
        Function that returns angular velocity (in body frame).
    Q : np.ndarray, shape (3,3)
        Current rotation matrix (needed by omega_func).
    """

    f=omega_func(Q)
    k1 = omega_to_quat_dot(q, f)
    k2 = omega_to_quat_dot(q + 0.5*dt*k1, f)
    k3 = omega_to_quat_dot(q + 0.5*dt*k2, f)
    k4 = omega_to_quat_dot(q + dt*k3, f)

    q_new = q + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return normalize_quaternion(q_new)




def omega_func(Q):
    # Rotate lab-frame background flow into particle frame
    
    phi = solve(lu,piv,Q, U, W, E)

    psi= phi[:-6]
    # print(sys.surface_matrix@psi)
    # print(sys.torque_matrix@psi)

    u=phi[-6:-3]
    omega = phi[-3:]
    
    return omega 


def solve(lu,piv,Q,U,W,E):
    U_body = Q.T @ U
    W_body = Q.T @ W
    E_body = Q.T @ E @ Q

    U_rhs = sys.set_boundary_condition(U_body, W_body, E_body)
    RHS = np.hstack((U_rhs, np.zeros(6)))
    phi = lu_solve((lu, piv), -RHS)

    return phi



def vector_to_quaternion_from_x(p):
    """
    Returns a quaternion that rotates the x-axis [1,0,0] into the direction of p.
    Uses SciPy's Rotation module.
    Output is in [w, x, y, z] convention.
    """
    p = np.array(p, dtype=float)
    p /= np.linalg.norm(p)  # ensure it's a unit vector

    # Define reference vector (x-axis)
    ex = np.array([1.0, 0.0, 0.0])

    # Compute rotation aligning ex → p
    rot, _ = R.align_vectors([p], [ex])

    # Get quaternion in SciPy's format [x, y, z, w]
    q_scipy = rot.as_quat()

    # Reorder to [w, x, y, z] if you prefer that convention
    q = np.roll(q_scipy, 1)

    return q



# warnings.simplefilter("once", category=UserWarning)

# path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid-variation/spheroid_mesh_b=0.1.mat"
# path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid/spheroid_05.mat"
# path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"
path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/jeffrey-orbits-fine/jeffrey_mesh_b=0.5.mat"

# import os

# folder_path = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/spheroid-variation/new-variation"

# entries = os.listdir(folder_path)


# files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#======================================================================



# for file in files:    
mesh=bem.Mesh(path)
# mesh.b=float(file.split("=")[-1].split(".mat")[0])

# mesh.plot_mesh(plot_normals=False)
sys=bem.MobilityProblem(mesh)


M=sys.construct_grand_mobility_matrix()


lu, piv = lu_factor(M)

dt=0.01
T=800

time=np.arange(0,T+dt,dt)

initial_orientation = np.array([1,0,0])
initial_position    = np.array([0,0,0])

q = vector_to_quaternion_from_x(initial_orientation)
Q = quat_to_matrix(q)


p = np.zeros((len(time), 3))
p[0] = Q[:, 0]

for k, t in enumerate(time[:-1]):
    if t%5==0:
        print(f"Calculating timestep t={t} out of {T}")
    # RK4 integrate quaternion
    q = rk4_quaternion(q,omega_func,Q, dt)

    # Update rotation matrix from quaternion
    Q = quat_to_matrix(q)
    # if k % 10 == 0:
    Ut, _, Vt = np.linalg.svd(Q)
    Q = Ut @ Vt
    

    # Q[:,0][2]=0
    # Store orientation (major axis)
    # print(Q[:,0])
    p[k+1] = Q[:, 0]


    


    # print(v,omega)
np.savetxt(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/orientation/pz_drift/jeffrey_orbit_point_singularity_spheroid_b=0.5_shear={round(gamma_dot,2)}_T={T}_with_dt={round(dt,3)}.txt",
            np.vstack((time,p.T)).T)
# psi, force, torque = sys.solve(U,W)







# psi, force, torque = sys.solve(U,W)



# MATRIX, surface_matrix, torque_matrix=sys.construct_mobility_matrix()
# print(force)

