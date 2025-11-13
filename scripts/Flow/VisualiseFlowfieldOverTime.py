import bemsolver as bem
import numpy as np

gamma_dot=0.5

def find_flow(x):
    U = np.zeros(3)

    U[0] = 0 #gamma_dot * x[1]
    U[1] = 0
    U[2] = 0

    # Background vorticity
    W = np.zeros(3)  

    W[0] = 0
    W[1] = 0
    W[2] = -gamma_dot/2

    # Rate of strain tensor
    E = gamma_dot/2*np.array([[0,1,0],
                              [1,0,0],
                              [0,0,0]])
    return U, W, E




path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/elongated-mesh-fine/elongated_spheroid_N=320.mat"


mesh=bem.Mesh(path)


initial_orientation = np.array([1,0,0])
initial_position    = np.array([0,0,0])

sys=bem.MobilityProblem(mesh,flow_function=find_flow,
                        initial_position=initial_position,
                        initial_orientation=initial_orientation,
                        particle_velocity=1)
dt=0.01
T=100

solution = sys.RBM_over_time(T,dt) 

Nx=200 + 1
Ny=200 + 1
xlim=3
ylim=3

x = np.linspace(-xlim, xlim, Nx)
y = np.linspace(-ylim, ylim, Ny)

xg, yg =np.meshgrid(x,y)
zg=np.zeros_like(xg)

xg = xg.ravel()
yg = yg.ravel()
zg = zg.ravel()
Ng = np.shape(xg)[0]

points = np.vstack((xg, yg, zg)).T


interaction = bem.FlowStokes(mesh,points)


def rotate_BCs(Q, U, W, E):
    """
    Rotate the boundary conditions to particle frame.
    """

    U_body = Q.T @ U
    W_body = Q.T @ W
    E_body = Q.T @ E @ Q

    return U_body, W_body, E_body


import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio

# === USER INPUT ===
savepath = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/videos/Jeffrey-orbit/Jeffrey-orbit_class-test.mp4"
tmp_dir = "frames"
fps = 10
quiver_density = 8
vector_scale = 100
vmax_factor = 1.2
time_indices = range(0, len(solution.time), 50)  # every 10th frame

max_mag = 5

# === PREPARE TEMP DIRECTORY ===
os.makedirs(tmp_dir, exist_ok=True)
frame_files = []

for i, iter in enumerate(time_indices):
    if i % 10==0:
        print(f"Rendering frame {i+1}/{len(time_indices)}")

    Q = solution.rotation_matrices[iter]

    U, W, E = find_flow(solution.X[iter])
    U_body, W_body, E_body = rotate_BCs(Q, U, W, E)
    psi, u, omega = solution.psi[iter], solution.u[iter], solution.omega[iter]

    U_field = interaction.calc_vector_field(psi, U_body, W_body, E_body)

    fig = interaction.plot_vector_field(
        x, y, U_field, max_mag,
        quiver_density=quiver_density,
        vector_scale=vector_scale
    )

    fig.set_size_inches(10, 8)
    ax = fig.axes[0]
    ax.set_title(f"t = {solution.time[iter]:.2f} s")
    ax.plot(mesh.isosurface[:,0],-mesh.isosurface[:,1],color='r')

    frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)

    frame_files.append(frame_path)

# === Create video ===
print("Creating video...")
frames = [imageio.imread(f) for f in frame_files]
imageio.mimsave(savepath, frames, fps=fps)

print(f"Animation saved to {savepath}")

import shutil
shutil.rmtree(tmp_dir)
