import numpy as np
from dataclasses import asdict
import matplotlib as mpl

import bemsolver as bem

# Set tick direction globally
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

import os
os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"]= "DejaVu Sans"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"

mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13

path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/elongated-mesh-fine/elongated_spheroid_N=320.mat"
plot_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Velocity"
plot_image_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Velocity"


gamma_dot=0.5

def find_flow(x):
    U = np.zeros(3)

    U[0] = gamma_dot * x[1]
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

mesh= bem.Mesh(path)

initial_orientation = np.array([1,0,0])
initial_position    = np.array([0,0,0])

velocity=10

sys=bem.MobilityProblem(mesh,flow_function=find_flow,
                        initial_position=initial_position,
                        initial_orientation=initial_orientation,
                        particle_velocity=velocity)
dt=0.01
T=100

solution = sys.RBM_over_time(T,dt) 

datapath ="/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/Velocity"
np.savez(f"{datapath}/solution_T={T}_dt={dt}_shear={gamma_dot}_V={velocity}.npz", **asdict(solution))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d  import Line3DCollection

X_coords = solution.X[:,:3]
x_s, y_s, z_s = solution.X[:,:3].T

points = np.array([x_s, y_s, z_s]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create the colored line
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection='3d')

norm = plt.Normalize(vmin=np.min(solution.time), vmax=np.max(solution.time))
lc = Line3DCollection(segments, cmap='plasma', norm=norm)
lc.set_array(solution.time)
lc.set_linewidth(2)

ax.add_collection3d(lc)

# Colorbar
cbar = fig.colorbar(lc, ax=ax, pad=0.1)
cbar.set_label("Time [s]", rotation=-90, labelpad=10)

ax.set_title(f"Swimmer Position Over Time with $\\dot{{\\gamma}}$={gamma_dot} s$^{{-1}}$, $V_{{\\text{{Swimmer}}}}={velocity}$ $\\mu$m s$^{{-1}}$")

# Axis labels
ax.set_xlabel(r'$x$ [$\mu$m]',labelpad=15)
ax.set_ylabel(r'$y$ [$\mu$m]',labelpad=15)
ax.set_zlabel(r'$z$ [$\mu$m]',labelpad=15)
# ax.view_init(elev=20, azim=10)

plt.tight_layout()

def set_axes_equal(ax):

        xs = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        ranges = xs[:,1] - xs[:,0]
        centers = np.mean(xs, axis=1)
        radius = 0.5 * ranges.max()
        ax.set_xlim3d(centers[0]-radius, centers[0]+radius)
        ax.set_ylim3d(centers[1]-radius, centers[1]+radius)
        ax.set_zlim3d(centers[2]-radius, centers[2]+radius)
        ax.zaxis.set_rotate_label(False)


set_axes_equal(ax)

fig.savefig(f"{plot_path}/trajectory_shear_rate={gamma_dot}_velocity={velocity}.pdf")
fig.savefig(f"{plot_image_path}/trajectory_shear_rate={gamma_dot}_velocity={velocity}.png",dpi=600)
