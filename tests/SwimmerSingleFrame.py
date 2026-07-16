from scipy.io import loadmat
import bemsolver as bem
import numpy as np
import os
import pickle

import matplotlib as mpl

from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import matplotlib.pyplot as plt

# ==============Plotting settings==================

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"]= "Palatino"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"
mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13

#===================================================



chlamy_path       = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/Chlamy/chlamy_N=320.mat"
# chlamy_path       = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/Chlamy/sphere_chlamy_N=320.mat"

movie_data="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/Chlamy-3D/CC125_4_2_T_avg_int.txt"
movie_data_2="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/Chlamy-3D/CC125_4_2_C_avg_int.txt"
coords =np.loadtxt(movie_data)
coords_2 =np.loadtxt(movie_data_2)


theta=np.pi/2 
rotmat=np.array([[np.cos(theta), -np.sin(theta),0],
                 [np.sin(theta), np.cos(theta),0],
                 [0,0,1]])


Rx = lambda angle: np.array([[1,0,0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
lf=12



R = coords.reshape(int(len(coords)/20),20,3)/7 * lf
# R2 = coords_2.reshape(int(len(coords_2)/20),20,3)/ 7 *lf
# R[:,:,2]*=0#1#1.5
# R2[:,:,2]*=0#1#1.5

R2 = R.copy()
R2[:,:,0] *= -1
R2[:,:,2] *= -1


angle_x =0# -5*np.pi/8

def find_flow(t: float, x: np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    # no shear flow
    gamma_dot=-2

    U = np.zeros(3)

    U[0] = 0#-1000 
    U[1] = 0
    U[2] = 0

    # Background vorticity
    W = np.zeros(3)  

    W[0] = 0
    W[1] = 0#gamma_dot
    W[2] =-gamma_dot

    # Rate of strain tensor
    E = gamma_dot/2*np.array([[0,1,0],
                            [1,0,0],
                            [0,0,0]])
    
    E=Rx(angle_x)@E@Rx(angle_x).T
    W=Rx(angle_x)@W
    return U, W, E


mesh = bem.Mesh(chlamy_path)

frame=15

R[frame]=(rotmat @ (R[frame]-R[frame][0]).T ).T +np.array([6,2,0])
R2[frame]=(rotmat @ (R2[frame]-R2[frame][0]).T  ).T+np.array([6,-2,0])

flagellum_1 =[bem.SlenderCoordinates(R[frame],np.zeros_like(R[frame]), flagellum_length=lf,flagellum_radius=0.1)]
flagellum_2 =[bem.SlenderCoordinates(R2[frame],np.zeros_like(R2[frame]), flagellum_length=lf,flagellum_radius=0.1)]

chlamy = bem.Swimmer(mesh, flagellum_1, flagellum_2)

solution = chlamy.solve(find_flow, 1)

# fig, ax, cbar = bem.plot_panels_stokes(mesh.panels, np.linalg.norm(solution.psi.reshape((chlamy.mesh.elements,3)),axis=1))

f1 = solution.f1[0]
f1 = f1.reshape(int(len(f1)/3),3)
# f1[:,0]*=0
# f1[:,1]*=0


f2 = solution.f2[0]
f2 = f2.reshape(int(len(f2)/3),3)
# f2[:,0]*=0
# f2[:,1]*=0

# print(f"f1:{np.mean(f1,axis=0)}\n f2: {np.mean(f2,axis=0)}")
# print(f"Total force: {np.mean(f1 + f2,axis=0)}")

print(f"f1:{np.sum(f1,axis=0)}\n f2: {np.sum(f2,axis=0)}")
print(f"Total force: {np.sum(f1+f2,axis=0)}")
U,W,E = find_flow(0, np.zeros(3))
print(f"f1 vorticity force: {np.dot(np.sum(f1,axis=0),W/np.linalg.norm(W))}")
print(f"f2 vorticity force: {np.dot(np.sum(f2,axis=0),W/np.linalg.norm(W))}")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')



frame_length = 4  # microns, adjust for visibility



flagella_lines = {
    "f1": ax.plot([], [], [], color="k", lw=1.5)[0],
    "f2": ax.plot([], [], [], color="k", lw=1.5)[0],
} 

flag_quivers = {
    "1": None,
    "2": None
}

# Colorbar

ax.set_title("Swimmer With flagella")
ax.set_xlabel(r'$x$ [$\mu$m]')
ax.set_ylabel(r'$y$ [$\mu$m]')
ax.set_zlabel(r'$z$ [$\mu$m]')
# ax.view_init(elev=60)

# ------------------------------------------------------
# Keep axes equal
# ------------------------------------------------------
def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ranges = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    max_range = 10*ranges.max() 

    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])


# # Set once
# ax.set_xlim(x_s.min(), x_s.max())
# ax.set_ylim(y_s.min(), y_s.max())
# ax.set_zlim(z_s.min(), z_s.max())
# ax.view_init(elev=20, azim=10)
# set_axes_equal(ax)


if chlamy.mesh.is_mat:
    panels = [chlamy.mesh.panels[1:, :, k] for k in range(chlamy.mesh.elements)]
else:
    panels = [chlamy.mesh.panels[k] for k in range(chlamy.mesh.elements)]



values = -np.linalg.norm(solution.psi.reshape((chlamy.mesh.elements,3)),axis=1)
cmap = plt.get_cmap('viridis')  # You can use any colormap you prefer
norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))  # Normalize to the range of values

# Apply the colormap to your values
colors = cmap(norm(values))


mesh_surf = Poly3DCollection(
    panels,
    facecolor=colors,
    edgecolor=(0, 0, 0, 0.5),
    linewidth=0.1,
    alpha=0.9,
    shade=False
)
# surf.set_edgecolor((0, 0, 0, 0.8))
ax.add_collection3d(mesh_surf)

# def update(frame):
#     global mesh_surf

    
    # ---------------- swimmer state ----------------
    # r = x[frame]                                # swimmer position
    # R = solution.rotation_matrices[frame]      # body -> lab rotation

    # ---------------- basis vectors ------------
    # ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]

for key in flag_quivers:
    if flag_quivers[key] is not None:
        flag_quivers[key].remove()




# ---------------- ellipsoid ----------------
# mesh_surf.remove()



#     mesh_surf = Poly3DCollection(
#     lab_panels,
#     facecolor='lightgray',
#     edgecolor=(0, 0, 0, 0.5),
#     linewidth=0.1,
#     alpha=0.3,
#     shade=False
# )

# ax.add_collection3d(mesh_surf)

# time_frame = int(round(frame % chlamy.N_frames))
# ---------------- flagellum 1 ----------------
r1 = chlamy.flagellum_1[0].r            # (N1, 3), swimmer frame
# r1_lab = (R @ r1.T).T + r                   # rotate + translate

flagella_lines["f1"].set_data(r1[:, 0], r1[:, 1])
flagella_lines["f1"].set_3d_properties(r1[:, 2])

# ---------------- flagellum 2 ----------------
r2 = chlamy.flagellum_2[0].r            # (N2, 3), swimmer frame
# r2_lab = (R @ r2.T).T + r

flagella_lines["f2"].set_data(r2[:, 0], r2[:, 1])
flagella_lines["f2"].set_3d_properties(r2[:, 2])


quiv = 1

flag_quivers["1"] = ax.quiver(
    r1[::quiv, 0], r1[::quiv, 1], r1[::quiv, 2],
    f1[::quiv, 0], f1[::quiv, 1], f1[::quiv, 2],
    color="b",length=1
)

flag_quivers["2"] = ax.quiver(
    r2[::quiv, 0], r2[::quiv, 1], r2[::quiv, 2],
    f2[::quiv, 0], f2[::quiv, 1], f2[::quiv, 2],
    color="b",length=1
)

plt.axis('equal')
# ---------------- camera ----------------
# ax.view_init(azim=30- frame/8)
plt.show()
