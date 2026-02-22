from scipy.io import loadmat
import bemsolver as bem
import numpy as np
import os
import pickle

import matplotlib as mpl

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



waveformfile      = loadmat("/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/lib02_1_90_2019-06-28_1640.mat")
chlamy_path       = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/Chlamy/chlamy_N=1280.mat"
# chlamy_path       = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/Chlamy/sphere_chlamy_N=320.mat"

movie_data="/home/sjoerd-buitjes/University/Master-Thesis/Paper-data/Holographic_microscopy_Chlamy/Movie data/CC125_4_2_T_avg_int.txt"
movie_data_2="/home/sjoerd-buitjes/University/Master-Thesis/Paper-data/Holographic_microscopy_Chlamy/Movie data/CC125_4_2_C_avg_int.txt"
coords =np.loadtxt(movie_data)
coords_2 =np.loadtxt(movie_data_2)


theta=np.pi/2
rotmat=np.array([[np.cos(theta), -np.sin(theta),0],
                 [np.sin(theta), np.cos(theta),0],
                 [0,0,1]])

R = coords.reshape(int(len(coords)/20),20,3)
# R2 = coords_2.reshape(int(len(coords_2)/20),20,3)
R2 = R.copy()
R2[:,:,0] *= -1
# R2[:,:,2] *= -1
R[:,:,2] *= -1


T1 = R[:, 1:,:] - R[:,:-1, :]
T1 = T1/ np.linalg.norm(T1, axis=2, keepdims=True)

T2 = R2[:, 1:,:] - R2[:,:-1, :]
T2 = T2/ np.linalg.norm(T2, axis=2, keepdims=True)

theta_1_map = np.arctan2(T1[:,:,1], T1[:,:,0])
phi_1_map = np.arctan2(T1[:,:,2], np.sqrt(T1[:,:,0]**2 + T1[:,:,1]**2))

theta_2_map = np.arctan2(T2[:,:,1], T2[:,:,0])
phi_2_map = np.arctan2(T2[:,:,2], np.sqrt(T2[:,:,0]**2 + T2[:,:,1]**2))
# frame =5

dt=400*10**(-6)
# dt=0.00004
factor = 1
Nt= len(R)


r1_map = np.zeros((factor*Nt, len(R[0])-1, 3))
r2_map = np.zeros((factor*Nt, len(R2[0])-1, 3))

scale_out_of_plane = 1
new_flag_length = 12

for time_frame in range(factor*Nt):

    frame = time_frame % Nt


    curve_1 = bem.SlenderAngles(theta_1_map[time_frame,:], scale_out_of_plane*phi_1_map[time_frame,:], flagellum_length=new_flag_length,flagellum_radius=0.1)
    curve_2 = bem.SlenderAngles(theta_2_map[time_frame,:], scale_out_of_plane*phi_2_map[time_frame,:], flagellum_length=new_flag_length,flagellum_radius=0.1)

    r1_map[time_frame,:,:] = curve_1.r
    r2_map[time_frame,:,:] = curve_2.r


V = 0*(np.roll(r1_map, -1, axis=0) - 
     np.roll(r1_map,  1, axis=0)) / (2*dt)

V2 = 0*(np.roll(r2_map, -1, axis=0) - 
     np.roll(r2_map,  1, axis=0)) / (2*dt)

# V = np.gradient(r1_map, dt, axis=0)
# V =(r1_map[1:]-r1_map[:-1])/dt
# V = np.vstack((V, (r1_map[-1]-r1_map[0]).reshape(1,len(r1_map[0]),3)/dt))

# V2 = np.gradient(r2_map, dt, axis=0)
# V2 =(r2_map[1:]-r2_map[:-1])/dt
# V2 = np.vstack((V2, (r2_map[-1]-r2_map[0]).reshape(1,len(r2_map[0]),3)/dt))

flagellum_1 = []
flagellum_2 = []

N_frames = len(R)-1

mesh = bem.Mesh(chlamy_path)
# mesh.plot_mesh()
# import matplotlib.pyplot as plt
# plt.show()

# loop over all frames to create flagellum objects
for frame in range(N_frames)[::3]:
    frame = frame % N_frames

    r1_map[frame]=(rotmat @ (r1_map[frame]-r1_map[frame][0]).T ).T +np.array([6,2,0])

    r2_map[frame]=(rotmat @ (r2_map[frame]-r2_map[frame][0]).T  ).T+np.array([6,-2,0])

    v= (rotmat @ V[frame].T ).T
    v2=(rotmat @ V2[frame].T).T

    flag1 = bem.SlenderCoordinates(r1_map[frame],v, flagellum_length=new_flag_length,flagellum_radius=0.1)
    
    flag2 = bem.SlenderCoordinates(r2_map[frame],v2, flagellum_length=new_flag_length,flagellum_radius=0.1)

    flagellum_1.append(flag1)
    flagellum_2.append(flag2)

# ===================Create swimmer object=====================


chlamy = bem.FreeSwimmer(mesh,
                        flagellum_1=flagellum_1,flagellum_2=flagellum_2, viscosity=1e-3)
# =============================================================


# ===================Save option 1=====================

# Save swimmer object without results (large file)
with open(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/scale-out-of-plane/chlamy_free_3d_waveform_scale={scale_out_of_plane}_zero_vel.pkl", "wb") as f:
    pickle.dump(chlamy, f)


# solution = chlamy.RBM_over_time(dt, find_flow)

# =====================================================


# ===================Save option 2=====================

# # save only the solution (small file)
# with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/RBM/chlamy_solution.pkl", "wb") as f:
#     pickle.dump(solution, f)

# =====================================================


# ===================Save option 3=====================

# save swimmer object with results (large file)
# with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/test/chlamy_with_solution.pkl", "wb") as f:
#     pickle.dump(chlamy, f)

# =====================================================