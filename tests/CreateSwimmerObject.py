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

R = coords.reshape(int(len(coords)/20),20,3)/7 * 12
R2 = coords_2.reshape(int(len(coords_2)/20),20,3)/ 7 *12
# frame =5

dt=400*10**(-6)
# dt=0.00004
V = (np.roll(R, -1, axis=0) - 
     np.roll(R,  1, axis=0)) / (2*dt)

V2 = (np.roll(R2, -1, axis=0) - 
     np.roll(R2,  1, axis=0)) / (2*dt)




flagellum_1 = []
flagellum_2 = []

N_frames = len(R)-1

mesh = bem.Mesh(chlamy_path)
# mesh.plot_mesh()
# import matplotlib.pyplot as plt
# plt.show()

# loop over all frames to create flagellum objects
for frame in range(N_frames):
    frame = frame % N_frames

    R[frame]=(rotmat @ (R[frame]-R[frame][0]).T ).T +np.array([6,2,0])

    R2[frame]=(rotmat @ (R2[frame]-R2[frame][0]).T  ).T+np.array([6,-2,0])

    v= (rotmat @ V[frame].T ).T
    v2=(rotmat @ V2[frame].T).T

    flag1 = bem.SlenderCoordinates(R[frame],v, flagellum_length=12,flagellum_radius=0.1)
    
    flag2 = bem.SlenderCoordinates(R2[frame],v2, flagellum_length=12,flagellum_radius=0.1)

    flagellum_1.append(flag1)
    flagellum_2.append(flag2)

# ===================Create swimmer object=====================


chlamy = bem.FreeSwimmer(mesh,
                        flagellum_1=flagellum_1,flagellum_2=flagellum_2, viscosity=1e-3)
# =============================================================


# ===================Save option 1=====================

# Save swimmer object without results (large file)
with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform_stepsize=1_quat.pkl", "wb") as f:
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