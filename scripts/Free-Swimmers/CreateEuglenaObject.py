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



waveformfile      = loadmat("/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/Euglena/Euglena_waveform_half_frames.mat")
euglena_path       = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/Euglena/Euglena_Rossi_N=320.mat"


# print(phi_body)

xbase =22
ybase = 1.5
zbase = -1.5



lf = 28 



r1 = waveformfile["r1"]
r2 = waveformfile["r2"]
r3 = waveformfile["r3"]



R = np.array([r1,r2,r3])*lf

N_frames = np.shape(r1)[-1]

dt = 0.025/N_frames



# V = np.gradient(R, dt,axis=2)
V = (np.roll(R, -1, axis=2) - 
     np.roll(R,  1, axis=2)) / (2*dt)
# print(r1[:,0])
# print(np.sum(np.linalg.norm(R[:,1:,:]-R[:,:-1,:], axis=0), axis=0))

flagellum= []




Rotmat = lambda angle: np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle) , np.cos(angle), 0],
                                 [0,0,1]])

Rx = lambda angle: np.array([[1,0,0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])

Ry = lambda angle: np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0,1,0],
                             [-np.sin(angle), 0, np.cos(angle)]])

import matplotlib.pyplot as plt
# loop over all frames to create flagellum objects
base_position = np.array([xbase , ybase, zbase]) 
# rotate_flag = np.pi/2 + 1.5*np.pi/6  #+np.pi/6
# angle_x = 0#np.pi/10
# angle_y = np.pi/6
# print(N_frames)


rotate_flag = np.pi/2 + 1.5*np.pi/6  #+np.pi/6
angle_x = 0#np.pi/10
angle_y = np.pi/6 
mesh = bem.Mesh(euglena_path)

for frame in range(N_frames):

    r = (Ry(angle_y) @ Rx(angle_x) @ Rotmat(rotate_flag) @  R[:,:,frame]).T + base_position 
    # print(r)
    v = (Ry(angle_y) @ Rx(angle_x) @ Rotmat(rotate_flag) @  V[:,:,frame]).T
    # print(v)
    # print(np.max(np.linalg.norm(v, axis =1)))
    



    flag = bem.SlenderCoordinates(r, velocity=v, flagellum_length=lf,flagellum_radius=0.35, smin=0.1)
    # print(flag.tangents)
    # mesh.plot_mesh()
    # plt.plot(r[:,0], r[:,1], r[:,2], color='r',zorder=3)
    # # plt.quiver(r[:,0], r[:,1], r[:,2], v[:,0], v[:,1], v[:,2], length=0.5, color='b', zorder=4)
    # plt.show()

    flagellum.append(flag)

# ===================Create swimmer object=====================

# mesh.plot_mesh()
# # fig.set_size_inches(20,20)
# plt.plot(r[:,0], r[:,1], r[:,2], color='r',zorder=3)
# plt.quiver(r[:,0], r[:,1], r[:,2], v[:,0], v[:,1], v[:,2], length=0.5, color='b', zorder=4)
# plt.show()

euglena = bem.FreeSwimmer(mesh,
                        flagellum_1=flagellum)
# # =============================================================


# # ===================Save option 1=====================

# Save swimmer object without results (large file)
with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Euglena/Rossi/Free/Euglena_N=320_full_frames.pkl", "wb") as f:
    pickle.dump(euglena, f)


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