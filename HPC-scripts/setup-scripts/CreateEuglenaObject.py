from scipy.io import loadmat
import bemsolver as bem
import numpy as np
import pickle

import matplotlib as mpl



# waveformfile      = loadmat("/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/Euglena/Euglena_waveform_quarter_frames_50_periods.mat")
# waveformfile      =loadmat("/home/sjoerd-buitjes/University/Master-Thesis/Paper-data/Euglena-Antonio/Figure 2 - Source Data & Code/flagellum.mat")

waveformfile = loadmat("/home/sbuitjes/code/Boundary-Element-Method/datafiles/waveform/Euglena/flagellum.mat")
euglena_path       = "/scratch/sbuitjes/meshes/euglena/Euglena_N=1280.mat"

savepath = "/scratch/sbuitjes/swimmer_objects/euglena/euglena_free_1280.pkl"


eug_I = waveformfile["splnoptptI"]
eug_J = waveformfile["splnoptptJ"]
eug_K = waveformfile["splnoptptK"]

R = np.array([eug_I.T , eug_J.T, eug_K.T])
R = R - R[:,0,None]

t = R[:,1:,:]-R[:,:-1,:]
print(np.sum(np.linalg.norm(t, axis=0),axis=0))
R = R/ np.sum(np.linalg.norm(t, axis=0),axis=0)

lf = 40#28
R= R*lf 



xbase =22 #22
ybase = 0#1#1.5
zbase = 0 #-1.5




N_frames = np.shape(R)[-1]
dt = 0.025/N_frames



# V = np.gradient(R, dt,axis=2)
V = (np.roll(R, -1, axis=2) - 
     np.roll(R,  1, axis=2)) / (2*dt)

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



rotate_flag =0# np.pi/2 + 1.5*np.pi/6  #+0.2*np.pi/6
angle_x = 0#np.pi/2#-1.6*np.pi/6
angle_y = np.pi/2#1*np.pi/7
mesh = bem.Mesh(euglena_path)

for frame in range(N_frames):

    r = (Ry(angle_y) @ Rx(angle_x) @ Rotmat(rotate_flag) @  R[:,:,frame]).T + base_position 
    
    v = (Ry(angle_y) @ Rx(angle_x) @ Rotmat(rotate_flag) @  V[:,:,frame]).T
    
    



    flag = bem.SlenderCoordinates(r, velocity=v, flagellum_length=lf,flagellum_radius=0.1, smin=0.1)
    

    flagellum.append(flag)

# ===================Create swimmer object=====================


euglena = bem.FreeSwimmer(mesh,
                        flagellum_1=flagellum)
# # =============================================================


# # ===================Save option 1=====================

# Save swimmer object without results (large file)
with open(savepath, "wb") as f:
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