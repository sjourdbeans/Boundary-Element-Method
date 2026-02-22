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
chlamy_path       = "/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/Chlamy/chlamy_N=320.mat"


cell = waveformfile["Cell"]
thetar = cell["thetar"].item()[0][0]
thetal = cell["thetal"].item()[0][0]
# print(thetal)

phi_body = cell["phi_body"].item()[0][0]
# print(phi_body)

xbase =-5#  cell["dist_base"].item()[0][0]
# print(xbase)
ybase = 0
zbase = 0



lf = waveformfile["lf0"][0,0]
# lf = 12
fps = waveformfile["fps"][0]
dt = 1 /fps


def find_flow(t: float, x: np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    # no shear flow
    gamma_dot=0

    U = np.zeros(3)

    U[0] = 0
    U[1] = 0
    U[2] = 0

    # Background vorticity
    W = np.zeros(3)  

    W[0] = 0
    W[1] = 0
    W[2] = -gamma_dot

    # Rate of strain tensor
    E = gamma_dot/2*np.array([[0,1,0],
                            [1,0,0],
                            [0,0,0]])
    return U, W, E



flagellum_1 = []
flagellum_2 = []

N_frames = len(waveformfile["kappasave"])

Rotmat = lambda angle: np.array([[1, 0, 0],
                                 [ 0, np.cos(angle), -np.sin(angle)],
                                 [0, np.sin(angle), np.cos(angle)]])

# loop over all frames to create flagellum objects
for frame in range(N_frames):
    # frame=15
    # Set flagellum velocities
    # velx_1 = waveformfile["velx0"][frame,0] * lf
    # vely_1 = waveformfile["vely0"][frame,0] * lf
    # velz_1 = np.zeros_like(vely_1) * lf

    velx_1 =- waveformfile["velx0"][frame,0] * lf
    vely_1 =- waveformfile["vely0"][frame,0] * lf
    velz_1 = np.zeros_like(vely_1) * lf

    vel_1 = np.vstack([velx_1, vely_1, velz_1]).T #* mu

    vels_1=np.zeros_like(vel_1)

    # velx_2 = waveformfile["velx0"][frame,1] * lf
    # vely_2 = -waveformfile["vely0"][frame,1] * lf
    # velz_2 = np.zeros_like(vely_1) * lf

    velx_2 = -waveformfile["velx0"][frame,1] * lf
    vely_2 = waveformfile["vely0"][frame,1] * lf
    velz_2 = np.zeros_like(vely_1) * lf

    vel_2 = np.vstack([velx_2, vely_2, velz_2]).T #* mu
    vels_2 =np.zeros_like(vel_2)
    # set flagellum shapes and positions
    base_position_1 = np.array([xbase , ybase, zbase]) 
    curv_1 = waveformfile["kappasave"][frame,0,1:] 
    theta_0_1 = waveformfile["kappasave"][frame,0,0]

    base_position_2 = base_position_1 
    curv_2 = -waveformfile["kappasave"][frame,1,1:] 
    theta_0_2 = waveformfile["kappasave"][frame,1,0]

    initial_angle_1 =  np.pi- (thetal - phi_body) + theta_0_1
    initial_angle_2 =  np.pi- (thetar - phi_body) - theta_0_2

    #=======Symmetric flagella=========


    # velx_1 = waveformfile["velx0"][frame,0] * lf
    # vely_1 = waveformfile["vely0"][frame,0] * lf
    # velz_1 = np.zeros_like(vely_1) * lf

    # vel_1 = np.vstack([velx_1, vely_1, velz_1]).T #* mu

    # vels_1=np.zeros_like(vel_1)

    # velx_2 = velx_1
    # vely_2 = -vely_1
    # velz_2 = np.zeros_like(vely_1) * lf

    # vel_2 = np.vstack([velx_2, vely_2, velz_2]).T #* mu
    # vels_2 =np.zeros_like(vel_2)
    # # set flagellum shapes and positions
    # base_position_1 = np.array([xbase , ybase, zbase]) 
    # curv_1 = waveformfile["kappasave"][frame,0,1:] 
    # theta_0_1 = waveformfile["kappasave"][frame,0,0]

    # base_position_2 = base_position_1 
    # curv_2 = -curv_1
    # # theta_0_2 = waveformfile["kappasave"][frame,1,0]

    # initial_angle_1 =  - (thetal - phi_body) + theta_0_1
    # initial_angle_2 = - initial_angle_1
    # initial_angle_2 =  - (thetal - phi_body) - theta_0_1
    #==================================


    tors_1 = np.zeros_like(curv_1)
    # tors_1[0]=np.pi
    tors_2 = np.zeros_like(curv_1)
    # tors_2[0]=-np.pi

    # vel_1 = (Rotmat(np.pi/2) @ vel_1.T).T
    # vel_2 = (Rotmat(np.pi/2) @ vel_2.T).T


    flag1 = bem.SlenderCurvTors(curv_1,tors_1,
                           theta_0=initial_angle_1,
                           flagellum_length=lf,
                           base_position=base_position_1,
                           velocity=vel_1,
                           flagellum_radius=0.1,smin=0)
    
    flag2 = bem.SlenderCurvTors(curv_2, tors_2,
                            theta_0=initial_angle_2,
                            flagellum_length=lf,
                            base_position=base_position_1,
                            velocity=vel_2,
                            flagellum_radius=0.1,smin=0)
    
    # flag1.r = (Rotmat(np.pi/2) @ flag1.r.T ).T
    # flag1.tangents = (Rotmat(np.pi/2) @flag1.tangents.T ).T
    

    # flag2.r = (Rotmat(np.pi/2) @ flag2.r.T).T
    # flag2.tangents = (Rotmat(np.pi/2) @ flag2.tangents.T).T

    flagellum_1.append(flag1)   
    flagellum_2.append(flag2)

# ===================Create swimmer object=====================
mesh = bem.Mesh(chlamy_path)

chlamy = bem.FreeSwimmer(mesh,
                        flagellum_1=flagellum_1,flagellum_2=flagellum_2)
# =============================================================


# ===================Save option 1=====================

# Save swimmer object without results (large file)
with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_2D_waveform.pkl", "wb") as f:
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