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
phi_body = cell["phi_body"].item()[0][0]


xbase = - cell["dist_base"].item()[0][0]
ybase = 0
zbase = 0



lf = waveformfile["lf0"][0,0]
fps = waveformfile["fps"][0][0]
dt = 1 /fps




curvatures_2 = -waveformfile["kappasave"][:,1,1:]
thetas_2     =  waveformfile["kappasave"][:,1,0]

curvatures_1 =  waveformfile["kappasave"][:,0,1:]
thetas_1     =  waveformfile["kappasave"][:,0,0]

n_frames, n_el = curvatures_1.shape

T = dt*n_frames *10**3  # one beating period, or whatever physical period you have

t_frames = np.linspace(0, T, n_frames, endpoint=True)

from scipy.interpolate import CubicSpline

# curvatures: shape (n_frames, n_el)
# We build a spline that interpolates along axis=0 (time)
kappa_1_spline = CubicSpline(t_frames, curvatures_1, axis=0)
theta_1_spline = CubicSpline(t_frames, thetas_1, axis=0)
velx_1_spline = CubicSpline(t_frames, -waveformfile["velx0"][:,0]*lf, axis=0)
vely_1_spline = CubicSpline(t_frames, -waveformfile["vely0"][:,0]*lf, axis=0)

kappa_2_spline = CubicSpline(t_frames, curvatures_2, axis=0)
theta_2_spline = CubicSpline(t_frames, thetas_2, axis=0)
velx_2_spline = CubicSpline(t_frames, -waveformfile["velx0"][:,1]*lf, axis=0)
vely_2_spline = CubicSpline(t_frames, waveformfile["vely0"][:,1]*lf, axis=0)

def kappa2_at_time(t):
    # t can be scalar or array
    t_mod = np.mod(t, T) 
    return kappa_2_spline(t)   

def theta2_at_time(t):
    t_mod = np.mod(t, T) 
    return theta_2_spline(t)

def vel2_at_time(t):
    t_mod = np.mod(t, T) 
    velx = velx_2_spline(t)
    vely = vely_2_spline(t)
    velz = np.zeros_like(velx)
    return np.vstack([velx, vely, velz]).T

def kappa1_at_time(t):
    # t can be scalar or array
    t_mod = np.mod(t, T) 
    return kappa_1_spline(t)   

def theta1_at_time(t):
    t_mod = np.mod(t, T) 
    return theta_1_spline(t)

def vel1_at_time(t):
    t_mod = np.mod(t, T) 
    velx = velx_1_spline(t)
    vely = vely_1_spline(t)
    velz = np.zeros_like(velx)
    return np.vstack([velx, vely, velz]).T

# time_factor determines the amount of frames
time_factor:int = 3

time = np.linspace(0, dt*n_frames*10**3, time_factor*n_frames)


flagellum_1 = []
flagellum_2 = []

N_frames = len(waveformfile["kappasave"])

# loop over all frames to create flagellum objects
for i , t in enumerate(time):


    vel_1 = vel1_at_time(t)


    vel_2 = vel2_at_time(t)

    # set flagellum shapes and positions
    base_position_1 = np.array([xbase , ybase, zbase]) 
    curv_1 = kappa1_at_time(t) 
    theta_0_1 = theta1_at_time(t)

    base_position_2 = -base_position_1 
    curv_2 = kappa2_at_time(t)
    theta_0_2 = theta2_at_time(t)

    initial_angle_1 = np.pi - (thetal - phi_body) + theta_0_1
    initial_angle_2 = np.pi - (thetar - phi_body) - theta_0_2


    tors_1 = np.zeros_like(curv_1)


    flag1 = bem.SlenderCurvTors(curv_1,tors_1,
                           theta_0=initial_angle_1,
                           flagellum_length=lf,
                           base_position=base_position_1,
                           velocity=vel_1)
    
    flag2 = bem.SlenderCurvTors(curv_2,tors_1,
                            theta_0=initial_angle_2,
                            flagellum_length=lf,
                            base_position=base_position_1,
                            velocity=vel_2)
    flagellum_1.append(flag1)
    flagellum_2.append(flag2)

# ===================Create swimmer object=====================
mesh = bem.Mesh(chlamy_path)

chlamy = bem.FreeSwimmer(mesh,
                        flagellum_1=flagellum_1,flagellum_2=flagellum_2)
# =============================================================


# ===================Save option 1=====================

# Save swimmer object without results (large file)
with open(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_time_{time_factor}.pkl", "wb") as f:
    pickle.dump(chlamy, f)

# solution = chlamy.RBM_over_time(dt, find_flow)

# =====================================================


# solve
# solution = chlamy.solve(find_flow, dt)

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