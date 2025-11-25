from scipy.io import loadmat
import bemsolver as bem
import numpy as np

waveformfile      = loadmat("/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/lib02_1_90_2019-06-28_1640.mat")

frame = 10


cell = waveformfile["Cell"]
thetar = cell["thetar"].item()[0][0]
phi_body = cell["phi_body"].item()[0][0]


xbase = 2.5 #- cell["dist_base"].item()[0][0]
ybase = 0
zbase = 0



lf = waveformfile["lf0"][0,0]
fps = waveformfile["fps"]
dt = 1 /fps

velx_1 = waveformfile["velx0"][frame,0] * lf
vely_1 = waveformfile["vely0"][frame,0] * lf
velz_1 = np.zeros_like(vely_1) * lf

velx_2 = waveformfile["velx0"][frame,1] * lf
vely_2 = waveformfile["vely0"][frame,1] * lf
velz_2 = np.zeros_like(vely_1) * lf

base_position = np.array([xbase, ybase, zbase])
curv_1 = waveformfile["kappasave"][frame,0,1:] 
theta_0 = waveformfile["kappasave"][frame,0,0]

initial_angle_0 = np.pi - (thetar - phi_body) - theta_0


tors_1 = np.zeros_like(curv_1)

vel_1  = np.vstack((velx_1, vely_1, velz_1))


# points =  np.arange(0, 30,1).reshape((10,3))


flag = bem.SlenderBody(curv_1,tors_1,theta_0=initial_angle_0)
K = flag.calc_mobility()

print(K)

# r, t = flag.r, flag.tangents

# r= r[1:]
# t=t[1:]

# import matplotlib.pyplot as plt


# plt.plot(r[:,0],r[:,1],'o-')
# plt.quiver(r[:,0],r[:,1], t[:,0], t[:,1])
# plt.show()
# flag.calc_flagella_mobility()