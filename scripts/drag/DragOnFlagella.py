from scipy.io import loadmat
import bemsolver as bem
import numpy as np
import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import imageio.v2 as imageio


import matplotlib as mpl

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
mpl.rcParams["font.family"]= "Palatino"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"
mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13

waveformfile      = loadmat("/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/lib02_1_90_2019-06-28_1640.mat")


frame = 20


cell = waveformfile["Cell"]
thetar = cell["thetar"].item()[0][0]
thetal = cell["thetal"].item()[0][0]
phi_body = cell["phi_body"].item()[0][0]


xbase = 0 # - cell["dist_base"].item()[0][0]
ybase = 0
zbase = 0



lf = waveformfile["lf0"][0,0]
fps = waveformfile["fps"]
dt = 1 /fps


gamma_dot=0

U = np.zeros(3)

U[0] = 1#gamma_dot * x[1]
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




Nx=200 + 1
Ny=200 + 1
xlim=5
ylim=5

x = np.linspace(-7.5, 7.5, Nx)
y = np.linspace(0, 15, Ny)

xg, yg =np.meshgrid(x,y)
zg=np.zeros_like(xg)

xg = xg.ravel()
yg = yg.ravel()
zg = zg.ravel()

Ng = np.shape(xg)[0]

points = np.vstack((xg, yg, zg)).T



base_position = -np.array([xbase , ybase, zbase]) 


initial_angle =np.pi/2 

curv_zero = np.zeros(10)
tors_zero = np.zeros_like(curv_zero)

flag = bem.SlenderCurvTors(curv_zero,tors_zero,theta_0=initial_angle,base_position=base_position, smin=0,rho_0=0)
print(flag.r)

M = flag.construct_mobility_matrix()


RHS  = flag.set_boundary_condition(U,W,E) 

f = np.linalg.solve(M, -RHS)

fq = f.reshape(int(len(f)/3),3)

# np.savetxt("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/drag/flagella-drag/force_density.txt",
#             np.hstack((flag.r,fq)),delimiter=",", header="rx, ry, rz, fx, fy, fz")


K = flag.calc_interaction(points)
u_field  = K @ f 

u_field = u_field.reshape(Ng, 3)

ux = u_field[:,0].reshape(Ny,Nx)
uy = u_field[:,1].reshape(Ny,Nx)
uz = u_field[:,2].reshape(Ny,Nx)

U_magnitude = np.sqrt(ux**2 + uy**2 +uz**2)

quiver_density=10

Ux_quiver = ux[::quiver_density, ::quiver_density]
Uy_quiver = uz[::quiver_density, ::quiver_density]

x_quiver = x[::quiver_density]
y_quiver = y[::quiver_density]


vmax = 50* flag.flagellum_length


norm = colors.Normalize(vmin=0, vmax=np.average(U_magnitude)*2.5)
fig = plt.figure(figsize=(12, 10))
c = plt.pcolormesh(x, y, U_magnitude, shading='auto', cmap='viridis',norm=norm)
plt.title(f"$t$={frame*dt} s")        

plt.plot(flag.r[:,0],flag.r[:,1],color='red')

plt.quiver(x_quiver, y_quiver, Ux_quiver, Uy_quiver,
                color='white', headlength=4, headwidth=2,scale_units='xy',scale=2)
# plt.quiver(flag.r[:,0],flag.r[:,1], fq[:,0], fq[:,1],color="pink")
plt.colorbar(c,label=r'$|\mathbf{U}_{\text{field}}|$ [$\mu$m/s]')
plt.xlabel(f'$x$ [$\\mu$m]')
plt.ylabel(f'$y$ [$\\mu$m]')
plt.title('Flow magnitude and direction')
plt.axis('equal')

# fig = plt.figure(figsize=(6,5))
# ax = fig.add_subplot(projection='3d')
# ax.plot(flag.r[:,0],flag.r[:,1],flag.r[:,2], color='r')
# ax.view_init( azim=60)
# ax.set_xlabel(r'$x$ [$\mu$m]',labelpad=15)
# ax.set_ylabel(r'$y$ [$\mu$m]',labelpad=15)
# ax.set_zlabel(r'$z$ [$\mu$m]',labelpad=15)
# # ax.view_init(elev=0, azim=60)
# ax.set_xlim(-10,10)
# ax.set_ylim(0,20)
# ax.set_zlim(0,20)
plt.show()




