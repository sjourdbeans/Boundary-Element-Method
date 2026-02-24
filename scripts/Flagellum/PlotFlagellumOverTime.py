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

waveformfile      = loadmat("/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/Chlamy-2D/lib02_1_90_2019-06-28_1640.mat")


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

U[0] = 0#gamma_dot * x[1]
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




Nx=200 + 1
Ny=200 + 1
xlim=5
ylim=5

x = np.linspace(-15, 15, Nx)
y = np.linspace(0, 30, Ny)

xg, yg =np.meshgrid(x,y)
zg=np.zeros_like(xg)

xg = xg.ravel()
yg = yg.ravel()
zg = zg.ravel()

Ng = np.shape(xg)[0]

points = np.vstack((xg, yg, zg)).T

savepath = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/videos/Flagella/flagellum_test_new_coord.mp4"
tmp_dir = "frames"

os.makedirs(tmp_dir, exist_ok=True)
frame_files = []


for frame in range(len(waveformfile["kappasave"])):

    velx_2 = -waveformfile["velx0"][frame,1] * lf
    vely_2 = waveformfile["vely0"][frame,1] * lf
    velz_2 = np.zeros_like(vely_2) * lf

    vel_2 = np.vstack([velx_2, vely_2, velz_2]).T

    base_position_2 = -np.array([xbase , ybase, zbase]) 
    curv_2 = -waveformfile["kappasave"][frame,1,1:] 
    theta_0_2 = waveformfile["kappasave"][frame,1,0]
    tors_2 = np.zeros_like(curv_2)

    initial_angle_2 =np.pi -(thetar - phi_body) - theta_0_2 
    
    curv_test = 2*np.ones(30)
    tors_test = np.zeros_like(curv_test)

    flag = bem.SlenderCurvTors(curv_2,tors_2,theta_0=initial_angle_2,flagellum_length=lf,base_position=base_position_2, smin=0,
                            velocity=vel_2)
    
    

    M = flag.construct_mobility_matrix()


    RHS  = flag.set_boundary_condition(U,W,E)

    f = np.linalg.solve(M, -RHS)

    fq = f.reshape(int(len(f)/3),3)


    K = flag.calc_interaction(points)
    u_field  = K @ f

    u_field = u_field.reshape(Ng, 3)

    ux = u_field[:,0].reshape(Ny,Nx)
    uy = u_field[:,1].reshape(Ny,Nx)
    uz = u_field[:,2].reshape(Ny,Nx)

    U_magnitude = np.sqrt(ux**2 + uy**2 +uz**2)

    quiver_density=10

    Ux_quiver = ux[::quiver_density, ::quiver_density]
    Uy_quiver = uy[::quiver_density, ::quiver_density]

    x_quiver = x[::quiver_density]
    y_quiver = y[::quiver_density]


    vmax = 100* flag.flagellum_length


    norm = colors.Normalize(vmin=0, vmax=vmax)
    fig = plt.figure(figsize=(12, 10))
    c = plt.pcolormesh(x, y, U_magnitude, shading='auto', cmap='viridis',norm=norm)
    plt.title(f"$t$={frame*dt} s")        

    plt.plot(flag.r[:,0],flag.r[:,1],color='red')

    plt.quiver(x_quiver, y_quiver, Ux_quiver, Uy_quiver,
                    color='white', headlength=4, headwidth=2,scale_units='xy')
    # plt.quiver(flag2.r[:,0],flag2.r[:,1], fq[:,0], fq[:,1],color="pink")
    plt.colorbar(c,label=r'$|\mathbf{U}_{\text{field}}|$ [$\mu$m/s]')
    plt.xlabel(f'$x$ [$\\mu$m]')
    plt.ylabel(f'$y$ [$\\mu$m]')
    plt.title('Flow magnitude and direction')
    plt.axis('equal')
    plt.xlim(-15,15)
    plt.ylim(0,30)


    frame_path = os.path.join(tmp_dir, f"frame_{frame:04d}.png")
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)

    frame_files.append(frame_path)


print("Creating video...")
frames = [imageio.imread(f) for f in frame_files]
imageio.mimsave(savepath, frames, fps=10)

print(f"Animation saved to {savepath}")
import shutil
shutil.rmtree(tmp_dir)


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d  import Line3DCollection

# fig = plt.figure(figsize=(6,5))
# ax = fig.add_subplot(projection='3d')
# plt.plot(flag2.r[:,0],flag2.r[:,1],flag2.r[:,2], color='r')
# plt.xlabel("x")
# plt.ylabel("y")
# # plt.zlabel("z")
# # plt.plot(r[:,0],r[:,1],'o-')
# # plt.quiver(r[:,0],r[:,1], t[:,0], t[:,1])
# plt.show()