from scipy.io import loadmat
import bemsolver as bem
import numpy as np
import os
import pickle
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import imageio.v2 as imageio


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
savepath          = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/videos/Chlamy/Chlamy_new.mp4"

cell = waveformfile["Cell"]
thetar = cell["thetar"].item()[0][0]
thetal = cell["thetal"].item()[0][0]
phi_body = cell["phi_body"].item()[0][0]


xbase = - cell["dist_base"].item()[0][0]
ybase = 0
zbase = 0



lf = waveformfile["lf0"][0,0]
fps = waveformfile["fps"][0]
dt = 1 /fps


def find_flow(t: float, x: np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    # no shear flow
    gamma_dot=0

    U = np.zeros(3)

    U[0] = 1000 
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
    return U, W, E




flagellum_1 = []
flagellum_2 = []

N_frames = len(waveformfile["kappasave"])

# loop over all frames to create flagellum objects
for frame in range(N_frames):

    # Set flagellum velocities
    velx_1 = -waveformfile["velx0"][frame,0] * lf
    vely_1 = -waveformfile["vely0"][frame,0] * lf
    velz_1 = np.zeros_like(vely_1) * lf

    vel_1 = np.vstack([velx_1, vely_1, velz_1]).T


    velx_2 = -waveformfile["velx0"][frame,1] * lf
    vely_2 = waveformfile["vely0"][frame,1] * lf
    velz_2 = np.zeros_like(vely_1) * lf

    vel_2 = np.vstack([velx_2, vely_2, velz_2]).T

    # set flagellum shapes and positions
    base_position_1 = np.array([xbase , ybase, zbase]) 
    curv_1 = waveformfile["kappasave"][frame,0,1:] 
    theta_0_1 = waveformfile["kappasave"][frame,0,0]

    base_position_2 = -base_position_1 
    curv_2 = -waveformfile["kappasave"][frame,1,1:] 
    theta_0_2 = waveformfile["kappasave"][frame,1,0]

    initial_angle_1 = np.pi - (thetal - phi_body) + theta_0_1
    initial_angle_2 = np.pi - (thetar - phi_body) - theta_0_2


    tors_1 = np.zeros_like(curv_1)


    flag1 = bem.SlenderBody(curv_1,tors_1,
                           theta_0=initial_angle_1,
                           flagellum_length=lf,
                           base_position=base_position_1,
                           velocity=vel_1)
    
    flag2 = bem.SlenderBody(curv_2,tors_1,
                            theta_0=initial_angle_2,
                            flagellum_length=lf,
                            base_position=base_position_1,
                            velocity=vel_2)
    flagellum_1.append(flag1)
    flagellum_2.append(flag2)

# ===================Create swimmer object=====================
mesh = bem.Mesh(chlamy_path)

chlamy = bem.Swimmer(mesh,
                     flagellum_1=flagellum_1,flagellum_2=flagellum_2)
# =============================================================


# ===================Save option 1=====================

# # Save swimmer object without results (large file)
# with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/test/chlamy_swimmer.pkl", "wb") as f:
#     pickle.dump(chlamy, f)

# =====================================================


# solve
solution = chlamy.solve(find_flow, dt)

# ===================Save option 2=====================

# # save only the solution (small file)
# with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/test/result/chlamy_solution.pkl", "wb") as f:
#     pickle.dump(solution, f)

# =====================================================


# ===================Save option 3=====================

# save swimmer object with results (large file)
# with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/test/chlamy_with_solution.pkl", "wb") as f:
#     pickle.dump(chlamy, f)

# =====================================================



# ============Visualise Flowfield around swimmer============

# NOTE: you could include the loop below into the loop above. However, the loop below is only for visualisation purposes,
#       which is why it is kept separate from the simulation loop above.

# you could also load a pre calculated bem.FlowStokes object here instead of calculating it again every run.

Nx=200 + 1
Ny=200 + 1
xlim=5
ylim=5

x = np.linspace(-20, 20, Nx)
y = np.linspace(-20, 20, Ny)

xg, yg =np.meshgrid(x,y)
zg=np.zeros_like(xg)

xg = xg.ravel()
yg = yg.ravel()
zg = zg.ravel()

Ng = np.shape(xg)[0]

points = np.vstack((xg, yg, zg)).T


tmp_dir = "frames"

os.makedirs(tmp_dir, exist_ok=True)
frame_files = []

#=====open pre-saved flowstokes object=======
# with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/test/flowstokes_object.pkl", 'rb') as file:
#     flow_head = pickle.load(file)
#     file.close()
#============================================


flow_head = bem.FlowStokes(mesh, points)

#======save flowstokes object=========
# with open("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/test/flowstokes_object.pkl", "wb") as f:
#     pickle.dump(flow_head, f)
#=====================================

for frame in range(N_frames):

    psi, f1, f2 =solution.psi[frame], solution.f1[frame], solution.f2[frame]
    flag = chlamy.flagellum_1[frame]
    flag2 = chlamy.flagellum_2[frame]

    u_field = chlamy.calc_vector_field(flow_head, frame, find_flow)


    ux = u_field[:,0].reshape(Ny,Nx)
    uy = u_field[:,1].reshape(Ny,Nx)
    uz = u_field[:,2].reshape(Ny,Nx)

    insidemask_2d = chlamy.inside_mask.reshape(Ny,Nx)

    ux_masked = np.copy(ux)
    uy_masked = np.copy(uy)
    uz_masked = np.copy(uz)

    ux_masked[insidemask_2d] = np.nan
    uy_masked[insidemask_2d] = np.nan
    uz_masked[insidemask_2d] = np.nan


    U_magnitude = np.sqrt(ux**2 + uy**2 +uz**2)

    quiver_density=10

    Ux_quiver = ux_masked[::quiver_density, ::quiver_density]
    Uy_quiver = uy_masked[::quiver_density, ::quiver_density]

    x_quiver = x[::quiver_density]
    y_quiver = y[::quiver_density]

  
    vmax = 100* flag.flagellum_length

    r, t = flag.r, flag.tangents

    r= r[1:]
    t=t[1:]

    r2, t2 = flag2.r, flag2.tangents


    r2= r2[1:]
    t2=t2[1:]


    max_len = 10
    scale = np.minimum(1, max_len / np.max(U_magnitude))
    # scale[U_magnitude == 0] = 0

    norm = colors.Normalize(vmin=0, vmax=vmax)
    fig = plt.figure(figsize=(12, 10))
    c = plt.pcolormesh(x, y, U_magnitude, shading='auto', cmap='viridis',norm=norm)
    plt.title(f"$u_x = 10^3$ $\\mu \\text{{m/s }}$, $t$={round(frame*dt*10**3,2)} ms")        

    plt.plot(r[:,0],r[:,1],color='red')
    plt.plot(r2[:,0],r2[:,1],color='g')
    plt.plot(mesh.isosurface[:,0],mesh.isosurface[:,1],color='r')
   

    plt.quiver(x_quiver, y_quiver, Ux_quiver, Uy_quiver,
                color='white', headlength=4, headwidth=2,scale_units='xy',scale=500)
    plt.colorbar(c,label=r'$|\mathbf{U}_{\text{field}}|$ [$\mu$m/s]')
    plt.xlabel(f'$x$ [$\\mu$m]')
    plt.ylabel(f'$y$ [$\\mu$m]')
    # plt.title('Flow magnitude and direction')
    plt.axis('equal')
    # plt.show()
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





