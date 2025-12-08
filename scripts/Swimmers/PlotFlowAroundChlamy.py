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


#===============Temporary before integrated into code==================
def U_colloc(U          :np.ndarray,
             W          :np.ndarray,
             centroids  :np.ndarray,
             r          :int,
             E          :np.ndarray=np.zeros((3,3)))->tuple[np.ndarray,np.ndarray]:
    """
    Calculate translational and rotational velocity at each collocation point.

    Parameters
    ----------
    U : np.ndarray, shape (3,)
        Translational velocity of external flow [micron/s].
    W : np.ndarray, shape (3,)
        Rotational velocity of external flow [rad/s].
    centroid : np.ndarray, shape (r, 3)
        XYZ coordinates of centroids of cell mesh [micron].
    r : int
        Number of rows / collocation points.
    E : np.ndarray, shape (3,3)
        Rate of strain tensor.

    Returns
    -------
    U_t : np.ndarray, shape (3*r,)
        Translational velocity vector [micron/s].
    U_r : np.ndarray, shape (3*r,)
        Rotational velocity vector [micron/s].
    U_e : np.ndarray, shape (3*r,)
        Strain rate velocity vector [micron/s].
    """
     # Translational velocity: just repeat U for each collocation point
    U_t = np.tile(U, r)

    # Rotational velocity: cross product W x centroid
    U_r = np.zeros(3*r)
    U_r[0::3] =  W[1]*centroids[:,2] - W[2]*centroids[:,1]
    U_r[1::3] = -W[0]*centroids[:,2] + W[2]*centroids[:,0]
    U_r[2::3] =  W[0]*centroids[:,1] - W[1]*centroids[:,0]

    # NEED TO ADJUST FOR GENERAL CASE

    
    U_e = (E @ centroids.T).T

    U_e = U_e.flatten() 

    
    return U_t, U_r, U_e

#=================================================================

waveformfile      = loadmat("/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/waveform/lib02_1_90_2019-06-28_1640.mat")
chlamy_path ="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/Chlamy/chlamy_N=320.mat"

mesh = bem.Mesh(chlamy_path)


frame = 20


cell = waveformfile["Cell"]
thetar = cell["thetar"].item()[0][0]
thetal = cell["thetal"].item()[0][0]
phi_body = cell["phi_body"].item()[0][0]


xbase = - cell["dist_base"].item()[0][0]
ybase = 0
zbase = 0



lf = waveformfile["lf0"][0,0]
fps = waveformfile["fps"]
dt = 1 /fps


gamma_dot=0

U = np.zeros(3)

U[0] = 1000 #gamma_dot * x[1]
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

x = np.linspace(-20, 20, Nx)
y = np.linspace(-20, 20, Ny)

xg, yg =np.meshgrid(x,y)
zg=np.zeros_like(xg)

xg = xg.ravel()
yg = yg.ravel()
zg = zg.ravel()

Ng = np.shape(xg)[0]

points = np.vstack((xg, yg, zg)).T

savepath = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/videos/Chlamy/Chlamy_test.mp4"
tmp_dir = "frames"

os.makedirs(tmp_dir, exist_ok=True)
frame_files = []

#============frame dependent============
flow_head = bem.FlowStokes(mesh, points)

sys = bem.ResistanceProblem(mesh)
sys.construct_mobility_matrix()

# U_field =flow_head.calc_vector_field(psi,U, W, E)

for frame in range(len(waveformfile["kappasave"])):

    velx_1 = -waveformfile["velx0"][frame,0] * lf
    vely_1 = -waveformfile["vely0"][frame,0] * lf
    velz_1 = np.zeros_like(vely_1) * lf

    vel_1 = np.vstack([velx_1, vely_1, velz_1]).T


    velx_2 = -waveformfile["velx0"][frame,1] * lf
    vely_2 = waveformfile["vely0"][frame,1] * lf
    velz_2 = np.zeros_like(vely_1) * lf

    vel_2 = np.vstack([velx_2, vely_2, velz_2]).T


    base_position_1 = np.array([xbase , ybase, zbase]) 
    curv_1 = waveformfile["kappasave"][frame,0,1:] 
    theta_0_1 = waveformfile["kappasave"][frame,0,0]

    base_position_2 = -base_position_1 
    curv_2 = -waveformfile["kappasave"][frame,1,1:] 
    theta_0_2 = waveformfile["kappasave"][frame,1,0]

    initial_angle_1 = np.pi - (thetal - phi_body) + theta_0_1
    initial_angle_2 = np.pi - (thetar - phi_body) - theta_0_2


    tors_1 = np.zeros_like(curv_1)


    flag = bem.SlenderBody(curv_1,tors_1,theta_0=initial_angle_1,flagellum_length=lf,base_position=base_position_1)
    flag2 = bem.SlenderBody(curv_2,tors_1,theta_0=initial_angle_2,flagellum_length=lf,base_position=base_position_1)
    # flag2.r[:,1] = flag2.r[:,1]

    M = flag.construct_mobility_matrix()
    M2 = flag2.construct_mobility_matrix()

    
    M12 = flag.calc_interaction(flag2.r)
    M1h = flag.calc_interaction(mesh.centroids)


    M21 = flag2.calc_interaction(flag.r)
    M2h = flag.calc_interaction(mesh.centroids)
    
    Mh1 = bem.FlowStokes(mesh, flag.r)
    Mh2 = bem.FlowStokes(mesh, flag2.r)
    
    
    M_total = np.block([
        [sys.MATRIX, M1h, M2h],
        [Mh1.MATRIX, M, M21],
        [Mh2.MATRIX, M12, M2]
    ])


    vel_1_flat = vel_1[flag.indstart+1:].flatten()
    vel_2_flat = vel_2[flag2.indstart+1:].flatten()

    RHS_h = sys.set_boundary_condition(U,W,E)
    RHS1  = flag.set_boundary_condition(U,W,E) + vel_1_flat
    RHS2  = flag2.set_boundary_condition(U,W,E) + vel_2_flat

    RHS = np.concatenate([RHS_h, RHS1, RHS2])
    
    f = np.linalg.solve(M_total,-RHS)

    Nh = np.shape(sys.MATRIX)[1]
    Nf1 = np.shape(M)[1]
    Nf2 = np.shape(M2)[1]

    psi = f[:Nh]
    f1 = f[Nh:Nh+Nf1]
    f2 = f[Nh+Nf1:]



    K = flag.calc_interaction(points)
    K2 = flag2.calc_interaction(points)

    u_field  = K @ f1 + K2 @ f2 + flow_head.MATRIX  @ psi

    # use for mask 
    flow_head.calc_vector_field(psi,U, W, E)
    #===========
    
    u_boundary = flow_head.set_background_flow(U, W, E)

    u_field = u_field + u_boundary


    u_field = u_field.reshape(Ng, 3)
    u_field[flow_head.inside_mask,:]=0

    ux = u_field[:,0].reshape(Ny,Nx)
    uy = u_field[:,1].reshape(Ny,Nx)
    uz = u_field[:,2].reshape(Ny,Nx)

    insidemask_2d = flow_head.inside_mask.reshape(Ny,Nx)

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

    Xq, Yq =  np.meshgrid(x_quiver, y_quiver)
    Zq = np.zeros_like(Xq)


    quiver_points = np.stack([Xq, Yq, Zq], axis=-1)

    d2_1 = np.min(
    np.sum((quiver_points[:, :, None, :] - r[None, None, :, :])**2, axis=-1),
    axis=2)

    d2_2 = np.min(
    np.sum((quiver_points[:, :, None, :] - r2[None, None, :, :])**2, axis=-1),
    axis=2)

    # print(d2_2)

    d2 = np.minimum(d2_1, d2_2)
    # print(d2)

    delta = 4 #* flag.flagellum_radius
    mask = d2 < delta**2  

    Ux_quiver_masked = Ux_quiver.copy()
    Uy_quiver_masked = Uy_quiver.copy()

    # Ux_quiver_masked[mask] = np.nan
    # Uy_quiver_masked[mask] = np.nan



    max_len = 10
    scale = np.minimum(1, max_len / np.max(U_magnitude))
    # scale[U_magnitude == 0] = 0

    norm = colors.Normalize(vmin=0, vmax=vmax)
    fig = plt.figure(figsize=(12, 10))
    c = plt.pcolormesh(x, y, U_magnitude, shading='auto', cmap='viridis',norm=norm)
    plt.title(f"$u_x = 10^3$ $\\mu \\text{{m/s }}$, $t$={round(frame*dt[0][0]*10**3,2)} ms")        

    plt.plot(r[:,0],r[:,1],color='red')
    plt.plot(r2[:,0],r2[:,1],color='g')
    plt.plot(mesh.isosurface[:,0],mesh.isosurface[:,1],color='r')
   

    plt.quiver(x_quiver, y_quiver, Ux_quiver_masked, Uy_quiver_masked,
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


