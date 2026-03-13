import bemsolver as bem
import numpy as np
import pickle
from mpi4py import MPI




chlamy_path       = "/scratch/sbuitjes/meshes/chlamy/chlamy_N=1280.mat"

movie_data="/home/sbuitjes/code/Boundary-Element-Method/datafiles/waveform/Chlamy-3D/CC125_4_2_T_avg_int.txt"
movie_data_2="/home/sbuitjes/code/Boundary-Element-Method/datafiles/waveform/Chlamy-3D/CC125_4_2_C_avg_int.txt"
coords =np.loadtxt(movie_data)
coords_2 =np.loadtxt(movie_data_2)


savepath = "/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_1280.pkl"

theta=np.pi/2
rotmat=np.array([[np.cos(theta), -np.sin(theta),0],
                 [np.sin(theta), np.cos(theta),0],
                 [0,0,1]])
lf=12

R = coords.reshape(int(len(coords)/20),20,3)/7 * lf
R2 = coords_2.reshape(int(len(coords_2)/20),20,3)/ 7 *lf
# frame =5

# R2 = R.copy()
# R2[:,:,0] *= -1
# R2[:,:,2] *= -1

# R = R[::2,:,:]
# R2 =R2[::2,:,:]

dt=400*10**(-6)
# dt=0.00004
V = (np.roll(R, -1, axis=0) - 
     np.roll(R,  1, axis=0)) / (2*dt)

V2 = (np.roll(R2, -1, axis=0) - 
     np.roll(R2,  1, axis=0)) / (2*dt)





N_frames = len(R)

mesh = bem.Mesh(chlamy_path)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_frames = np.array_split(np.arange(N_frames), size)[rank]

local_data = []

for frame in local_frames:
    Rf = (rotmat @ (R[frame] - R[frame][0]).T).T + np.array([6, 2, 0])
    R2f = (rotmat @ (R2[frame] - R2[frame][0]).T).T + np.array([6, -2, 0])

    vf = (rotmat @ V[frame].T).T
    v2f = (rotmat @ V2[frame].T).T

    local_data.append((frame, Rf, vf, R2f, v2f))

all_data = comm.gather(local_data, root=0)

if rank == 0:
    flagellum_1 = [None] * N_frames
    flagellum_2 = [None] * N_frames

    for chunk in all_data:
        for frame, Rf, vf, R2f, v2f in chunk:
            flagellum_1[frame] = bem.SlenderCoordinates(
                Rf, vf, flagellum_length=lf, flagellum_radius=0.2
            )
            flagellum_2[frame] = bem.SlenderCoordinates(
                R2f, v2f, flagellum_length=lf, flagellum_radius=0.2
            )

    chlamy = bem.FreeSwimmer(mesh,
                            flagellum_1=flagellum_1,flagellum_2=flagellum_2, viscosity=1e-3)

# chlamy = bem.Swimmer(mesh,
#                     flagellum_1=flagellum_1,flagellum_2=flagellum_2)
# =============================================================


# ===================Save option 1=====================

# Save swimmer object without results (large file)
    with open(savepath, "wb") as f:
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