import bemsolver as bem
import numpy as np
import pickle





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




flagellum_1 = []
flagellum_2 = []

N_frames = len(R)

mesh = bem.Mesh(chlamy_path)

# loop over all frames to create flagellum objects
for frame in range(N_frames):
    frame = frame % N_frames

    R[frame]=(rotmat @ (R[frame]-R[frame][0]).T ).T +np.array([6,2,0])

    R2[frame]=(rotmat @ (R2[frame]-R2[frame][0]).T  ).T+np.array([6,-2,0])

    v= (rotmat @ V[frame].T ).T
    v2=(rotmat @ V2[frame].T).T

    flag1 = bem.SlenderCoordinates(R[frame],v, flagellum_length=lf,flagellum_radius=0.2)
    
    flag2 = bem.SlenderCoordinates(R2[frame],v2, flagellum_length=lf,flagellum_radius=0.2)

    flagellum_1.append(flag1)
    flagellum_2.append(flag2)

# ===================Create swimmer object=====================


chlamy = bem.FreeSwimmer(mesh,
                        flagellum_1=flagellum_1,flagellum_2=flagellum_2)

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