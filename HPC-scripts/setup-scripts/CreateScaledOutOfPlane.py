import bemsolver as bem
import numpy as np
import pickle



elements = 320

chlamy_path       =f"/scratch/sbuitjes/meshes/chlamy/chlamy_N={elements}.mat"

movie_data="/home/sbuitjes/code/Boundary-Element-Method/datafiles/waveform/Chlamy-3D/CC125_4_2_T_avg_int.txt"
movie_data_2="/home/sbuitjes/code/Boundary-Element-Method/datafiles/waveform/Chlamy-3D/CC125_4_2_C_avg_int.txt"
coords =np.loadtxt(movie_data)
coords_2 =np.loadtxt(movie_data_2)


theta=np.pi/2
rotmat=np.array([[np.cos(theta), -np.sin(theta),0],
                 [np.sin(theta), np.cos(theta),0],
                 [0,0,1]])
lf=12

R = coords.reshape(int(len(coords)/20),20,3)
R2 = coords_2.reshape(int(len(coords_2)/20),20,3)
# frame =5

# R2 = R.copy()
# R2[:,:,0] *= -1
# R2[:,:,2] *= -1

# R = R[::2,:,:]
# R2 =R2[::2,:,:]

T1 = R[:, 1:,:] - R[:,:-1, :]
T1 = T1/ np.linalg.norm(T1, axis=2, keepdims=True)

T2 = R2[:, 1:,:] - R2[:,:-1, :]
T2 = T2/ np.linalg.norm(T2, axis=2, keepdims=True)

theta_1_map = np.arctan2(T1[:,:,1], T1[:,:,0])
phi_1_map = np.arctan2(T1[:,:,2], np.sqrt(T1[:,:,0]**2 + T1[:,:,1]**2))

theta_2_map = np.arctan2(T2[:,:,1], T2[:,:,0])
phi_2_map = np.arctan2(T2[:,:,2], np.sqrt(T2[:,:,0]**2 + T2[:,:,1]**2))


dt=400*10**(-6)

factor = 1
Nt= len(R)


r1_map = np.zeros((factor*Nt, len(R[0])-1, 3))
r2_map = np.zeros((factor*Nt, len(R2[0])-1, 3))

scale_out_of_plane = 0.8
savepath = f"/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/non-symmetric/scale-out-of-plane/chlamy_free_{elements}_scale={scale_out_of_plane}.pkl"


for time_frame in range(Nt):

    frame = time_frame % Nt


    curve_1 = bem.SlenderAngles(theta_1_map[time_frame,:], scale_out_of_plane*phi_1_map[time_frame,:], flagellum_length=lf,flagellum_radius=0.2)
    curve_2 = bem.SlenderAngles(theta_2_map[time_frame,:], scale_out_of_plane*phi_2_map[time_frame,:], flagellum_length=lf,flagellum_radius=0.2)

    r1_map[time_frame,:,:] = curve_1.r
    r2_map[time_frame,:,:] = curve_2.r


V = (np.roll(r1_map, -1, axis=0) - 
     np.roll(r1_map,  1, axis=0)) / (2*dt)

V2 = (np.roll(r2_map, -1, axis=0) - 
     np.roll(r2_map,  1, axis=0)) / (2*dt)



flagellum_1 = []
flagellum_2 = []

N_frames = len(R)

mesh = bem.Mesh(chlamy_path)

# loop over all frames to create flagellum objects
for frame in range(N_frames):
    frame = frame % N_frames

    r1_map[frame]=(rotmat @ (r1_map[frame]-r1_map[frame][0]).T ).T +np.array([6,2,0])


    r2_map[frame]=(rotmat @ (r2_map[frame]-r2_map[frame][0]).T  ).T+np.array([6,-2,0])

    v= (rotmat @ V[frame].T ).T
    v2=(rotmat @ V2[frame].T).T

    flag1 = bem.SlenderCoordinates(r1_map[frame],v, flagellum_length=lf,flagellum_radius=0.2)
    
    flag2 = bem.SlenderCoordinates(r2_map[frame],v2, flagellum_length=lf,flagellum_radius=0.2)

    flagellum_1.append(flag1)
    flagellum_2.append(flag2)

# ===================Create swimmer object=====================


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
