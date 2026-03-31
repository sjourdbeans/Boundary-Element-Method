from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path
import pickle
import h5py
import numpy as np
from mpi4py import MPI
import copy
import bemsolver as bem


# ============ Simulation Setup =============
elements = 320

shear_rate = 15
dt = 400e-6
n_periods = 1
save_dtype = np.float32

p_arr = np.linspace(-np.pi/2, np.pi/2, 90)
y_arr = np.linspace(-np.pi , np.pi , 90)
r_arr = np.linspace(-np.pi , np.pi, 90)

p_mesh, y_mesh, r_mesh= np.meshgrid(p_arr, y_arr,r_arr, indexing="ij")
initial_angles = np.column_stack([
    p_mesh.ravel(),
    y_mesh.ravel(),
    r_mesh.ravel(),
])

n_sims = len(initial_angles)

fileswimmer = f"/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_{elements}.pkl"

basefolder=f"/scratch/sbuitjes/simulation_results/orientation_map/chlamy/chlamy-3d/non-symmetric"
outdir = Path(f"{basefolder}/mesh={elements}_shear={round(shear_rate,1)}_N={n_sims}_periods_{n_periods}")

outdir.mkdir(parents=True, exist_ok=True)


with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

def find_flow(t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma_dot = shear_rate
    U = np.zeros(3)
    W = np.zeros(3)
    E = gamma_dot / 2 * np.array(
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 0]],
        dtype=float,
    )
    W[2] = -gamma_dot
    return U, W, E


initial_conditions = np.zeros((len(initial_angles), 4))

for i,initial_orientation in enumerate(initial_angles):
    pitch, yaw, roll = initial_orientation 

    r = R.from_euler('xzy', [roll, yaw, pitch])

    q = r.as_quat(scalar_first=True)
    initial_conditions[i] = q



# =======================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

split_indices = np.array_split(np.arange(n_sims), size)
local_indices = split_indices[rank]

rank_assignments = {}
for r, idxs in enumerate(split_indices):
    for idx in idxs:
        rank_assignments[int(idx)] = r

if rank == 0:
    print(f"Total simulations: {n_sims}")
    print(f"MPI ranks: {size}")
    print(f"Output directory: {outdir}")

print(f"Rank {rank}: handling simulation indices {local_indices}")


frames = n_periods * swimmer_template.N_frames + 1
t_end = frames * dt

rank_file = outdir / f"rank_{rank:03d}.h5"

with h5py.File(rank_file, "w") as h5:
    h5.attrs["rank"] = rank
    h5.attrs["size"] = size
    h5.attrs["dt"] = dt
    h5.attrs["n_periods"] = n_periods
    h5.attrs["shear_rate"] = shear_rate
    h5.attrs["source_swimmer_file"] = fileswimmer

    h5.create_dataset("assigned_sim_indices", data=np.asarray(local_indices, dtype=np.int32))

    for sim_idx in local_indices:
        cond = initial_conditions[sim_idx]
        print(f"Rank {rank}: starting sim {sim_idx} with IC {cond}")


        swimmer = copy.deepcopy(swimmer_template)

        sol = swimmer.RBM_over_time(
            dt,
            t_end,
            find_flow,
            initial_quaternion=cond,
        )

        grp = h5.create_group(f"sim_{sim_idx:05d}")
        grp.attrs["sim_index"] = int(sim_idx)

        grp.attrs["complete"] = False

        grp.create_dataset("X", data=np.asarray(sol.X, dtype=save_dtype), compression="gzip")
        grp.create_dataset("quaternions", data=np.asarray(sol.quaternions, dtype=save_dtype), compression="gzip")
        grp.create_dataset("u", data=np.asarray(sol.u, dtype=save_dtype), compression="gzip")
        grp.create_dataset("omega", data=np.asarray(sol.omega, dtype=save_dtype), compression="gzip")
        grp.create_dataset("rotation", data=np.asarray(sol.rotation_matrices, dtype=save_dtype),compression="gzip")


        grp.attrs["complete"] = True
        h5.flush()

        print(f"Rank {rank}: finished sim {sim_idx}")

comm.Barrier()

if rank == 0:
    with open(outdir / "manifest.txt", "w") as f:
        for sim_idx, cond in enumerate(initial_conditions):
            f.write(
                f"sim_{sim_idx:05d}  "
                f"rank={rank_assignments[sim_idx]}  "
                f"p={cond[0]: .8f}  "
                f"y={cond[1]: .8f}  "
                f"r={cond[2]: .8f}\n"
            )
    print("All simulations finished.")


    Omega = np.zeros((n_sims, 3))

    for rank_file in sorted(outdir.glob("rank_*.h5")):
        with h5py.File(rank_file, "r") as f:

            for sim_name in f:
                if sim_name == "assigned_sim_indices":
                    continue
                grp = f[sim_name]
                sim_idx = int(grp.attrs["sim_index"])
                Q = grp["rotation"][:]
                w = grp["omega"][:]

                proj_omega = np.array([Q[i].T @ w[i] for i in range(len(w))])
                mean_omega = np.mean(proj_omega, axis=0)
                Omega[sim_idx, :] = mean_omega




    np.savez_compressed(outdir / "angles_pyr_omega_rpy.npz", A=initial_angles, B=Omega)

    from scipy.interpolate import RBFInterpolator

    f = RBFInterpolator(initial_conditions, Omega)

    with open(outdir / "rbf_interpolation.pkl", "wb") as file:
        pickle.dump(f, file)