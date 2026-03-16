
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

shear_rate = 0.0
dt = 400e-6
n_periods = 300
save_dtype = np.float32


p_arr = np.linspace(-np.pi / 3, np.pi / 3, 2)
y_arr = np.linspace(-np.pi / 4, np.pi / 4, 2)
r_arr = np.linspace(-np.pi / 4, np.pi / 4, 2)

p_mesh, y_mesh, r_mesh = np.meshgrid(p_arr, y_arr, r_arr, indexing="ij")
initial_conditions = np.column_stack([
    p_mesh.ravel(),
    y_mesh.ravel(),
    r_mesh.ravel(),
])
n_sims = len(initial_conditions)


fileswimmer = f"/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_{elements}.pkl"
outdir = Path(f"/scratch/sbuitjes/simulation_results/trajectory-sims/chlamy/chlamy-3d/mesh={elements}_shear={round(shear_rate,1)}_N={n_sims}_periods_{n_periods}")

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
            initial_orientation=cond,
        )

        grp = h5.create_group(f"sim_{sim_idx:05d}")
        grp.attrs["sim_index"] = int(sim_idx)

        grp.attrs["complete"] = False

        grp.create_dataset("initial_orientation", data=np.asarray(cond, dtype=save_dtype))
        grp.create_dataset("time", data=np.asarray(sol.time, dtype=save_dtype))
        grp.create_dataset("X", data=np.asarray(sol.X, dtype=save_dtype), compression="gzip")
        grp.create_dataset("quaternions", data=np.asarray(sol.quaternions, dtype=save_dtype), compression="gzip")
        grp.create_dataset("u", data=np.asarray(sol.u, dtype=save_dtype), compression="gzip")
        grp.create_dataset("omega", data=np.asarray(sol.omega, dtype=save_dtype), compression="gzip")


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