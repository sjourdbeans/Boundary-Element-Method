import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib as mpl


shear_rate = 4
elements =320

fileswimmer = f"/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_{elements}.pkl"

with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

folder = f"/scratch/sbuitjes/simulation_results/trajectory-sims/chlamy/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
outdir =  Path(folder)


manifest = np.loadtxt(folder + "/manifest.txt", dtype=str, delimiter="\t")
N_conditions = len(manifest)

frames_per_beat = swimmer_template.N_frames

periods = int(folder.split("_")[-1])

frames = frames_per_beat*periods + 1
initial_conditions = np.zeros((N_conditions, 4), dtype=np.float32) 
quaternions = np.zeros((N_conditions, frames, 4), dtype=np.float32)

savepath = f"/scratch/sbuitjes/simulation_results/trajectory-sims/chlamy/chlamy-3d/vary_quats/distributions/mesh={elements}/shear={shear_rate}_N={N_conditions}_periods_{periods}.npz"

for rank_file in sorted(outdir.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:

        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp = f[sim_name]
            sim_idx = int(grp.attrs["sim_index"])
            initial_conditions[sim_idx,:] = grp["initial_orientation"][:]
            Q = grp["quaternions"][:]
            quaternions[sim_idx, :, :] = Q
            # process here



# =========== Exclude duplicates ============

initial_q = quaternions[:, 0, :]

def canonicalize_quaternion_sign(Q):
    Q = Q.copy()
    for i in range(Q.shape[0]):
        q = Q[i]
        for val in q:
            if abs(val) > 1e-12:
                if val < 0:
                    Q[i] = -q
                break
    return Q

initial_q_canon = canonicalize_quaternion_sign(initial_q)

unique_q, unique_idx = np.unique(
    np.round(initial_q_canon, 6),
    axis=0,
    return_index=True
)

unique_idx = np.sort(unique_idx)

quaternions = quaternions[unique_idx, :, :]


# =================================
import matplotlib
import matplotlib.pyplot as plt

def quat_to_director(q):
    """First column of rotation matrix = swimmer symmetry axis in lab frame."""
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])

step=0

plot_steps = 1

new_frames = (step+1) *frames_per_beat

discard_beats = step
strobo_idx = np.arange(discard_beats * frames_per_beat, frames, frames_per_beat)

# strobo_idx = np.arange(0, frames, frames_per_beat)
strobo_quats = quaternions[::plot_steps, strobo_idx, :]

# Convert quaternions to directors
directors = np.array([[quat_to_director(q) for q in ic] for ic in strobo_quats])

# Flatten all sampled orientations
dirs_flat = directors.reshape(-1, 3)


# azimuth in units of pi: [-1, 1]
azimuth_pi = np.arctan2(dirs_flat[:, 1], dirs_flat[:, 0]) / np.pi

# polar angle theta in units of pi: [0, 1]
theta = np.arccos(np.clip(-dirs_flat[:, 2], -1, 1))
theta_pi = theta / np.pi

# histogram with raw counts
H, th_edges, az_edges = np.histogram2d(
    theta_pi,
    azimuth_pi,
    bins=[120, 240],
    range=[[0, 1], [-1, 1]]
)

# Save distribution
np.savez_compressed(
    savepath,
    H=H,
    th_edges=th_edges,
    az_edges=az_edges
)