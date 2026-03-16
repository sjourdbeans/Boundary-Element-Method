import h5py
from pathlib import Path
import pickle
import numpy as np

# fileswimmer = "/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_1280.pkl"
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

# file = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/shear=0.0_N=8_periods_10/rank_000.h5"
folder = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/mesh=320_shear=0.0_N=125_periods_300"
outdir =  Path(folder)

manifest = np.loadtxt(folder + "/manifest.txt", dtype=str, delimiter="\t")
N_conditions = len(manifest)

frames_per_beat = swimmer_template.N_frames

periods = int(folder.split("_")[-1])

frames = frames_per_beat*periods + 1
quaternions = np.zeros((N_conditions, frames, 4), dtype=np.float32)


for rank_file in sorted(outdir.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:
        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp = f[sim_name]
            sim_idx = int(grp.attrs["sim_index"])
            Q = grp["quaternions"][:]
            quaternions[sim_idx, :, :] = Q
            # process here

