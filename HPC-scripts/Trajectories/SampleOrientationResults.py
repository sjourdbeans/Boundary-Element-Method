import h5py
from pathlib import Path
import pickle
import numpy as np

shear_rate = 5
elements   = 320
periods    = 140
N_swimmers = 4500

ratio      = 5.8

fileswimmer = f"/scratch/sbuitjes/swimmer_objects/chlamy/chlamy-3d/chlamy_free_{elements}.pkl"
base_folder = Path(f"/scratch/sbuitjes/simulation_results/trajectory-sims/chlamy/chlamy-3d")

folder = base_folder / f"mesh={elements}_shear={shear_rate}_N={N_swimmers}_periods_{periods}"

with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)


frames_per_beat = swimmer_template.N_frames

output_folder = base_folder / "sampled"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / f"mesh={elements}_shear={shear_rate}_N={N_swimmers}_periods_{periods}.h5"

discard_beats =0
# ===================== Load swimmer template =====================

frames          = frames_per_beat * periods + 1

# ===================== First pass: discover all datasets and their shapes =====================
# Open one file and one simulation to see what datasets exist
dataset_info = {}  # name -> (shape_per_sim, dtype)

for rank_file in sorted(folder.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:
        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp = f[sim_name]
            for ds_name in grp:
                if ds_name not in dataset_info:
                    ds = grp[ds_name]
                    dataset_info[ds_name] = (ds.shape, ds.dtype)
            break  # one sim is enough to discover all datasets
    break          # one file is enough

print("Datasets found per simulation:")
for name, (shape, dtype) in dataset_info.items():
    print(f"  {name}: shape={shape}, dtype={dtype}")

# ===================== Categorise datasets by time axis =====================
# Datasets with first dimension == frames are time-resolved -> subsample
# Others (e.g. initial_orientation) are stored as-is
time_resolved  = {}  # name -> full array (N_conditions, frames, ...)
static_datasets = {}  # name -> array (N_conditions, ...)

for name, (shape, dtype) in dataset_info.items():
    if len(shape) >= 1 and shape[0] == frames:
        # time-resolved: allocate (N_conditions, frames, ...)
        time_resolved[name] = np.zeros((N_swimmers,) + shape, dtype=dtype)
    else:
        static_datasets[name] = np.zeros((N_swimmers,) + shape, dtype=dtype)

# ===================== Second pass: fill arrays =====================
for rank_file in sorted(folder.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:
        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp     = f[sim_name]
            sim_idx = int(grp.attrs["sim_index"])

            for name in time_resolved:
                time_resolved[name][sim_idx] = grp[name][:]

            for name in static_datasets:
                static_datasets[name][sim_idx] = grp[name][:]

# ===================== Subsample time-resolved arrays =====================
beat_indices   = np.arange(discard_beats, periods)
strobo_indices = beat_indices * frames_per_beat

print(f"\nSubsampling {frames} frames -> {len(beat_indices)} stroboscopic frames")

# ===================== Save =====================
with h5py.File(output_file, "w") as f_out:
    # Metadata
    f_out.attrs["frames_per_beat"] = frames_per_beat
    f_out.attrs["periods"]         = periods
    f_out.attrs["discard_beats"]   = discard_beats
    f_out.attrs["N_conditions"]    = N_swimmers

    f_out.create_dataset("beat_indices", data=beat_indices, compression="gzip")

    # Time-resolved datasets: subsample to stroboscopic
    grp_strobo = f_out.create_group("stroboscopic")
    for name, arr in time_resolved.items():
        subsampled = arr[:, strobo_indices]  # (N_conditions, N_beats, ...)
        grp_strobo.create_dataset(name, data=subsampled, compression="gzip")
        print(f"  stroboscopic/{name}: {arr.shape} -> {subsampled.shape}")

    # Static datasets: store as-is
    grp_static = f_out.create_group("static")
    for name, arr in static_datasets.items():
        grp_static.create_dataset(name, data=arr, compression="gzip")
        print(f"  static/{name}: {arr.shape}")

print(f"\nSaved to {output_file}")