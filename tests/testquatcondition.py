
import os
from pathlib import Path
import pickle

import h5py
import numpy as np
from mpi4py import MPI
import copy
import bemsolver as bem

def random_quaternion():
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()

    # Use a standard method to generate a random quaternion
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    return q

def quat_to_director(q):
    """First column of rotation matrix = swimmer symmetry axis in lab frame."""
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])



def sample_quarter_sphere():
    """Sample orientation vector restricted to quarter sphere."""
    q = random_quaternion()
    e = quat_to_director(q)
    
    # Reflect into quarter sphere using symmetries
    if (e[0]>0) & (e[2]>0):
        return q
    else:
        return None
    

# Create N random quaternions
N_conditions = 500

initial_conditions=[]
size =0
while size <N_conditions:

    quat =sample_quarter_sphere()
    if quat is not None:    
        initial_conditions.append(quat)
        size+=1
    
initial_conditions = np.array(initial_conditions)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R





# Shape: (n_ics, n_beats, 4)
# strobo_quats    = quaternions[::plot_steps, strobo_idx, :]

# Convert to director vectors — shape: (n_ics, n_beats, 3)
directors = np.array([quat_to_director(q) for q in initial_conditions])

n_ics, n_beats= directors.shape


# ── Compute spherical coords ───────────────────────────────────────────────────
theta = np.degrees(np.arctan2(directors[:, 1], directors[:, 0]))  # azimuth
phi   = np.degrees(np.arcsin(np.clip(directors[:, 2], -1, 1)))        # elevation

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 6))

# Colour each IC distinctly
colors = plt.cm.hsv(np.linspace(0, 1, n_ics))

# ── Panel 1: Azimuth vs elevation ─────────────────────────────────────────────
# ── Panel 2: 3D sphere ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(111, projection='3d')

# Reference sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
ax2.plot_wireframe(
    np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
    color='gray', alpha=0.4, lw=0.4
)

for i, color in enumerate(colors):
    x, y, z = directors[i,  0], directors[i,  1], directors[i, 2]
    # ax2.plot(x[1:], y[1:], z[1:],   color=color, alpha=0.4)
    ax2.scatter(x,  y,  z,  s=20, color='blue', marker='.', zorder=5)

ax2.set_title('Initial Orientations (440 out of 4400)')
ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$'); ax2.set_zlabel('$z$')
ax2.view_init(elev=30, azim=30)
plt.tight_layout()

# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/Initial-orientations-unit-sphere.pdf")
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/Initial-orientations-unit-sphere.png",dpi=600)
plt.show()