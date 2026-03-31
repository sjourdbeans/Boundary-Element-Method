import h5py
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"]= "Palatino"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"
mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13

shear_rate=7

# ── Load data ──────────────────────────────────────────────────────────────────
fileswimmer = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"
with open(fileswimmer, "rb") as f:
    swimmer_template = pickle.load(f)

folder =f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/quarter_sphere/mesh=320_shear=15_N=1100_periods_140"
# folder = f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/chlamy-3d/vary_quats/mesh=320_shear={shear_rate}_N=4500_periods_140"
outdir = Path(folder)

manifest = np.loadtxt(folder + "/manifest.txt", dtype=str, delimiter="\t")
N_conditions = len(manifest)

frames_per_beat = swimmer_template.N_frames
periods = int(folder.split("_")[-1])

frames = frames_per_beat * periods + 1
quaternions = np.zeros((N_conditions, frames, 4), dtype=np.float32)

for rank_file in sorted(outdir.glob("rank_*.h5")):
    with h5py.File(rank_file, "r") as f:
        for sim_name in f:
            if sim_name == "assigned_sim_indices":
                continue
            grp = f[sim_name]
            sim_idx = int(grp.attrs["sim_index"])
            quaternions[sim_idx] = grp["quaternions"][:]

# ── Helpers ────────────────────────────────────────────────────────────────────
def quat_to_director(q):
    """First column of rotation matrix = swimmer symmetry axis in lab frame."""
    w, x, y, z = q
    return np.array([
        1 - 2*(y**2 + z**2),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ])

def wrapped_angle_diff_deg(a_next, a_now):
    """Minimal signed angular difference in degrees, in [-180, 180)."""
    return (a_next - a_now + 180.0) % 360.0 - 180.0

# ── Stroboscopic samples ───────────────────────────────────────────────────────
plot_steps = 1
discard_beats = 0
strobo_idx = np.arange(discard_beats * frames_per_beat, frames, frames_per_beat)

strobo_quats = quaternions[::plot_steps, strobo_idx, :]
directors = np.array([[quat_to_director(q) for q in ic] for ic in strobo_quats])

n_ics, n_beats, _ = directors.shape

theta = np.degrees(np.arctan2(directors[:, :, 1], directors[:, :, 0]))   # [-180, 180]
phi   = np.degrees(np.arcsin(np.clip(directors[:, :, 2], -1, 1)))        # [-90, 90]

# Base points and step vectors
theta0 = theta[:, :-1]
phi0   = phi[:, :-1]
dtheta = wrapped_angle_diff_deg(theta[:, 1:], theta[:, :-1])
dphi   = phi[:, 1:] - phi[:, :-1]

# Flatten all trajectories
theta0_flat = theta0.ravel()
phi0_flat   = phi0.ravel()
dtheta_flat = dtheta.ravel()
dphi_flat   = dphi.ravel()

# ── Bin the domain and average vectors ────────────────────────────────────────
n_theta_bins = 36   # e.g. 10 degree bins
n_phi_bins   = 18   # e.g. 10 degree bins

theta_edges = np.linspace(-180, 180, n_theta_bins + 1)
phi_edges   = np.linspace(-90, 90, n_phi_bins + 1)

theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])

# Storage
avg_dtheta = np.full((n_phi_bins, n_theta_bins), np.nan)
avg_dphi   = np.full((n_phi_bins, n_theta_bins), np.nan)
counts     = np.zeros((n_phi_bins, n_theta_bins), dtype=int)

# Assign each point to a bin
theta_idx = np.digitize(theta0_flat, theta_edges) - 1
phi_idx   = np.digitize(phi0_flat, phi_edges) - 1

# Keep only valid bins
valid = (
    (theta_idx >= 0) & (theta_idx < n_theta_bins) &
    (phi_idx >= 0) & (phi_idx < n_phi_bins)
)

theta_idx = theta_idx[valid]
phi_idx   = phi_idx[valid]
dtheta_v  = dtheta_flat[valid]
dphi_v    = dphi_flat[valid]

# Accumulate sums
sum_dtheta = np.zeros((n_phi_bins, n_theta_bins))
sum_dphi   = np.zeros((n_phi_bins, n_theta_bins))

for ti, pi, u, v in zip(theta_idx, phi_idx, dtheta_v, dphi_v):
    sum_dtheta[pi, ti] += u
    sum_dphi[pi, ti]   += v
    counts[pi, ti]     += 1

# Compute averages where count > 0
mask = counts > 0
avg_dtheta[mask] = sum_dtheta[mask] / counts[mask]
avg_dphi[mask]   = sum_dphi[mask] / counts[mask]

# Grid for quiver
TH, PH = np.meshgrid(theta_centers, phi_centers)

# Optional: suppress bins with very little data
min_count = 3
plot_mask = counts >= min_count

# ── Plot ───────────────────────────────────────────────────────────────────────
# ── Plot ───────────────────────────────────────────────────────────────────────


fig, ax = plt.subplots(figsize=(9, 6))

# Regular grid coordinates
x = theta_centers          # shape (n_theta_bins,)
y = phi_centers            # shape (n_phi_bins,)

# Mean vector field on grid
U = np.radians(avg_dtheta.copy())      # shape (n_phi_bins, n_theta_bins)
V = np.radians(avg_dphi.copy())

# Suppress poorly sampled bins
min_count = 3
valid = counts >= min_count

U[~valid] = np.nan
V[~valid] = np.nan

# Magnitude for coloring
speed = np.sqrt(U**2 + V**2)*50

# For streamplot, masked arrays usually work better than raw NaNs
U_masked = np.ma.masked_invalid(U)
V_masked = np.ma.masked_invalid(V)
speed_masked = np.ma.masked_invalid(speed)

strm = ax.streamplot(
    x/180, y/180,
    U_masked, V_masked,
    color=speed_masked,
    cmap="plasma",
    density=1.4,
    linewidth=1.2,
    arrowsize=1.2
)

# Optional: show bin centers that were actually used
# ax.scatter(TH[valid], PH[valid], s=8, color="k", alpha=0.25)

ax.set_xlabel("Azimuthal Angle $\\phi/\\pi$")
ax.set_ylabel("Polar Angle $\\theta/\\pi$")
ax.set_title(f"Orientational Dynamics of Chlamy with $\\dot{{\\gamma}}={shear_rate}$ s$^{{-1}}$")
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 0.5)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(strm.lines, ax=ax)
cbar.set_label(r"$\sqrt{\langle \Delta \theta \rangle^2 + \langle \Delta \phi \rangle^2}/T$ \, [rad/s]")

plt.tight_layout()

# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Orientations/chlamy/3D/Vector-Fields/2D/Dynamics_mesh=320_shear={shear_rate}_N={N_conditions}_periods_{periods}.pdf")
# plt.savefig(f"/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Orientations/chlamy/3D/Vector-Fields/2D/Dynamics_mesh=320_shear={shear_rate}_N={N_conditions}_periods_{periods}.png",dpi=600)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from matplotlib import cm, colors

# # -----------------------------------------------------------------------------
# # Assumes these already exist from your binning code:
# # theta_centers, phi_centers
# # avg_dtheta, avg_dphi
# # counts
# # TH, PH
# # -----------------------------------------------------------------------------

# # -----------------------------
# # 1. Build a smooth vector field on the regular theta-phi grid
# # -----------------------------
# x = theta_centers
# y = phi_centers
# X, Y = np.meshgrid(x, y)

# U = avg_dtheta.copy()   # dtheta per beat, in degrees
# V = avg_dphi.copy()     # dphi per beat, in degrees

# min_count = 3
# valid = counts >= min_count

# pts = np.column_stack([X[valid], Y[valid]])
# U_known = U[valid]
# V_known = V[valid]

# # Interpolate onto full grid
# U_lin = griddata(pts, U_known, (X, Y), method="linear")
# V_lin = griddata(pts, V_known, (X, Y), method="linear")

# # Fill remaining holes with nearest
# U_near = griddata(pts, U_known, (X, Y), method="nearest")
# V_near = griddata(pts, V_known, (X, Y), method="nearest")

# U_filled = np.where(np.isnan(U_lin), U_near, U_lin)
# V_filled = np.where(np.isnan(V_lin), V_near, V_lin)

# speed = np.sqrt(U_filled**2 + V_filled**2)

# # -----------------------------
# # 2. Compute streamlines in the 2D domain
# # -----------------------------
# fig = plt.figure(figsize=(16, 7))
# ax1 = fig.add_subplot(121)

# strm = ax1.streamplot(
#     x, y,
#     U_filled, V_filled,
#     color=speed,
#     cmap="viridis",
#     density=1.6,
#     linewidth=1.2,
#     arrowsize=1.0
# )

# ax1.scatter(X[valid], Y[valid], s=8, color="k", alpha=0.2)
# ax1.set_xlabel("Azimuth  [deg]")
# ax1.set_ylabel("Elevation  [deg]")
# ax1.set_title("Streamlines in orientation space")
# ax1.set_xlim(-180, 180)
# ax1.set_ylim(-90, 90)
# ax1.axhline(0, color='k', lw=0.5, ls='--')
# ax1.axvline(0, color='k', lw=0.5, ls='--')
# ax1.grid(True, alpha=0.3)

# cbar1 = plt.colorbar(strm.lines, ax=ax1)
# cbar1.set_label("Average vector magnitude")

# # -----------------------------
# # 3. Helper functions for mapping to sphere
# # -----------------------------
# def sph_to_cart(theta_deg, phi_deg):
#     """
#     theta: azimuth in degrees
#     phi: elevation in degrees
#     returns x,y,z on unit sphere
#     """
#     th = np.deg2rad(theta_deg)
#     ph = np.deg2rad(phi_deg)

#     x = np.cos(ph) * np.cos(th)
#     y = np.cos(ph) * np.sin(th)
#     z = np.sin(ph)
#     return x, y, z

# def tangent_vector_on_sphere(theta_deg, phi_deg, dtheta_deg, dphi_deg):
#     """
#     Convert a local vector (dtheta, dphi) in degree coordinates
#     into a 3D tangent vector on the unit sphere.

   
#     """
#     th = np.deg2rad(theta_deg)
#     ph = np.deg2rad(phi_deg)
#     dth = np.deg2rad(dtheta_deg)
#     dph = np.deg2rad(dphi_deg)

#     # Partial wrt theta
#     dr_dtheta = np.array([
#         -np.cos(ph) * np.sin(th),
#          np.cos(ph) * np.cos(th),
#          0.0
#     ])

#     # Partial wrt phi
#     dr_dphi = np.array([
#         -np.sin(ph) * np.cos(th),
#         -np.sin(ph) * np.sin(th),
#          np.cos(ph)
#     ])

#     vec = dth * dr_dtheta + dph * dr_dphi
#     return vec

# # -----------------------------
# # 4. Plot the same streamlines on the sphere
# # -----------------------------
# ax2 = fig.add_subplot(122, projection='3d')

# # Sphere wireframe
# u, v = np.mgrid[0:2*np.pi:40j, -np.pi/2:np.pi/2:25j]
# xs = np.cos(v) * np.cos(u)
# ys = np.cos(v) * np.sin(u)
# zs = np.sin(v)
# ax2.plot_wireframe(xs, ys, zs, color='gray', alpha=0.25, linewidth=0.5)

# # Extract streamline segments from the 2D streamplot
# segments_2d = strm.lines.get_segments()

# # Map each 2D streamline segment to 3D
# segments_3d = []
# segment_colors = []

# norm = colors.Normalize(vmin=np.nanmin(speed), vmax=np.nanmax(speed))
# cmap = cm.get_cmap("viridis")

# for seg in segments_2d:
#     # seg is shape (2,2): [[theta1, phi1], [theta2, phi2]]
#     (th1, ph1), (th2, ph2) = seg

#     x1, y1, z1 = sph_to_cart(th1, ph1)
#     x2, y2, z2 = sph_to_cart(th2, ph2)

#     segments_3d.append([[x1, y1, z1], [x2, y2, z2]])

#     # color by midpoint speed from interpolated field
#     thm = 0.5 * (th1 + th2)
#     phm = 0.5 * (ph1 + ph2)

#     # nearest grid cell for a simple color lookup
#     i = np.argmin(np.abs(x - thm))
#     j = np.argmin(np.abs(y - phm))
#     segment_colors.append(cmap(norm(speed[j, i])))

# lc = Line3DCollection(segments_3d, colors=segment_colors, linewidths=1.2)
# ax2.add_collection3d(lc)

# # -----------------------------
# # 5. Optional: local tangent arrows on the sphere
# # -----------------------------
# # only show arrows where data were originally supported
# # arrow_mask = valid.copy()

# # # Reduce arrow density for readability
# # arrow_stride_theta = 2
# # arrow_stride_phi = 2
# # arrow_mask_sparse = np.zeros_like(arrow_mask, dtype=bool)
# # arrow_mask_sparse[::arrow_stride_phi, ::arrow_stride_theta] = True
# # arrow_mask_sparse &= arrow_mask

# # # Base points on sphere
# # bx, by, bz = sph_to_cart(X[arrow_mask_sparse], Y[arrow_mask_sparse])

# # # Tangent vectors
# # vecs = np.array([
# #     tangent_vector_on_sphere(th, ph, u0, v0)
# #     for th, ph, u0, v0 in zip(
# #         X[arrow_mask_sparse],
# #         Y[arrow_mask_sparse],
# #         U_filled[arrow_mask_sparse],
# #         V_filled[arrow_mask_sparse]
# #     )
# # ])

# # # Scale arrows for display
# # arrow_scale = 8.0
# # ax2.quiver(
# #     bx, by, bz,
# #     vecs[:, 0], vecs[:, 1], vecs[:, 2],
# #     length=arrow_scale,
# #     normalize=False,
# #     color="k",
# #     linewidth=0.7,
# #     alpha=0.5
# # )

# # Formatting
# ax2.set_title("Same vector field on the unit sphere")
# ax2.set_xlabel("x")
# ax2.set_ylabel("y")
# ax2.set_zlabel("z")
# ax2.set_box_aspect([1, 1, 1])
# ax2.set_xlim(-1.05, 1.05)
# ax2.set_ylim(-1.05, 1.05)
# ax2.set_zlim(-1.05, 1.05)

# # Add a matching colorbar for streamline colors
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])
# cbar2 = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
# cbar2.set_label("Average vector magnitude")

# plt.tight_layout()
# plt.show()