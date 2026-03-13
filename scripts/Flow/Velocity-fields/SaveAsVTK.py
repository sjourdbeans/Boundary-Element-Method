import json
from pathlib import Path
import pickle

import numpy as np
import pyvista as pv
import bemsolver as bem

shear_rate = 0

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


# =========================
# User settings
# =========================
filechlamy = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/swimmer-objects/Chlamy/free/chlamy_free_3d_waveform.pkl"

outdir = Path("/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/chlamy_vtk_export")
frames_dir = outdir / "frames"
geom_dir = outdir / "geometry"
flag1_dir = geom_dir / "flagella_1"
flag2_dir = geom_dir / "flagella_2"

Nx = 51
Ny = 51
Nz = 51

xlim = 20
ylim = 20
zlim = 20

dt = 400e-6

# Save arrays as float32 to reduce file size
USE_FLOAT32 = True


# =========================
# Helpers
# =========================
def as_xyz_points(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r)
    if r.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {r.shape}")
    if r.shape[1] == 3:
        return r
    if r.shape[0] == 3:
        return r.T
    raise ValueError(f"Cannot interpret coordinates with shape {r.shape}")


def arr_dtype(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) if USE_FLOAT32 else a.astype(np.float64)




# =========================
# Prepare output folders
# =========================
frames_dir.mkdir(parents=True, exist_ok=True)
flag1_dir.mkdir(parents=True, exist_ok=True)
flag2_dir.mkdir(parents=True, exist_ok=True)

print(f"Export directory: {outdir}")


# =========================
# Load swimmer and solve RBM
# =========================
with open(filechlamy, "rb") as f:
    chlamy = pickle.load(f)

solution = chlamy.RBM_over_time(
    dt,
    chlamy.N_frames * dt,
    find_flow,
    initial_orientation=np.array([0, 0, 0]),
)

print(f"N_frames = {chlamy.N_frames}")


# =========================
# Build regular grid
# =========================
x = np.linspace(-xlim, xlim, Nx)
y = np.linspace(-ylim, ylim, Ny)
z = np.linspace(-zlim, zlim, Nz)

dx = float(abs(x[1] - x[0]))
dy = float(abs(y[1] - y[0]))
dz = float(abs(z[1] - z[0]))

xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
points = np.column_stack((
    xg.ravel(order="F"),
    yg.ravel(order="F"),
    zg.ravel(order="F"),
))

interaction = bem.FlowStokes(chlamy.mesh, points)


# =========================
# Save body mesh once
# =========================
panels = chlamy.mesh.panels[1:, :, :]
mesh_points = panels.transpose(2, 0, 1).reshape(-1, 3)
unique_points, inverse = np.unique(mesh_points, axis=0, return_inverse=True)
triangles = inverse.reshape(-1, 3)

body_mesh = pv.make_tri_mesh(arr_dtype(unique_points), triangles)
body_mesh.save(geom_dir / "body_mesh.vtp")
print("Saved body mesh")


# =========================
# Pass 1: compute global max speed
# =========================
max_speed = 1800    #microns/s

# for frame in range(chlamy.N_frames):
#     u_field = chlamy.calc_vector_field(
#         interaction,
#         frame,
#         find_flow,
#         include_rbm=False,
#     )
#     speed = np.linalg.norm(u_field, axis=1)
#     frame_max = float(np.nanmax(speed))
#     if frame_max > max_speed:
#         max_speed = frame_max

print(f"Global max speed = {max_speed:.6g}")


# =========================
# Pass 2: export each frame
# =========================
for frame in range(chlamy.N_frames):
    print(f"Exporting frame {frame+1}/{chlamy.N_frames}")

    u_field = chlamy.calc_vector_field(
        interaction,
        frame,
        find_flow,
        include_rbm=False,
    )
    speed = np.linalg.norm(u_field, axis=1)
    umag = speed / max_speed if max_speed > 0 else speed.copy()

    grid = pv.ImageData(
        dimensions=(Nx, Ny, Nz),
        spacing=(dx, dy, dz),
        origin=(float(x.min()), float(y.min()), float(z.min())),
    )

    grid["u"] = arr_dtype(u_field)
    grid["speed"] = arr_dtype(speed)
    grid["umag"] = arr_dtype(umag)
    grid.set_active_vectors("u")

    # Derivative-based quantities
    deriv_grid = grid.compute_derivative(
        scalars="u",
        gradient=False,
        vorticity=True,
        qcriterion=True,
        preference="point",
    )

    if "vorticity" in deriv_grid.cell_data or "qcriterion" in deriv_grid.cell_data:
        deriv_grid = deriv_grid.cell_data_to_point_data()

    vorticity = np.asarray(deriv_grid["vorticity"])
    vort_mag = np.linalg.norm(vorticity, axis=1)
    qcriterion = np.asarray(deriv_grid["qcriterion"])
    grid["qcriterion"] = arr_dtype(qcriterion)

    grid["vorticity"] = arr_dtype(vorticity)
    grid["vort_mag"] = arr_dtype(vort_mag)

    frame_path = frames_dir / f"frame_{frame:04d}.vti"
    grid.save(frame_path)

    # Save flagella as polylines
    r1 = as_xyz_points(chlamy.flagellum_1[frame].r)
    r2 = as_xyz_points(chlamy.flagellum_2[frame].r)

    line1 = pv.lines_from_points(arr_dtype(r1))
    line2 = pv.lines_from_points(arr_dtype(r2))

    line1.save(flag1_dir / f"frame_{frame:04d}.vtp")
    line2.save(flag2_dir / f"frame_{frame:04d}.vtp")


# =========================
# Save metadata
# =========================
metadata = {
    "n_frames": int(chlamy.N_frames),
    "Nx": int(Nx),
    "Ny": int(Ny),
    "Nz": int(Nz),
    "xlim": float(xlim),
    "ylim": float(ylim),
    "zlim": float(zlim),
    "dx": dx,
    "dy": dy,
    "dz": dz,
    "dt": float(dt),
    "max_speed": float(max_speed),
    "float_dtype": "float32" if USE_FLOAT32 else "float64",
    "shear_rate": float(shear_rate),
}

with open(outdir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Done.")