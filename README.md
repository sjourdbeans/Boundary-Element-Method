# Boundary Element Method

This is a repository which includes a Boundary Element Method (BEM) code to calculate the flow around a microswimmer and compute its rigid body motion. This code solves the double layer density integral equation of the second kind described by Power and Miranda with the completion flow of Keaveny and Shelley and includes the flagella using Slender Body Theory. 


## Quick start

Clone the repository and enter it:

```bash
git clone https://github.com/sjourdbeans/Boundary-Element-Method.git
cd Boundary-Element-Method
```
To use the code in this project, we must first setup a virtual environment that includes the correct python version and packages.

This project is set up for [uv](https://docs.astral.sh/uv/), which creates the virtual environment and installs the locked project dependencies in one command. Python 3.10 or newer is required.

Install `uv` if it is not already available:

<details>
<summary>Linux and macOS</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal if `uv` is not found afterwards. The [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) also lists package-manager installation options.
</details>

<details>
<summary>Windows (PowerShell)</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, install it with `winget install --id=astral-sh.uv -e`. Open a new PowerShell window after installation.
</details>

From the repository root, create the environment and install the package:

```bash
uv sync
```
Check the installation by running:

```bash
uv run python scripts/check_install.py
```
Then you can run a script inside the virtual environment with `uv run`:

```bash
uv run python path/to/your_script.py
```

On Windows, use the same `uv` commands in PowerShell. In Python paths, prefer `pathlib.Path` or forward slashes (`datafiles/mesh/body.mat`) so that scripts work on every platform.

## Meshes

The solver needs a closed triangular surface mesh of the cell body before it can assemble a system. The repository includes a small spheroid mesh generator in [`mesher-code/mesher.py`](mesher-code/mesher.py). Edit the geometry and output path near the bottom of that file, create the output directory if necessary, then run:

```bash
uv run python mesher-code/mesher.py
```

The generator writes a MATLAB `.mat` file containing a prolate spheroid. Increase `subdiv` for a finer mesh; each subdivision multiplies the triangle count by four, and hence makes BEM assembly and solves substantially more expensive.

`bem.Mesh` accepts the following formats:

| Format | Contents |
| --- | --- |
| `.mat` | The legacy format produced by the included generator. It requires `panels` with shape `(4, 3, N)` and `pv`; `a` and `b` are optional semi-axes. |
| `.npz` | A NumPy archive with the same named arrays as the `.mat` format, or `panels` in the standard `(N, 3, 3)` triangle layout. Include `pv` when using `FlowStokes` or mesh/flow plotting. |
| `.msh` | A triangular surface mesh readable by the Python `gmsh` package. The loader corrects triangle orientation when it is read. |

For every format, `panels` must describe a closed, non-degenerate surface. The supplied `.mat` convention reserves `panels[0]` for legacy metadata; the three vertices of triangle `i` are `panels[1:, :, i]`. For a standard NumPy layout, each triangle is simply `panels[i]` with shape `(3, 3)`.

`pv` is an axial outline with two columns, `[x, radial_distance]`. It is used to mask the interior in flow-field plots; it is not needed for a fixed-particle or free-particle solve. A `.msh` mesh does not supply `pv`, `a`, `b`, or volume automatically. Set `mesh.parameters["volume"]` yourself for gravity/buoyancy calculations, and set `mesh.isosurface` if you will use `FlowStokes.calc_vector_field` or plotting.

Example NumPy mesh archive:

```python
import numpy as np

# triangles has shape (number_of_triangles, 3, 3)
np.savez("body.npz", panels=triangles, pv=pv, a=semi_major_axis, b=semi_minor_axis)
```

## Using `bemsolver`

The package is installed from `src/` by `uv sync` and is imported as:

```python
import bemsolver as bem
```

Start by loading a mesh:

```python
mesh = bem.Mesh("datafiles/mesh/body.mat")
mesh.plot_mesh()  # optional visual check
```

`bem.Mesh` has a `parameters` dictionary as attribute that contains

| Parameter | Definition | Standard Value |
| --- | --- | --- |
| `XG` | Centre of the centreline, calculated with the maximum and minimum x coordinate of the mesh. | 0.5*(max(x)+ min(x)) |
| `line_scale` | Scaling factor to decrease the length of the line singularity inside the mesh. A factor of 0.9 means that the line does not reach the maximum and minimum point of the mesh. A factor of 0 means a point singularity (similar to the completion flow of Power and Miranda). | 0.9 |
| `Delta_rho` |  Density difference in kg/m^3 between particle/cell and fluid medium. | 0 |
| `medium_rho` | Density of the fluid medium in kg/m^3 | 1000 |
| `COM_offset` | The Centre-of-Mass offset from the geometric centre in microns. | 0 |
| `volume` | The volume of the mesh is automatically assumed to be a spheroid. If that is not the case, manually set the volume. | `(4/3)*pi*a*b^2` if a and b are available, otherwise 0. |

For example, to give your particle a bottom heaviness of 0.1 microns use:
```python
mesh = bem.Mesh("datafiles/mesh/body.mat")
mesh.parameters["COM_offset"] = 0.1
```

### Fixed body: force and torque in a prescribed flow

Use `FixedParticle` when the body is clamped. `U`, `W`, and `E` are respectively the translational flow, vorticity vector, and rate-of-strain tensor.

```python
import numpy as np
import bemsolver as bem

mesh = bem.Mesh("datafiles/mesh/body.mat")
system = bem.FixedParticle(mesh)

U = np.array([1.0, 0.0, 0.0])
W = np.zeros(3)
E = np.zeros((3, 3))

psi, force, torque = system.solve(U, W, E)
print("force:", force)
print("torque:", torque)
```

### Free particle: translation and rotation over time

Use `FreeParticle` for a force- and torque-free body. Supply a function that takes the current simulation time `t` and position of the particle `x` and returns `(U, W, E)`. By including the `t` argument, the background flow can be time dependent. `RBM_over_time` returns a `Solution` object with the time, positions, rotation matrices, quaternions, velocities, angular velocities, and double-layer density history.

`RBM_over_time` takes either the Euler angles pitch, yaw, and roll as initial orientation with `initial_orientation` or as a quaternion unit vector with `initial_quaternion`.


```python
import numpy as np
import bemsolver as bem

mesh = bem.Mesh("datafiles/mesh/body.mat")
system = bem.FreeParticle(mesh)

def shear_flow(t, x):
    shear_rate = 1.0
    U = np.array([shear_rate * x[1], 0.0, 0.0])
    W = np.array([0.0, 0.0, -shear_rate])
    E = shear_rate / 2 * np.array([[0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]])
    return U, W, E

# pitch, yaw, and roll
p, y, r = 0, 0, 0

initial_orientation =np.array([p, y, r])


solution = system.RBM_over_time(
    dt=0.01,
    t_end=1.0,
    flow_function=shear_flow,
    initial_position=np.zeros(3),
    initial_orientation=initial_orientation,
    #initial_quaternion=np.array([1, 0, 0, 0])      # Or a unit quaternion as   orientation
)
print(solution.X)
```

### Velocity field around a solved body

After solving a fixed-body problem, evaluate the flow at points away from the surface with `FlowStokes`:

```python
points = np.array([[6.0, 0.0, 0.0],
                   [0.0, 6.0, 0.0]])
field = bem.FlowStokes(mesh, points)
velocity = field.calc_vector_field(psi, U, W, E)
```

`FlowStokes` constructs an interaction matrix during initialisation, so make the evaluation grid only as fine as needed.

### Flagella and swimmers

For a centerline already represented by Cartesian points, construct each frame with `bem.SlenderCoordinates`. `bem.SlenderAngles` and `bem.SlenderCurvTors` are available when the waveform is given as angles or curvature/torsion instead. A slender rod with 30 elements along the x-axis with zero material velocity is then defined as

```python
import numpy as np
import bemsolver as bem

N=30
x_points=np.linspace(0, 1 , N+1)

rod_points = np.vstack([x_points, np.zeros_like(x_points), np.zeros_like(x_points)]).T
rod = bem.SlenderCoordinates(points=rod_points, velocity=np.zeros_like(rod_points), flagellum_radius=0.2, flagellum_length=1)
```

Pass one list or two lists containing multiple flagella objects to `bem.Swimmer` (fixed body) or `bem.FreeSwimmer` (free swimming), then call `solve` or `RBM_over_time`.

The most complete working examples are in:

- [`scripts/drag/`](scripts/drag/) for drag and torque on fixed bodies;
- [`scripts/orbits/`](scripts/orbits/) for free-particle motion in shear flow;
- [`scripts/Flow/`](scripts/Flow/) for velocity-field evaluation and visualisation;
- [`scripts/Free-Swimmers/`](scripts/Free-Swimmers/) and [`scripts/Swimmers/`](scripts/Swimmers/) for constructing swimmer objects and working with flagellar waveforms.

Many scripts were written for specific research runs and contain absolute local paths and plotting preferences. Treat them as templates: change input/output paths and parameters before running them. Source files in [`src/bemsolver/`](src/bemsolver/) contain the detailed API docstrings.

## Running and troubleshooting

Run a modified example from the repository root so that relative paths resolve consistently:
To create a swimmer object (for example of Chlamydomonas), edit the directory to your mesh and where to save the swimmer object in `scripts/Free-Swimmers/CreateSwimmerObject.py` and run

```bash
uv run python scripts/Free-Swimmers/CreateSwimmerObject.py
```

This swimmer object can then be used to easily simulate trajectories by continuously looping the waveform. The trajectory (and orientation, forces, double-layer density) can be found once again with the function `RBM_over_time`.

```python
import numpy as np
import bemsolver as bem
import pickle

# Add path to your swimmer object file 
swimmer_object = "path/to/swimmer_object.pkl"

with open(swimmer_object, 'rb') as file:
    chlamy = pickle.load(file)
    file.close()


def find_flow(t,x):
    """No background flow"""
   return np.zeros(3), np.zeros(3), np.zeros((3,3))

dt = 400*10**(-6)
solution = chlamy.RBM_over_time(dt,10*(chlamy.N_frames)*dt, find_flow,initial_orientation=np.array([0, 0, 0]))
```

`gmsh`, `pyvista`, and `mpi4py` are project dependencies. If `uv sync` reports a platform-specific build problem, update `uv` first and consult the dependency's installation instructions; the solver itself does not require MPI unless your own workflow uses it.

The BEM mobility matrix is dense. Begin with a coarse mesh to verify geometry, orientation, units, and boundary conditions before increasing its resolution.
