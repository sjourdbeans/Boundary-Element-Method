# bemsolver Code Structure

This package implements a Boundary Element Method (BEM) solver for Stokes flow
around rigid particles and flagellated swimmers. The code is organized around a
small set of geometry objects, mobility matrix builders, solvers, and helper
functions.

## High-level Dependency Tree

```text
bemsolver
|-- Mesh
|   `-- loads panel geometry, normals, centroids, and mesh parameters
|
|-- BaseSystem
|   |-- FixedParticle
|   |-- FreeParticle
|   |-- FlowStokes
|   |-- Swimmer
|   `-- FreeSwimmer
|
|-- SlenderBody
|   |-- SlenderCoordinates
|   |-- SlenderCurvTors
|   `-- SlenderAngles
|
|-- Solution
|   `-- dataclass container for simulation output
|
`-- helper modules
    |-- kernels
    |-- quadrature
    |-- time_integration
    |-- utils
    `-- plotting
```

## Public API

The package exports the following objects from `__init__.py`:

```text
Mesh
FixedParticle
FreeParticle
FlowStokes
Solution
SlenderCurvTors
SlenderCoordinates
SlenderAngles
Swimmer
FreeSwimmer
```

## Module and Class Tree

```text
src/bemsolver/
|-- __init__.py
|   `-- exposes the main public classes listed above
|
|-- mesh.py
|   `-- class Mesh
|       |-- __post_init__()
|       |   `-- loads .mat, .npz, or .msh meshes
|       |-- load_panels()
|       |   `-- computes panel normals and centroids
|       `-- plot_mesh()
|           `-- delegates to plotting.plot_mesh()
|
|-- system_base.py
|   `-- class BaseSystem
|       |-- __post_init__()
|       |   `-- stores evaluation points and integral equation settings
|       |-- construct_mobility_matrix()
|       |   `-- builds mesh self/interactions, surface matrix, torque matrix
|       |-- set_boundary_condition(U, W, E)
|       |   `-- builds translational, rotational, and strain boundary velocities
|       `-- calc_mobility_contribution(panel)
|           `-- evaluates stresslet, Stokeslet, and rotlet panel contribution
|
|-- stokes_problems.py
|   |-- class FixedParticle(BaseSystem)
|   |   |-- solve(U, W, E)
|   |   |   `-- solves resistance problem for a clamped particle
|   |   `-- plot_singularity_density()
|   |       `-- plots the solved double-layer density components
|   |
|   `-- class FreeParticle(BaseSystem)
|       |-- __post_init__()
|       |   `-- constructs the grand mobility matrix
|       |-- construct_grand_mobility_matrix()
|       |   `-- adds force-free and torque-free constraints
|       |-- RBM_over_time(...)
|       |   `-- integrates particle rigid-body motion over time
|       |-- solve_RBM(x_initial, q_initial, t, dt)
|       |   `-- advances position and quaternion one timestep
|       |-- calc_Y_dot(t, Y)
|       |   `-- computes the ODE right-hand side
|       `-- calc_RBM(x, q, t)
|           `-- solves current singularity density, translation, rotation
|
|-- swimmers.py
|   |-- class Swimmer(BaseSystem)
|   |   |-- __post_init__()
|   |   |   `-- preallocates per-frame matrices and solution arrays
|   |   |-- populate_mobility_matrix()
|   |   |   `-- assembles fixed-swimmer body/flagella mobility matrices
|   |   |-- solve(find_flow, dt)
|   |   |   `-- solves all prescribed flagellar frames
|   |   |-- solve_step(frame_index, U, W, E)
|   |   |   `-- solves one fixed-swimmer frame
|   |   `-- calc_vector_field(interaction_object, frame_index, find_flow)
|   |       `-- evaluates the resulting flow field
|   |
|   `-- class FreeSwimmer(BaseSystem)
|       |-- __post_init__()
|       |   `-- preallocates grand mobility matrices for all frames
|       |-- populate_grand_mobility_matrix()
|       |   `-- assembles body/flagella/RBM constraint matrices
|       |-- RBM_over_time(...)
|       |   `-- integrates swimmer position and orientation over time
|       |-- solve_RBM(x_initial, q_initial, time, dt)
|       |   `-- advances swimmer state one timestep
|       |-- calc_Y_dot(t, Y)
|       |   `-- computes translation/quaternion derivatives
|       |-- calc_RBM(x, q, t)
|       |   `-- solves current singularities, swimming velocity, angular velocity
|       `-- calc_vector_field(...)
|           `-- evaluates flow around the free swimmer
|
|-- flagella.py
|   |-- class SlenderBody
|   |   |-- construct_mobility_matrix()
|   |   |   `-- builds the slender-body mobility matrix
|   |   |-- set_boundary_condition(U, W, E)
|   |   |   `-- applies background flow minus prescribed flagellar velocity
|   |   |-- calc_mobility()
|   |   |   `-- computes Tornberg-Shelley slender-body matrix
|   |   |-- calc_interaction(evaluation_points)
|   |   |   `-- computes flow influence from flagellum to other points
|   |   `-- calc_r_cross_matrix(X_center)
|   |       `-- builds stacked cross-product matrix for torque/RBM terms
|   |
|   |-- class SlenderCoordinates(SlenderBody)
|   |   `-- __post_init__()
|   |       `-- builds a flagellum from Cartesian centerline points
|   |
|   |-- class SlenderCurvTors(SlenderBody)
|   |   |-- __post_init__()
|   |   |   `-- initializes Frenet-Serret integration data
|   |   `-- calc_curve()
|   |       `-- integrates curvature/torsion to Cartesian coordinates
|   |
|   |-- class SlenderAngles(SlenderBody)
|   |   `-- __post_init__()
|   |       `-- builds a flagellum from in-plane and out-of-plane angles
|   |
|   `-- _rk4_step(f, y, s, ds)
|       `-- helper used by SlenderCurvTors.calc_curve()
|
|-- flowfield.py
|   `-- class FlowStokes(BaseSystem)
|       |-- __post_init__()
|       |   `-- builds interaction matrix at off-surface evaluation points
|       |-- set_background_flow(U, W, E)
|       |   `-- computes background flow at evaluation points
|       |-- calc_vector_field(psi, U, W, E)
|       |   `-- evaluates velocity field from solved density plus background flow
|       |-- plot_vector_field(...)
|       |   `-- plots gridded vector field output
|       `-- set_boundary_condition()
|           `-- intentionally unavailable for FlowStokes
|
|-- SaveData.py
|   `-- class Solution
|       `-- stores time, position, orientation, singularity densities,
|           flagellar forces, translational velocity, and angular velocity
|
|-- kernels.py
|   |-- stresslet_vectorized(...)
|   |   `-- vectorized stresslet integration over all collocation points
|   |-- line_singularity_vectorized(...)
|   |   `-- vectorized Stokeslet/rotlet line singularity integration
|   |-- stokeslet(Xij, Yij, Zij)
|   |   `-- slender-body Stokeslet matrix blocks
|   |-- tangential(Lij, Sij, T)
|   |   `-- tangential slender-body correction blocks
|   |-- stresslet(...)
|   |   `-- non-vectorized/numba stresslet implementation, marked unused
|   `-- line_singularity(...)
|       `-- non-vectorized/numba line singularity implementation, marked unused
|
|-- quadrature.py
|   `-- triquad(N_in, v)
|       `-- Gaussian quadrature rule for triangular panels
|
|-- time_integration.py
|   |-- pyr_to_quat(pitch, yaw, roll)
|   |-- vector_to_quaternion_from_x(p)
|   |-- RK4(RHS, Y, t, dt)
|   |-- forward_euler(RHS, Y, t, dt)
|   |-- rk2(RHS, Y, t, dt)
|   |-- rotate_BCs(Q, U, W, E)
|   |-- omega_to_quat_dot(q, omega)
|   |-- quat_multiply(q1, q2)
|   `-- quat_exp(omega, dt)
|
|-- utils.py
|   |-- find_panel_data(panel)
|   |-- U_colloc(U, W, centroids, r, E)
|   |-- points_in_polygon(x_points, y_points, poly_x, poly_y)
|   |-- fix_gmsh_normals(nodes, triangles, center)
|   `-- skew_stack(r)
|
`-- plotting.py
    |-- plot_mesh(...)
    |-- plot_panels_stokes(panels, f)
    `-- plot_vector_field(...)
```

## Main Workflows

### Fixed rigid particle

```text
Mesh
`-- FixedParticle(mesh)
    |-- construct_mobility_matrix()
    |-- set_boundary_condition(U, W, E)
    `-- solve(U, W, E)
        `-- Solution-like outputs: psi, force, torque
```

### Free rigid particle

```text
Mesh
`-- FreeParticle(mesh)
    |-- construct_grand_mobility_matrix()
    |-- RBM_over_time(dt, t_end, flow_function, ...)
    |   |-- solve_RBM(...)
    |   |-- calc_Y_dot(...)
    |   `-- calc_RBM(...)
    `-- Solution
```

### Fixed swimmer with prescribed flagella

```text
Mesh
SlenderBody frames
`-- Swimmer(mesh, flagellum_1, flagellum_2=None)
    |-- populate_mobility_matrix()
    |   |-- BaseSystem.construct_mobility_matrix()
    |   |-- SlenderBody.construct_mobility_matrix()
    |   |-- SlenderBody.calc_interaction()
    |   `-- FlowStokes(mesh, flagellum.r).MATRIX
    |-- solve(find_flow, dt)
    `-- calc_vector_field(...)
```

### Free swimmer with prescribed flagella

```text
Mesh
SlenderBody frames
`-- FreeSwimmer(mesh, flagellum_1, flagellum_2=None)
    |-- populate_grand_mobility_matrix()
    |   `-- assembles body, flagella, force/torque constraints, and RBM blocks
    |-- RBM_over_time(dt, t_end, flow_function, ...)
    |   |-- solve_RBM(...)
    |   |-- calc_Y_dot(...)
    |   `-- calc_RBM(...)
    `-- Solution
```

## Data Containers and Common Attributes

```text
Mesh
|-- panels
|-- normals
|-- centroids
|-- elements
|-- center
|-- isosurface
`-- parameters
    |-- XG
    |-- line_scale
    |-- Delta_rho
    |-- medium_rho
    |-- COM_offset
    `-- volume

BaseSystem and children
|-- mesh
|-- evaluation_points
|-- MATRIX
|-- surface_matrix
|-- torque_matrix
`-- r_cross_matrix

SlenderBody and children
|-- r
|-- tangents
|-- velocity
|-- flag_centroids
|-- element_lengths
|-- Nf
|-- flagellum_radius
|-- flagellum_length
`-- slend_2

Solution
|-- time
|-- X
|-- rotation_matrices
|-- quaternions
|-- psi
|-- f1
|-- f2
|-- u
`-- omega
```

## Conceptual Notes

- `BaseSystem` is the core panel-BEM implementation for the cell body. Child
  classes change the problem setup: fixed resistance solve, free RBM solve,
  off-surface flow evaluation, fixed swimmer solve, or free swimmer solve.
- `SlenderBody` is the core slender-body implementation for flagella. Child
  classes only differ in how the flagellar centerline is constructed.
- `FlowStokes` is an interaction/evaluation helper. It computes how the solved
  surface singularities affect arbitrary points in space.
- `Solution` is intentionally broad: different simulations fill different
  fields.
- The vectorized kernels in `kernels.py` are the active high-performance path
  used by `BaseSystem`; the non-vectorized numba versions are kept as reference
  or fallback code and are marked unused in the source.
