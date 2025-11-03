import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors
import numpy as np

def plot_mesh(
    mesh:object,
    plot_normals:bool=False,
    normal_scale: float = 0.2,
    face_alpha: float = 0.8,
    edge_alpha: float = 0.8,
):
    """
    Quick visualization of a Mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh instance (must have panels, centroids, normals).
    plot_normals : bool
        If True, plot normal vectors at panel centroids.
    normal_scale : float
        Quiver length for normals.
    face_alpha : float
        Face transparency of the surface.
    edge_alpha : float
        Edge transparency.
    """
    panels = [mesh.panels[1:, :, k] for k in range(mesh.elements)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = Poly3DCollection(
        panels,
        facecolor='lightgray',
        edgecolor='k',
        linewidth=0.3,
        alpha=face_alpha,
    )
    surf.set_edgecolor((0, 0, 0, edge_alpha))
    ax.add_collection3d(surf)

    if plot_normals:
        ax.quiver(
            mesh.centroids[:,0], mesh.centroids[:,1], mesh.centroids[:,2],
            mesh.normals[:,0], mesh.normals[:,1], mesh.normals[:,2],
            length=normal_scale,
            linewidth=0.5,
            color='blue',
            normalize=True,
            label='normals'
        )

    all_pts = mesh.panels[1:, :, :].reshape(-1, 3)
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)

    max_range = max(xmax-xmin, ymax-ymin, zmax-zmin)
    xmid = 0.5*(xmax+xmin)
    ymid = 0.5*(ymax+ymin)
    zmid = 0.5*(zmax+zmin)

    ax.set_xlim(xmid - max_range/2, xmid + max_range/2)
    ax.set_ylim(ymid - max_range/2, ymid + max_range/2)
    ax.set_zlim(zmid - max_range/2, zmid + max_range/2)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if plot_normals:
        plt.legend(frameon=False)
    plt.show()

def plot_panels_stokes(panels, f):
    """
    Panels: 3D numpy array of shape (num_panel_points+1, 3, num_panels)
            First row of each panel gives n (number of vertices)
            Remaining rows are the coordinates
    f: array of length num_panels with values to color each panel
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.cla()  # clear axis
    
    num_panels = panels.shape[2]
    
    # colormap and normalization
    cmap = cm.viridis_r
    norm = colors.Normalize(vmin=np.min(f), vmax=np.max(f))
    
    for ii in range(num_panels):
        panel = panels[1:, :, ii]   # skip first row (n)
        n = int(panels[0, 0, ii])   # number of vertices
        verts = panel[:n, :]        # take only the vertices
        
        color = cmap(norm(f[ii]))
        poly = Poly3DCollection([verts], facecolors=color, edgecolors='k')
        ax.add_collection3d(poly)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_box_aspect([1,1,1])  # axis equal
    
    # Colorbar using the same normalization
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(f)
    plt.colorbar(mappable, ax=ax)

    return fig, ax
    # plt.show()


def plot_vector_field(x, y, U_magnitude, x_quiver, y_quiver, Ux_quiver, Uy_quiver, quiver, view):

    fig = plt.figure(figsize=(8, 4))
    plt.pcolormesh(x, y, U_magnitude, shading='auto', cmap='viridis',vmax= np.average(U_magnitude)*2)
    if quiver:
        plt.quiver(x_quiver, y_quiver, Ux_quiver, Uy_quiver,
                    color='white', scale=50)
    plt.colorbar(label=r'$|\mathbf{U}_{field}|$')
    plt.xlabel(f'${view[0]}$ [$\\mu$m]')
    plt.ylabel(f'${view[1]}$ [$\\mu$m]')
    plt.title('Flow magnitude and direction')
    plt.axis('equal')
    return fig