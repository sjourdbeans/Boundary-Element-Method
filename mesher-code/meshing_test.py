import bemsolver as bem
import importlib
import numpy as np
import matplotlib.pyplot as plt

def fix_gmsh_normals(nodes, triangles, center=None):
    """
    Ensures all triangle normals point outward from the given center.
    Parameters
    ----------
    nodes : (N,3) array of node coordinates
    triangles : (M,3) array of vertex indices (zero-based)
    center : (3,) array, optional center of object (default = mean of all nodes)
    Returns
    -------
    triangles_out : (M,3) array with vertex ordering corrected
    normals_out : (M,3) array of outward normals
    """
    if center is None:
        center = nodes.mean(axis=0)

    tri_coords = nodes[triangles]
    v1 = tri_coords[:,1] - tri_coords[:,0]
    v2 = tri_coords[:,2] - tri_coords[:,0]
    normals = np.cross(v1, v2)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    centroids = tri_coords.mean(axis=1)
    to_centroid = centroids - center
    dot_sign = np.einsum('ij,ij->i', normals, to_centroid)

    inward = dot_sign > 0

    # 🔧 Corrected swap
    tmp = triangles[inward, 1].copy()
    triangles[inward, 1] = triangles[inward, 2]
    triangles[inward, 2] = tmp

    # Recompute normals after flipping
    tri_coords = nodes[triangles]
    v1 = tri_coords[:,1] - tri_coords[:,0]
    v2 = tri_coords[:,2] - tri_coords[:,0]
    normals = np.cross(v1, v2)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    return triangles, normals

path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/spheroid.msh"

import gmsh
gmsh.initialize()
gmsh.open(path)

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
nodes = node_coords.reshape(-1, 3)
elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
triangles = elem_node_tags[0].reshape(-1, 3).astype(int) - 1

# print(panels)
gmsh.finalize()

triangles_fixed, normals = fix_gmsh_normals(nodes, triangles)
panels= nodes[triangles_fixed]
print(np.mean(panels.mean(axis=1),axis=0))

