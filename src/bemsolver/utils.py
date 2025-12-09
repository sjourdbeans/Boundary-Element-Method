import numpy as np


def find_panel_data(panel:np.ndarray)->tuple:
        """
        Find the orthonormal unit vectors of a given panel (element) and calculate the geometric center of the panel.\\
        
        Parameters
        ----------
        panel   :   np.ndarray
                    Array of shape (Mx3) with M being the amount of vertices of the panel.

        Returns
        -------
        X           :   np.ndarray
                        Unit x basis vector of the panel
        Y           :   np.ndarray
                        Unit y basis vector of the panel
        Z           :   np.ndarray
                        Unit z basis vector of the panel which is also the normal
        centroid    :   np.ndarray
                        xyz coordinates of the center of the panel
        """
        num_verts,_=np.shape(panel)
        
        # Calculate the XYZ axes for every element (panel)
        X=panel[2]-panel[0]

        if num_verts==3:
            Y=panel[1]-panel[0]  
        else:
            # If the elements have 4 vertices, use a different Y axis
            Y=panel[1]-panel[3] 

        Z=np.cross(X,Y)    
        area =0.5* np.linalg.norm(Z)

        # Normalise X and Z vectors and calculate the corresponding orthonormal Y axis
        Z   =Z/np.linalg.norm(Z)
        X   =X/np.linalg.norm(X)
        Y   =np.cross(Z,X)

        # Calculate the centroid of the panel
        vertex_1=panel[1]-panel[0]
        if num_verts==4:
            vertex_3=panel[3]-panel[0]
        else:
            vertex_3=panel[2]-panel[0]
    
        # Project into the panel axis
        y1  =np.dot(vertex_1,Y)
        y3  =np.dot(vertex_3,Y)
        x1  =np.dot(vertex_1,X)
        x3  =np.dot(vertex_3,X)
        yc  = (y1 + y3)/3
        xc  = (x1 + x3)/3

        # Compute the panel centroid
        centroid = panel[0] + xc * X + yc * Y
        
        return X,Y,Z, centroid, area


def U_colloc(U          :np.ndarray,
             W          :np.ndarray,
             centroids  :np.ndarray,
             r          :int,
             E          :np.ndarray=np.zeros((3,3)))->tuple[np.ndarray,np.ndarray]:
    """
    Calculate translational and rotational velocity at each collocation point.

    Parameters
    ----------
    U : np.ndarray, shape (3,)
        Translational velocity of external flow [micron/s].
    W : np.ndarray, shape (3,)
        Rotational velocity of external flow [rad/s].
    centroid : np.ndarray, shape (r, 3)
        XYZ coordinates of centroids of cell mesh [micron].
    r : int
        Number of rows / collocation points.
    E : np.ndarray, shape (3,3)
        Rate of strain tensor.

    Returns
    -------
    U_t : np.ndarray, shape (3*r,)
        Translational velocity vector [micron/s].
    U_r : np.ndarray, shape (3*r,)
        Rotational velocity vector [micron/s].
    U_e : np.ndarray, shape (3*r,)
        Strain rate velocity vector [micron/s].
    """
     # Translational velocity: just repeat U for each collocation point
    U_t = np.tile(U, r)

    # Rotational velocity: cross product W x centroid
    U_r = np.zeros(3*r)
    U_r[0::3] =  W[1]*centroids[:,2] - W[2]*centroids[:,1]
    U_r[1::3] = -W[0]*centroids[:,2] + W[2]*centroids[:,0]
    U_r[2::3] =  W[0]*centroids[:,1] - W[1]*centroids[:,0]

    # NEED TO ADJUST FOR GENERAL CASE

    
    U_e = (E @ centroids.T).T

    U_e = U_e.flatten() 

    
    return U_t, U_r, U_e



def points_in_polygon(x_points, y_points, poly_x, poly_y):
    """Return a boolean mask of which (x_points, y_points) lie inside a closed polygon."""
    n = len(poly_x)
    inside = np.zeros_like(x_points, dtype=bool)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        # Check if the horizontal ray crosses this polygon edge
        intersect = ((yi > y_points) != (yj > y_points)) & \
                    (x_points < (xj - xi) * (y_points - yi) / (yj - yi + 1e-15) + xi)
        inside ^= intersect
    return inside


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

    # Corrected swap
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

def skew_stack(r):
    """
    r: (M,3) input coordinates
    returns: (3M,3) stacked skew-symmetric matrices
    """
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    # Build all skew matrices in a (M,3,3) array
    A = np.zeros((len(r), 3, 3))

    A[:, 0, 1] = -z
    A[:, 0, 2] =  y
    A[:, 1, 0] =  z
    A[:, 1, 2] = -x
    A[:, 2, 0] = -y
    A[:, 2, 1] =  x

    # Stack them vertically  (3M, 3)
    return A.reshape(-1, 3)