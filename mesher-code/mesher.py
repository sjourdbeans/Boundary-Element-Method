import numpy as np
from scipy.io import savemat
import trimesh

# -------- utilities --------
def normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def build_panels(p, t):
    t0 = t.astype(int)
    Ntri = t0.shape[0]
    panels = np.empty((4, 3, Ntri), dtype=float)
    panels[0, :, :] = np.array([[3.0], [1.0], [0.0]])
    for k in range(Ntri):
        i, j, kidx = t0[k]
        panels[1, :, k] = p[i]
        panels[2, :, k] = p[j]
        panels[3, :, k] = p[kidx]
    return panels

# -------- rock-solid icosphere --------
def icosphere(subdiv=2,pole_density=1):
    """Return unit-sphere vertices p (N,3) and triangular faces t (M,3)."""
    phi = (1 + np.sqrt(5.0)) / 2.0
    verts = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [phi, 0, -1], [phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ]
    verts = np.asarray(verts, float)
    verts[:, 0] *= pole_density

    verts = normalize_rows(np.asarray(verts, float))
    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
    ], dtype=int)

    for _ in range(int(subdiv)):
        edge_mid = {}
        new_faces = []
        verts_list = verts.tolist()  # extendable

        def get_mid(i, j):
            key = (i, j) if i < j else (j, i)
            if key in edge_mid:
                return edge_mid[key]
            m = (verts[i] + verts[j]) * 0.5
            m = m / np.linalg.norm(m)
            idx = len(verts_list)
            verts_list.append(m.tolist())
            edge_mid[key] = idx
            return idx

        for f in faces:
            i, j, k = int(f[0]), int(f[1]), int(f[2])
            a = get_mid(i, j)
            b = get_mid(j, k)
            c = get_mid(k, i)
            new_faces.extend([[i, a, c], [a, j, b], [c, b, k], [a, b, c]])

        verts = np.asarray(verts_list, float)
        faces = np.asarray(new_faces, int)

    # ensure outward orientation
    cent = verts[faces].mean(axis=1)
    nrm  = np.cross(verts[faces[:,1]]-verts[faces[:,0]],
                    verts[faces[:,2]]-verts[faces[:,0]])
    flip = (np.einsum('ij,ij->i', nrm, cent) > 0)
    faces[flip] = faces[flip][:, [0,2,1]]

    return verts, faces

if __name__ == "__main__":

    # a_arr=np.arange(0.1,1,0.05)
    # -------- prolate spheroid elongated along x (uniform mesh) --------
    a = 0.5   # short radii (y,z)
    # for a in a_arr:
    c = 2  # long radius (x)
    subdiv = 2  # 20*4^2 = 320 triangles


    p_unit, t = icosphere(subdiv=subdiv,pole_density=1)

    # mesh = trimesh.Trimesh(vertices=p_unit, faces=t)
    # mesh_simplified = mesh.simplify_quadric_decimation(face_count=800)

    # a, c = 0.5, 2
    # p = mesh_simplified.vertices * np.array([c, a, a])
    # t = mesh_simplified.faces

    p = p_unit * np.array([c, a, a])  # scale to spheroid

    panels = build_panels(p, t)
    nz=301
    x = np.linspace(-c, c, nz)  # axial coordinate along the long axis (x)

    r = a * np.sqrt(np.clip(1.0 - (x/c)**2, 0.0, 1.0))  # radius in the yz-plane
    pv = np.column_stack([x,r]) 

    print("Vertices:", p.shape[0], "Triangles:", t.shape[0]) 

    # from scipy.io import savemat 
    # Can be changed to npz file later
    savemat(f"/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/jeffrey-orbits-fine/jeffrey_mesh_b={round(a,3)}.mat",
                {"p": p, "t": t+1, "panels": panels, "pv": pv, "a":c,"b":a})
