from scipy.io import loadmat
import warnings
from dataclasses import dataclass
import numpy as np


from .utils import find_panel_data, fix_gmsh_normals
# warnings.simplefilter("once", category=UserWarning)


@dataclass
class Mesh:
    """
    A python dataclass to extract the mesh data from the provided meshfile.

    Meshfile
    ---------------
    filepath    : The filetype of the meshfile can be .mat, .npz


    """
    filepath:str
    
    def __post_init__(self):
        
        print("Loading meshfile ...")

        self.a = None
        self.b = None

        if self.filepath.split(".")[-1]=="mat":
            self.is_mat=True
        # Maybe add a condition for filetypes like npz or other common mesh types.
            self.meshfile       =loadmat(self.filepath)

            self.isosurface     =self.meshfile["pv"]      # Analytic description of the surface of the mesh, i.e. the distance from the centerline
            self.panels         =self.meshfile["panels"]  # Shape of panels is (M x 3 x N)
                                                      # with M being amount of vertices+1 and N the amount of elements
                                                      # To select the first panel use self.panels[:,:,0]
            self.a=self.meshfile.get("a",None)[0][0]            # Major axis of ellipsoid
            self.b=self.meshfile.get("b",None) [0][0]

            self.elements   =np.shape(self.panels)[2]

        elif self.filepath.split(".")[-1]=="msh":
            import gmsh
            gmsh.initialize()
            gmsh.open(self.filepath)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes = node_coords.reshape(-1, 3)
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
            triangles = elem_node_tags[0].reshape(-1, 3).astype(int) - 1

            gmsh.finalize()

            triangles_fixed, normals = fix_gmsh_normals(nodes, triangles)
            self.is_mat = False
            

            self.panels = nodes[triangles_fixed]
            self.elements   = np.shape(self.panels)[0]

                   # Minor axis of ellipsoid

        if self.a is None or self.b is None:
            warnings.warn(f"Values for a and b are not stored in meshfile, assign them yourself using 'instance_name.a=value'. \n Also assign volume by 'instance_name.parameters[\"volume\"]=value' if needed.",
                        category=UserWarning,
                        stacklevel=3)

        self.normals,self.centroids=self.load_panels()

        # Find the center of the mesh (usually (0,0,0))
        self.center = np.array([(np.max(self.centroids[:,0])+np.min(self.centroids[:,0]))/2,
                                (np.max(self.centroids[:,1])+np.min(self.centroids[:,1]))/2,
                                (np.max(self.centroids[:,2])+np.min(self.centroids[:,2]))/2])
        # self.centroids = self.centroids - self.center
        
        self.x_max=np.max(self.centroids[:,0])
        self.x_min=np.min(self.centroids[:,0])


        # XG is the center of the centerline
        self.parameters={"XG":(self.x_max+self.x_min)/2,
                         "line_scale":0.9,
                         "Delta_rho": 30,            # kg/ m^3 density difference between cell and fluid      (values from Ishikawa 2020: doi:10.1242/jeb.205989)
                         "medium_rho":1000,          # kg/ m^3 density of the fluid
                         "COM_offset": 30 * 10**-3,  # Offset from geometric center to center of mass in micron
                         "volume":4/3*np.pi*self.a*self.b**2 if self.a is not None and self.b is not None else 0} # Volume of the ellipsoid in micron^3
        # NOTE: Volume is only correct if the mesh is an ellipsoid. If it has a different shape assign it yourself.
        print("Loading complete!")


    def load_panels(self)->tuple[np.ndarray,np.ndarray]:
        """
        Load in the normals and centroids of all the panels in the mesh.

        Returns
        -------
        normals     :   np.ndarray
                        Array with the surface normal of each element (ordered in the same way as self.panels)
        centroids   :   np.ndarray
                        Array with all the centroid coordinates of the panels
        """
        normals=np.zeros((self.elements,3))
        centroids=np.zeros((self.elements,3))

        for i in range(self.elements):
            if self.is_mat:
                panel=self.panels[1:,:,i]
            else:
                panel=self.panels[i]

            _,_,normal,centroid, _ =       find_panel_data(panel)  #from utils.py
            
            normals[i]      =normal
            centroids[i]    =centroid

        return normals,centroids
        



    def plot_mesh(self, *args, **kwargs):
        """
        Use the plotting function in plotting.py to plot the current loaded mesh.
        """
        from . import plotting  # local import so mesh.py doesn't always import matplotlib
        return plotting.plot_mesh(self, *args, **kwargs)