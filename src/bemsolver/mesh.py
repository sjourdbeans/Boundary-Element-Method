from scipy.io import loadmat
import warnings
from dataclasses import dataclass
import numpy as np


from .utils import find_panel_data
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

        # Maybe add a condition for filetypes like npz or other common mesh types.
        self.meshfile       =loadmat(self.filepath)

        self.isosurface     =self.meshfile["pv"]      # Analytic description of the surface of the mesh, i.e. the distance from the centerline
        self.panels         =self.meshfile["panels"]  # Shape of panels is (M x 3 x N)
                                                      # with M being amount of vertices+1 and N the amount of elements
                                                      # To select the first panel use self.panels[:,:,0]

        self.elements   =np.shape(self.panels)[2]     # Amount of elements  
        
        self.a=self.meshfile.get("a",None)            # Major axis of ellipsoid
        self.b=self.meshfile.get("b",None)            # Minor axis of ellipsoid

        if self.a is None or self.b is None:
            warnings.warn(f"Values for a and b are not stored in meshfile, assign them yourself using 'instance_name.a=value'.",
                        category=UserWarning,
                        stacklevel=3)

        self.normals,self.centroids=self.load_panels()
        self.x_max=np.max(self.centroids[:,0])
        self.x_min=np.min(self.centroids[:,0])

        # XG is the center of the centerline
        self.parameters={"XG":(self.x_max-self.x_min)/2,
                         "line_scale":0.9}

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
            panel=self.panels[1:,:,i]
            _,_,normal,centroid =       find_panel_data(panel)  #from utils.py
            
            normals[i]      =normal
            centroids[i]    =centroid

        return normals,centroids
        



    def plot_mesh(self, *args, **kwargs):
        """
        Use the plotting function in plotting.py
        """
        from . import plotting  # local import so mesh.py doesn't always import matplotlib
        return plotting.plot_mesh(self, *args, **kwargs)