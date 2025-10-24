from scipy.io import loadmat
import warnings
from dataclasses import dataclass, field
import numpy as np
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

        self.normals,self.centroids=self.loadPanels()

        print("Loading complete!")


    def loadPanels(self)->tuple[np.ndarray,np.ndarray]:
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
            _,_,normal,centroid =self.findPanelData(panel)
            
            normals[i]      =normal
            centroids[i]    =centroid

        return normals,centroids
        

    def findPanelData(self,panel:np.ndarray)->tuple:
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
        
        return X,Y,Z, centroid


