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