import numpy as np
from dataclasses import dataclass, field


@dataclass
class Solution:
    """
    Dataclass to store solutions over time. 
    
    NOTE : Some fields may contain empty arrays depending on the simulation. 
    For example, if no flagellum is present, f1 and f2 will be empty arrays.

    Contains
    --------
    time               : 1D array of size (1,n) with n being the timesteps
                         timesteps of the simulation 
    X                  : solution array (n, 6)
                         Array with the position and orientation over time
    rotation_matrices  : solution array (n, 3, 3)
                         Rotation matrices to convert between lab and particle frame
    quaternions        : solution array (n, 4)
                         Quaternion vectors over time (related to the rotation matrix)
    psi                : solution array (n, 3N) with N being the amount of elements
                         Singularity distribution over the surface of the mesh
    f1                 : solution array (n, 3M) with M being the amount of elements of the flagellum
                         Force distribution on each element of the flagellum.
    f2                 : solution array (n, 3M) with M being the amount of elements of the flagellum
                         Force distribution on each element of the flagellum.
    u                  : solution array (n,3)
                         Translational velocity of the particle over time
    omega              : solution array (n,3)
                         Angular velocity of the particle over time
    """

    time                :np.ndarray = field(default_factory=lambda: np.empty(0))
    X                   :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    rotation_matrices   :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    quaternions         :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    psi                 :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    f1                  :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    f2                  :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    u                   :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    omega               :np.ndarray = field(default_factory=lambda: np.empty((0,)))
