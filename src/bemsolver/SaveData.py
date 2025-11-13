import numpy as np
from dataclasses import dataclass, field


@dataclass
class Solution:
    """
    Dataclass to store solutions over time

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
    u                   :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    omega               :np.ndarray = field(default_factory=lambda: np.empty((0,)))

  