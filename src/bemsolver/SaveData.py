import numpy as np
from dataclasses import dataclass, field


@dataclass
class Solution:
    """
    Dataclass to store solutions over time
    """

    time                :np.ndarray = field(default_factory=lambda: np.empty(0))
    X                   :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    rotation_matrices   :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    quaternions         :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    psi                 :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    u                   :np.ndarray = field(default_factory=lambda: np.empty((0,)))
    omega               :np.ndarray = field(default_factory=lambda: np.empty((0,)))

  