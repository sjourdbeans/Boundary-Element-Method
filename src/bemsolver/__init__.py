"""
BEM Stokes swimmer solver.

- Mesh: load panel geometry / normals from .mat
"""


from .mesh import Mesh
# from .system_base import BaseSystem
from .stokes_problems import MobilityProblem, ResistanceProblem


__all__ = [
    "Mesh",
    "MobilityProblem",
    "ResistanceProblem"
]

__version__ = "0.1.0"