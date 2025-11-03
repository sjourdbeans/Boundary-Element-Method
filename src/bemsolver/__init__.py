"""
BEM Stokes swimmer solver.

- Mesh: load panel geometry / normals from .mat
"""


from .mesh import Mesh
from .stokes_problems import MobilityProblem, ResistanceProblem
from.flowfield import FlowStokes


__all__ = [
    "Mesh",
    "MobilityProblem",
    "ResistanceProblem",
    "FlowStokes"
]

__version__ = "0.1.0"