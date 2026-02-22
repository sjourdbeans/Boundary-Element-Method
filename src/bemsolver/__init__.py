"""
BEM Stokes swimmer solver.

- Mesh: load panel geometry / normals from .mat
"""


from .mesh import Mesh
from .stokes_problems import MobilityProblem, ResistanceProblem
from .flowfield import FlowStokes
from .SaveData import Solution
from .flagella import SlenderBody, SlenderCurvTors, SlenderCoordinates, SlenderAngles
from .swimmers import Swimmer, FreeSwimmer

__all__ = [
    "Mesh",
    "MobilityProblem",
    "ResistanceProblem",
    "FlowStokes",
    "Solution",
    "SlenderBody",
    "SlenderCurvTors",
    "SlenderCoordinates",
    "SlenderAngles",
    "Swimmer",
    "FreeSwimmer"
]

__version__ = "0.1.0"