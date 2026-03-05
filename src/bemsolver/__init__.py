"""
BEM Stokes swimmer solver.

- Mesh: load panel geometry / normals from .mat
"""


from .mesh import Mesh
from .stokes_problems import MobilityProblem, ResistanceProblem
from .flowfield import FlowStokes
from .SaveData import Solution
from .flagella import SlenderCurvTors, SlenderCoordinates, SlenderAngles
from .swimmers import Swimmer, FreeSwimmer
from .plotting import plot_panels_stokes

__all__ = [
    "Mesh",
    "MobilityProblem",
    "ResistanceProblem",
    "FlowStokes",
    "Solution",
    "SlenderCurvTors",
    "SlenderCoordinates",
    "SlenderAngles",
    "Swimmer",
    "FreeSwimmer",
    "plot_panels_stokes"
]

__version__ = "0.1.0"