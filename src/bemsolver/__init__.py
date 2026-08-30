"""
BEM Stokes swimmer solver.

- Mesh: load panel geometry / normals from .mat
"""


from .mesh import Mesh
from .stokes_problems import FreeParticle, FixedParticle
from .flowfield import FlowStokes
from .SaveData import Solution
from .flagella import SlenderCurvTors, SlenderCoordinates, SlenderAngles
from .swimmers import Swimmer, FreeSwimmer

__all__ = [
    "Mesh",
    "FixedParticle",
    "FreeParticle",
    "FlowStokes",
    "Solution",
    "SlenderCurvTors",
    "SlenderCoordinates",
    "SlenderAngles",
    "Swimmer",
    "FreeSwimmer"
    ]

__version__ = "1.0.0"