"""
BEM Stokes swimmer solver.

- Mesh: load panel geometry / normals from .mat
"""


from .mesh import Mesh
from .solver import Solver



__all__ = [
    "Mesh",
    "Solver",
]

__version__ = "0.1.0"