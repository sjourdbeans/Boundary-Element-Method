"""
BEM Stokes swimmer solver.

- Mesh: load panel geometry / normals from .mat
"""


from .mesh import Mesh
from .solver import System



__all__ = [
    "Mesh",
    "System",
]

__version__ = "0.1.0"