"""Check that bemsolver's dependencies and environment are set up correctly.

Run with:
    uv run python scripts/check_install.py
"""

import importlib
import sys

MIN_PYTHON = (3, 10)

# (import name, pyproject name) — these differ for a couple of packages
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("imageio", "imageio"),
    ("numba", "numba"),
    ("gmsh", "gmsh"),
    ("pyvista", "pyvista"),
    ("mpi4py", "mpi4py"),
    ("h5py", "h5py"),
]


def check_python_version():
    if sys.version_info < MIN_PYTHON:
        got = f"{sys.version_info.major}.{sys.version_info.minor}"
        want = ".".join(str(x) for x in MIN_PYTHON)
        print(f"[FAIL] Python {want}+ required, found {got}")
        return False
    print(f"[ OK ] Python {sys.version.split()[0]}")
    return True


def check_packages():
    all_ok = True
    for import_name, display_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ OK ] {display_name} ({version})")
        except ImportError as e:
            print(f"[FAIL] {display_name}: {e}")
            all_ok = False
    return all_ok


def check_mpi():
    """mpi4py needs a working MPI runtime underneath it, not just the Python package."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        print(f"[ OK ] MPI runtime reachable (rank {comm.Get_rank()} of {comm.Get_size()})")
        return True
    except Exception as e:
        print(f"[FAIL] MPI runtime not working: {e}")
        return False


def check_bemsolver_importable():
    try:
        import bemsolver  # noqa: F401
        print("[ OK ] bemsolver package importable")
        return True
    except ImportError as e:
        print(f"[FAIL] bemsolver not importable: {e}")
        print("       Did you install the project itself, e.g. 'uv sync' or 'uv pip install -e .'?")
        return False


def main():
    print("Checking bemsolver installation...\n")

    checks = [
        check_python_version(),
        check_packages(),
        check_mpi(),
        check_bemsolver_importable(),
    ]

    print()
    if all(checks):
        print("All checks passed. Installation looks good.")
        sys.exit(0)
    else:
        print("One or more checks failed. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()