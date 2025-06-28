try:
    from .symbolic import lagrangian_to_maptor_dynamics  # type: ignore[misc]

    __all__ = ["lagrangian_to_maptor_dynamics"]
except ImportError:

    def lagrangian_to_maptor_dynamics(*args, **kwargs):
        """Convert SymPy LagrangesMethod to MAPTOR dynamics format.

        Raises:
            ImportError: SymPy is not installed.
        """
        raise ImportError(
            "SymPy is required for mechanics utilities.\nInstall with: pip install sympy"
        )

    __all__ = ["lagrangian_to_maptor_dynamics"]
