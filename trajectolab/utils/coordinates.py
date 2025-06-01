# trajectolab/utils/coordinates.py
"""
Radically simplified coordinate transformation - ONE function for all cases.
"""

import casadi as ca

from ..tl_types import FloatArray


def tau_to_time(
    tau: float | FloatArray | ca.MX,
    mesh_start: float,
    mesh_end: float,
    time_start: float | ca.MX,
    time_end: float | ca.MX,
) -> float | FloatArray | ca.MX:
    """
    Convert tau coordinates to physical time. Works for scalars, arrays, and CasADi.

    This is the ONLY coordinate transformation function you need.

    Math: tau ∈ [-1,1] → global_tau → physical_time
    Combined: physical_time = scale * tau + offset

    Args:
        tau: Local tau coordinate(s) in [-1, 1]
        mesh_start: Global mesh start coordinate
        mesh_end: Global mesh end coordinate
        time_start: Physical start time
        time_end: Physical end time

    Returns:
        Physical time(s) corresponding to tau coordinate(s)
    """
    # Single formula that combines both transformation steps
    mesh_scale = (mesh_end - mesh_start) / 2
    mesh_offset = (mesh_end + mesh_start) / 2
    time_scale = (time_end - time_start) / 2
    time_offset = (time_end + time_start) / 2

    # Combined transformation: tau → global_tau → physical_time
    # physical_time = time_scale * (mesh_scale * tau + mesh_offset) + time_offset
    return time_scale * mesh_scale * tau + time_scale * mesh_offset + time_offset
