"""CoSafe geometry and collision detection utilities."""

from cs_cosafe_c.geometry.cs_collision_utils import (  # noqa: F401
    _build_obb_2d,
    _build_trajectory_corridor,
    _infer_heading_t0,
    _sat_overlap_2d,
    check_same_timestep_collision,
    determine_yield_aabb,
)
