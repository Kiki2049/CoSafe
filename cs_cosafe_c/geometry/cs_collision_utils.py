"""Shared geometry and collision detection utilities for CoSafe.

Functions migrated from cs_cosafe_c.safety.cs_risk_oracle to be shared
between the risk oracle and the candidate selector.

When the compiled Cython kernel is available (cs_cosafe_c.kernels._geom),
the hot functions (_build_obb_2d, _sat_overlap_2d,
check_same_timestep_collision, determine_yield_aabb) are dispatched to
the compiled versions.  The Python implementations below are kept as
verified-equivalent fallbacks for use when the extension cannot be
imported.
"""

import math

import numpy as np

try:
    from cs_cosafe_c.kernels import _geom as _cy_geom
    _CY_GEOM_AVAILABLE = True
except ImportError:
    _cy_geom = None
    _CY_GEOM_AVAILABLE = False


def _infer_heading_t0(vehicle) -> float:
    """Infer heading at t=0 from trajectory direction; fallback to heading_rad."""
    if vehicle.trajectory_world:
        dx = vehicle.trajectory_world[0][0] - vehicle.position_world[0]
        dy = vehicle.trajectory_world[0][1] - vehicle.position_world[1]
        if dx * dx + dy * dy > 0.01:
            return math.atan2(dy, dx)
    return vehicle.heading_rad


def _build_obb_2d(cx, cy, heading, half_l, half_w, margin=0.0):
    """Build 2D oriented bounding box corners (4x2 ndarray)."""
    if _CY_GEOM_AVAILABLE:
        return _cy_geom.build_obb_2d(
            float(cx), float(cy), float(heading),
            float(half_l), float(half_w), float(margin))
    hl = half_l + margin
    hw = half_w + margin
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    local = np.array([
        [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw],
    ], dtype=np.float64)

    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy])


def _sat_overlap_2d(corners_a, corners_b) -> bool:
    """2D SAT test for two convex quadrilaterals. Returns True if overlapping."""
    if _CY_GEOM_AVAILABLE:
        return _cy_geom.sat_overlap_2d(corners_a, corners_b)
    for corners in [corners_a, corners_b]:
        for i in range(2):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis = axis / norm
            proj_a = corners_a @ axis
            proj_b = corners_b @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True


def _build_trajectory_corridor(vehicle, half_l, half_w, margin, d_proj):
    """Build a list of OBBs along the vehicle's trajectory + extension.

    Returns list of (4x2) ndarray corners covering:
    1. Current position OBB
    2. Each trajectory waypoint OBB
    3. Extension OBB: from last trajectory point, extend d_proj meters forward
    """
    points = [vehicle.position_world] + list(vehicle.trajectory_world)
    obbs = []

    for i, pt in enumerate(points):
        if i == 0:
            heading = _infer_heading_t0(vehicle)
        elif i < len(points):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            if dx * dx + dy * dy > 0.01:
                heading = math.atan2(dy, dx)
        obbs.append(_build_obb_2d(pt[0], pt[1], heading, half_l, half_w, margin))

    if len(points) >= 2:
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0.01:
            ext_heading = math.atan2(dy, dx)
        else:
            ext_heading = heading
    else:
        ext_heading = _infer_heading_t0(vehicle)

    last = points[-1]
    cos_h = math.cos(ext_heading)
    sin_h = math.sin(ext_heading)
    ext_cx = last[0] + cos_h * (d_proj / 2.0)
    ext_cy = last[1] + sin_h * (d_proj / 2.0)
    obbs.append(_build_obb_2d(
        ext_cx, ext_cy, ext_heading, d_proj / 2.0, half_w, margin))

    return obbs


def check_same_timestep_collision(traj_a, traj_b, half_l, half_w,
                                  margin, margin_rate, T):
    """Phase 1 collision detection with distance pre-filter.

    For each timestep t, checks if two vehicle OBBs overlap.  A vectorised
    centre-distance filter skips timesteps where overlap is geometrically
    impossible, avoiding expensive OBB construction + SAT for distant pairs.

    The distance threshold is exact (not approximate): if two OBBs overlap,
    their centres must be within 2 * half_diagonal + 2 * margin_t.

    Args:
        traj_a, traj_b: (>=T, 3) arrays [x, y, heading].
        half_l, half_w: OBB half-dimensions (metres).
        margin: base collision margin at t=0.
        margin_rate: margin growth per timestep.
        T: number of timesteps to check.

    Returns:
        int: first collision timestep (>=0), or -1 if no collision.
    """
    if _CY_GEOM_AVAILABLE:
        return _cy_geom.check_same_timestep_collision(
            traj_a, traj_b, float(half_l), float(half_w),
            float(margin), float(margin_rate), int(T))
    half_diag = math.sqrt(half_l * half_l + half_w * half_w)

    dists = np.hypot(traj_a[:T, 0] - traj_b[:T, 0],
                     traj_a[:T, 1] - traj_b[:T, 1])

    for t in range(T):
        m_t = margin + t * margin_rate
        threshold = 2.0 * (half_diag + m_t)
        if dists[t] > threshold:
            continue
        obb_a = _build_obb_2d(traj_a[t, 0], traj_a[t, 1], traj_a[t, 2],
                              half_l, half_w, m_t)
        obb_b = _build_obb_2d(traj_b[t, 0], traj_b[t, 1], traj_b[t, 2],
                              half_l, half_w, m_t)
        if _sat_overlap_2d(obb_a, obb_b):
            return t
    return -1


def determine_yield_aabb(traj_a, traj_b, half_l, half_w,
                         margin, margin_rate, T):
    """Determine yield direction via AABB corridor intersection in O(T).

    Instead of O(T^2) cross-time OBB SAT, this function:
    1. Builds an AABB corridor for each trajectory (expanded by max OBB extent).
    2. Checks corridor intersection (4 comparisons).
    3. If no intersection → INDEPENDENT.
    4. Computes the intersection AABB.
    5. Finds earliest timestep each trajectory's centre enters the intersection.
    6. Compares entry times → yield direction.

    This is a conservative approximation: corridor AABBs over-estimate spatial
    extent, so false-positive overlaps are possible.  These only produce
    unnecessary yield rules (safe), never missed collisions (Phase 1 handles
    that precisely).

    Args:
        traj_a, traj_b: (>=T, 3) arrays [x, y, heading].
        half_l, half_w: OBB half-dimensions.
        margin: base margin.
        margin_rate: margin growth per step.
        T: number of timesteps.

    Returns:
        (direction, t_enter_a, t_enter_b) where direction is one of
        'independent', 'a_yields', 'b_yields', 'collision' (tie).
    """
    if _CY_GEOM_AVAILABLE:
        return _cy_geom.determine_yield_aabb(
            traj_a, traj_b, float(half_l), float(half_w),
            float(margin), float(margin_rate), int(T))
    max_margin = margin + (T - 1) * margin_rate
    half_diag = math.sqrt((half_l + max_margin) ** 2
                          + (half_w + max_margin) ** 2)

    ax = traj_a[:T, 0]
    ay = traj_a[:T, 1]
    a_xmin, a_xmax = float(ax.min()) - half_diag, float(ax.max()) + half_diag
    a_ymin, a_ymax = float(ay.min()) - half_diag, float(ay.max()) + half_diag

    bx = traj_b[:T, 0]
    by = traj_b[:T, 1]
    b_xmin, b_xmax = float(bx.min()) - half_diag, float(bx.max()) + half_diag
    b_ymin, b_ymax = float(by.min()) - half_diag, float(by.max()) + half_diag

    if a_xmax < b_xmin or b_xmax < a_xmin:
        return ('independent', -1, -1)
    if a_ymax < b_ymin or b_ymax < a_ymin:
        return ('independent', -1, -1)

    ix_min = max(a_xmin, b_xmin)
    ix_max = min(a_xmax, b_xmax)
    iy_min = max(a_ymin, b_ymin)
    iy_max = min(a_ymax, b_ymax)

    t_enter_a = T
    for t in range(T):
        if ix_min <= traj_a[t, 0] <= ix_max and iy_min <= traj_a[t, 1] <= iy_max:
            t_enter_a = t
            break

    t_enter_b = T
    for t in range(T):
        if ix_min <= traj_b[t, 0] <= ix_max and iy_min <= traj_b[t, 1] <= iy_max:
            t_enter_b = t
            break

    if t_enter_a == T and t_enter_b == T:
        return ('independent', -1, -1)

    if t_enter_a < t_enter_b:
        return ('b_yields', t_enter_a, t_enter_b)
    elif t_enter_b < t_enter_a:
        return ('a_yields', t_enter_a, t_enter_b)
    else:
        return ('collision', t_enter_a, t_enter_b)
