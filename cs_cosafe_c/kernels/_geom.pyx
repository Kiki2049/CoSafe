# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython geometric kernels for CoSafe.

Provides accelerated implementations of:
    - _build_obb_2d   (OBB corner construction, given (cx, cy, heading))
    - _sat_overlap_2d (separating-axis-theorem 2D overlap test)
    - check_same_timestep_collision
    - determine_yield_aabb

Output is byte-identical to the Python implementations in
cs_cosafe_c.geometry.cs_collision_utils, verified by
cs_cosafe_c/tests/test_geom_kernel.py.
"""
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, cos, sin


# ---------------------------------------------------------------------
# Inline helpers (nogil, no Python objects)
# ---------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _build_obb_inline(double cx, double cy, double h,
                                   double hl, double hw, double margin,
                                   double[:, :] out) noexcept nogil:
    """Build OBB corners in 'out' (4x2 contiguous buffer)."""
    cdef double ehl = hl + margin
    cdef double ehw = hw + margin
    cdef double cs = cos(h)
    cdef double sn = sin(h)
    out[0, 0] = cx + (-ehl) * cs - (-ehw) * sn
    out[0, 1] = cy + (-ehl) * sn + (-ehw) * cs
    out[1, 0] = cx + ehl * cs - (-ehw) * sn
    out[1, 1] = cy + ehl * sn + (-ehw) * cs
    out[2, 0] = cx + ehl * cs - ehw * sn
    out[2, 1] = cy + ehl * sn + ehw * cs
    out[3, 0] = cx + (-ehl) * cs - ehw * sn
    out[3, 1] = cy + (-ehl) * sn + ehw * cs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _sat_overlap_inline(double[:, :] a, double[:, :] b) noexcept nogil:
    """SAT overlap test on two 4x2 corner sets."""
    cdef int p, i, k, ni
    cdef double edge_x, edge_y, axis_x, axis_y, norm
    cdef double a_min, a_max, b_min, b_max
    cdef double v

    # 2 edges per rectangle (perpendicular axes only), 2 rectangles -> 4 axes.
    for p in range(2):
        for i in range(2):
            ni = (i + 1) & 3
            if p == 0:
                edge_x = a[ni, 0] - a[i, 0]
                edge_y = a[ni, 1] - a[i, 1]
            else:
                edge_x = b[ni, 0] - b[i, 0]
                edge_y = b[ni, 1] - b[i, 1]
            axis_x = -edge_y
            axis_y = edge_x
            norm = sqrt(axis_x * axis_x + axis_y * axis_y)
            if norm < 1e-10:
                continue
            axis_x /= norm
            axis_y /= norm

            # Project A
            a_min = a[0, 0] * axis_x + a[0, 1] * axis_y
            a_max = a_min
            for k in range(1, 4):
                v = a[k, 0] * axis_x + a[k, 1] * axis_y
                if v < a_min:
                    a_min = v
                if v > a_max:
                    a_max = v

            # Project B
            b_min = b[0, 0] * axis_x + b[0, 1] * axis_y
            b_max = b_min
            for k in range(1, 4):
                v = b[k, 0] * axis_x + b[k, 1] * axis_y
                if v < b_min:
                    b_min = v
                if v > b_max:
                    b_max = v

            if a_max < b_min or b_max < a_min:
                return False
    return True


# ---------------------------------------------------------------------
# Python-callable entry points
# ---------------------------------------------------------------------

def build_obb_2d(double cx, double cy, double heading,
                  double half_l, double half_w, double margin=0.0):
    """Public wrapper matching cs_collision_utils._build_obb_2d.

    Returns a new (4, 2) numpy.float64 array.
    """
    cdef cnp.ndarray[double, ndim=2] out = np.empty((4, 2), dtype=np.float64)
    cdef double[:, :] outv = out
    _build_obb_inline(cx, cy, heading, half_l, half_w, margin, outv)
    return out


def sat_overlap_2d(corners_a, corners_b):
    """Public wrapper matching cs_collision_utils._sat_overlap_2d."""
    cdef cnp.ndarray[double, ndim=2] a = np.ascontiguousarray(
        corners_a, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] b = np.ascontiguousarray(
        corners_b, dtype=np.float64)
    cdef double[:, :] av = a
    cdef double[:, :] bv = b
    cdef bint result
    with nogil:
        result = _sat_overlap_inline(av, bv)
    return bool(result)


def check_same_timestep_collision(traj_a, traj_b, double half_l, double half_w,
                                   double margin, double margin_rate, int T):
    """Cython port of check_same_timestep_collision.

    Returns first colliding timestep index in [0, T) or -1.
    """
    cdef cnp.ndarray[double, ndim=2] ta = np.ascontiguousarray(
        traj_a[:T], dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tb = np.ascontiguousarray(
        traj_b[:T], dtype=np.float64)
    cdef double[:, :] tav = ta
    cdef double[:, :] tbv = tb
    cdef int t
    cdef double m_t, threshold, dx, dy, dist
    cdef double hl = half_l, hw = half_w
    cdef double half_diag = sqrt(hl * hl + hw * hw)

    # OBB buffers (allocated once, reused per step).
    cdef cnp.ndarray[double, ndim=2] obb_a_np = np.empty((4, 2), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] obb_b_np = np.empty((4, 2), dtype=np.float64)
    cdef double[:, :] obb_a = obb_a_np
    cdef double[:, :] obb_b = obb_b_np
    cdef int found = -1

    with nogil:
        for t in range(T):
            m_t = margin + t * margin_rate
            threshold = 2.0 * (half_diag + m_t)
            dx = tav[t, 0] - tbv[t, 0]
            dy = tav[t, 1] - tbv[t, 1]
            dist = sqrt(dx * dx + dy * dy)
            if dist > threshold:
                continue
            _build_obb_inline(tav[t, 0], tav[t, 1], tav[t, 2],
                              hl, hw, m_t, obb_a)
            _build_obb_inline(tbv[t, 0], tbv[t, 1], tbv[t, 2],
                              hl, hw, m_t, obb_b)
            if _sat_overlap_inline(obb_a, obb_b):
                found = t
                break
    return found


def determine_yield_aabb(traj_a, traj_b, double half_l, double half_w,
                         double margin, double margin_rate, int T):
    """Cython port of determine_yield_aabb.

    Returns (direction_str, t_enter_a, t_enter_b) where direction is one of
        'independent', 'a_yields', 'b_yields', 'collision' (tie).
    """
    cdef cnp.ndarray[double, ndim=2] ta = np.ascontiguousarray(
        traj_a[:T], dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tb = np.ascontiguousarray(
        traj_b[:T], dtype=np.float64)
    cdef double[:, :] tav = ta
    cdef double[:, :] tbv = tb

    cdef double max_margin = margin + (T - 1) * margin_rate
    cdef double half_diag = sqrt(
        (half_l + max_margin) * (half_l + max_margin) +
        (half_w + max_margin) * (half_w + max_margin))

    cdef double a_xmin, a_xmax, a_ymin, a_ymax
    cdef double b_xmin, b_xmax, b_ymin, b_ymax
    cdef double v
    cdef int t

    a_xmin = tav[0, 0]; a_xmax = tav[0, 0]
    a_ymin = tav[0, 1]; a_ymax = tav[0, 1]
    b_xmin = tbv[0, 0]; b_xmax = tbv[0, 0]
    b_ymin = tbv[0, 1]; b_ymax = tbv[0, 1]
    for t in range(1, T):
        v = tav[t, 0]
        if v < a_xmin: a_xmin = v
        if v > a_xmax: a_xmax = v
        v = tav[t, 1]
        if v < a_ymin: a_ymin = v
        if v > a_ymax: a_ymax = v
        v = tbv[t, 0]
        if v < b_xmin: b_xmin = v
        if v > b_xmax: b_xmax = v
        v = tbv[t, 1]
        if v < b_ymin: b_ymin = v
        if v > b_ymax: b_ymax = v

    a_xmin -= half_diag; a_xmax += half_diag
    a_ymin -= half_diag; a_ymax += half_diag
    b_xmin -= half_diag; b_xmax += half_diag
    b_ymin -= half_diag; b_ymax += half_diag

    if a_xmax < b_xmin or b_xmax < a_xmin:
        return ('independent', -1, -1)
    if a_ymax < b_ymin or b_ymax < a_ymin:
        return ('independent', -1, -1)

    cdef double ix_min = a_xmin if a_xmin > b_xmin else b_xmin
    cdef double ix_max = a_xmax if a_xmax < b_xmax else b_xmax
    cdef double iy_min = a_ymin if a_ymin > b_ymin else b_ymin
    cdef double iy_max = a_ymax if a_ymax < b_ymax else b_ymax

    cdef int t_enter_a = T
    cdef int t_enter_b = T
    for t in range(T):
        if ix_min <= tav[t, 0] <= ix_max and iy_min <= tav[t, 1] <= iy_max:
            t_enter_a = t
            break
    for t in range(T):
        if ix_min <= tbv[t, 0] <= ix_max and iy_min <= tbv[t, 1] <= iy_max:
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
