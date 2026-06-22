# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython kernel for CsCollisionOracle's pair-collision inner loop.

CsCollisionOracle.assess iterates over (scene, vehicle_pair, timestep)
and runs distance pre-filter + OBB-SAT. The Python loop carries
significant per-iteration overhead (object attribute access, dict
membership). This kernel takes pre-extracted typed arrays and performs
the entire triple loop in C.

Public entry point:
    collision_pair_scan(positions, headings, n_per_scene,
                        dist_threshold, half_l, half_w, margin)

Returns:
    (pair_hits_dict, pair_earliest_dict)
    pair_hits_dict[(i, j)] = number of scenes in which i and j collide
                              at any future timestep
    pair_earliest_dict[(i, j)] = (t_earliest, k_earliest) -- earliest
                                  collision timestep & scene index
    where i, j are vehicle indices into the scene's vehicle list
    (caller maps them to agent_ids).
"""
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, cos, sin


# ---------------------------------------------------------------------
# Geometry inlines (copied from _geom to avoid Python-call overhead)
# ---------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _build_obb(double cx, double cy, double h,
                            double hl, double hw, double margin,
                            double[:, :] out) noexcept nogil:
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
cdef inline bint _sat_overlap(double[:, :] a, double[:, :] b) noexcept nogil:
    cdef int p, i, k, ni
    cdef double edge_x, edge_y, axis_x, axis_y, norm
    cdef double a_min, a_max, b_min, b_max, v
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
            a_min = a[0, 0] * axis_x + a[0, 1] * axis_y
            a_max = a_min
            for k in range(1, 4):
                v = a[k, 0] * axis_x + a[k, 1] * axis_y
                if v < a_min: a_min = v
                if v > a_max: a_max = v
            b_min = b[0, 0] * axis_x + b[0, 1] * axis_y
            b_max = b_min
            for k in range(1, 4):
                v = b[k, 0] * axis_x + b[k, 1] * axis_y
                if v < b_min: b_min = v
                if v > b_max: b_max = v
            if a_max < b_min or b_max < a_min:
                return False
    return True


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def collision_pair_scan(positions,
                         headings,
                         n_per_scene,
                         double dist_threshold,
                         double half_l,
                         double half_w,
                         double margin):
    """Scan all vehicle pairs in all scenes for OBB collision.

    Args:
        positions: (K, V_max, T+1, 2) float64 array.  positions[k, v, t]
            is the (x, y) of vehicle v in scene k at timestep t.
        headings:  (K, V_max, T+1)    float64 array.  Same indexing.
        n_per_scene: (K,) int32 array giving the actual number of valid
            vehicles in each scene (entries beyond n[k] in V dim are
            unused).
        dist_threshold: exact geometric pre-filter threshold
            (2 * (half_diag + margin)).
        half_l, half_w, margin: OBB parameters.

    Returns:
        (pair_hits, pair_earliest)
        pair_hits[(i, j)] : int  number of scenes with any collision
        pair_earliest[(i, j)] : (t_earliest, k_earliest)
    """
    cdef cnp.ndarray[double, ndim=4] pos_np = np.ascontiguousarray(
        positions, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] hed_np = np.ascontiguousarray(
        headings, dtype=np.float64)
    cdef cnp.ndarray[int, ndim=1] n_np = np.ascontiguousarray(
        n_per_scene, dtype=np.intc)
    cdef double[:, :, :, :] pos = pos_np
    cdef double[:, :, :] hed = hed_np
    cdef int[:] nvec = n_np

    cdef int K = pos.shape[0]
    cdef int Vmax = pos.shape[1]
    cdef int Tplus1 = pos.shape[2]

    # OBB scratch
    cdef cnp.ndarray[double, ndim=2] obb_a_np = np.empty((4, 2), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] obb_b_np = np.empty((4, 2), dtype=np.float64)
    cdef double[:, :] obb_a = obb_a_np
    cdef double[:, :] obb_b = obb_b_np

    pair_hits = {}
    pair_earliest = {}

    cdef int k, i, j, t, n_v
    cdef double xi, yi, xj, yj, hi, hj
    cdef double dx, dy, dist
    cdef int first_hit_t

    for k in range(K):
        n_v = nvec[k]
        for i in range(n_v):
            for j in range(i + 1, n_v):
                first_hit_t = -1
                for t in range(Tplus1):
                    xi = pos[k, i, t, 0]
                    yi = pos[k, i, t, 1]
                    xj = pos[k, j, t, 0]
                    yj = pos[k, j, t, 1]
                    dx = xi - xj
                    dy = yi - yj
                    dist = sqrt(dx * dx + dy * dy)
                    if dist > dist_threshold:
                        continue
                    hi = hed[k, i, t]
                    hj = hed[k, j, t]
                    _build_obb(xi, yi, hi, half_l, half_w, margin, obb_a)
                    _build_obb(xj, yj, hj, half_l, half_w, margin, obb_b)
                    if _sat_overlap(obb_a, obb_b):
                        first_hit_t = t
                        break
                if first_hit_t >= 0:
                    key = (i, j)
                    pair_hits[key] = pair_hits.get(key, 0) + 1
                    prev = pair_earliest.get(key)
                    if prev is None or first_hit_t < prev[0]:
                        pair_earliest[key] = (first_hit_t, k)
    return pair_hits, pair_earliest
