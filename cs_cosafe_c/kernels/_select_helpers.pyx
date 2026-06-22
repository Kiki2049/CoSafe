# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython kernels for corridor filtering + GO cost computation.

These two functions together cover the 4_corridor + 5-7_cost_pool
sub-steps of Phase 4 BnB selection (~480 ms in the Python prototype
on dense scenes).

Public entry points:
    point_to_polyline_dist(px, py, polyline_array)  -- exposed for tests
    filter_by_corridor(...)
    compute_go_cost_batch(...)
"""
import math

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, hypot, exp, atan2, fabs, M_PI


# ---------------------------------------------------------------------
# Inline polyline distance (used by both corridor filter and J_ref).
# ---------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _point_to_polyline(double px, double py,
                                       double[:, :] poly,
                                       int n_pts) noexcept nogil:
    """Min distance from (px, py) to polyline of n_pts (x, y) points."""
    cdef double min_d, ax, ay, bx, by, abx, aby, ab_sq, t
    cdef double proj_x, proj_y, dx, dy, d
    cdef int i
    if n_pts <= 0:
        return 0.0
    if n_pts == 1:
        dx = px - poly[0, 0]
        dy = py - poly[0, 1]
        return sqrt(dx * dx + dy * dy)
    min_d = 1.0e300
    for i in range(n_pts - 1):
        ax = poly[i, 0]; ay = poly[i, 1]
        bx = poly[i + 1, 0]; by = poly[i + 1, 1]
        abx = bx - ax; aby = by - ay
        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-9:
            dx = px - ax; dy = py - ay
            d = sqrt(dx * dx + dy * dy)
        else:
            t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
            if t < 0.0: t = 0.0
            elif t > 1.0: t = 1.0
            proj_x = ax + t * abx
            proj_y = ay + t * aby
            dx = px - proj_x; dy = py - proj_y
            d = sqrt(dx * dx + dy * dy)
        if d < min_d:
            min_d = d
    return min_d


def point_to_polyline_dist(double px, double py, polyline):
    """Python-callable wrapper for the polyline distance kernel."""
    if not polyline:
        return float("inf")
    cdef cnp.ndarray[double, ndim=2] arr = np.asarray(
        polyline, dtype=np.float64).reshape(-1, 2)
    cdef double[:, :] v = arr
    return float(_point_to_polyline(px, py, v, arr.shape[0]))


# ---------------------------------------------------------------------
# Corridor filter (primary + fallback)
# ---------------------------------------------------------------------
def filter_by_corridor(viable,
                        world_cands,
                        int stop_idx,
                        ref_path,         # None or [(x,y), ...] (already extended)
                        vehicle_pos,      # (vx, vy)
                        eval_steps,
                        double corridor_width,
                        double heading_tolerance,
                        double ref_heading,   # NaN if no ref_path
                        double fallback_target_dist,
                        double fallback_angle_tol,
                        fallback_target_xy,   # (tx, ty) or None — caller computes
                        ):
    """Return (go_list, non_go_list).

    Mirrors CsCandidateSelector._filter_by_corridor.  All polyline
    pre-processing (extension, ref_heading, fallback target point)
    must be done by the caller because they're tiny ops dominated by
    Python overhead anyway; this kernel handles the O(N_cand * eval_steps
    * N_poly) inner loop.
    """
    cdef int T_cand
    cdef int m
    cdef int step
    cdef int t
    cdef int t_1s
    cdef double cx, cy
    cdef double d
    cdef bint in_corridor
    cdef double cand_heading
    cdef double heading_diff
    cdef double vx_f, vy_f
    cdef bint has_ref = ref_path is not None and len(ref_path) >= 2

    go = []
    non_go = []

    cdef cnp.ndarray[double, ndim=3] wc_np
    wc_np = np.ascontiguousarray(world_cands, dtype=np.float64)
    cdef double[:, :, :] wc = wc_np
    T_cand = wc.shape[1]

    cdef cnp.ndarray[double, ndim=2] poly_np
    cdef double[:, :] poly_v
    cdef int n_poly
    if has_ref:
        poly_np = np.asarray(ref_path, dtype=np.float64).reshape(-1, 2)
        poly_v = poly_np
        n_poly = poly_np.shape[0]
    else:
        poly_v = np.zeros((1, 2), dtype=np.float64)
        n_poly = 0

    cdef cnp.ndarray[int, ndim=1] es_np = np.asarray(
        eval_steps, dtype=np.intc) if eval_steps else np.empty(0, dtype=np.intc)
    cdef int[:] es = es_np
    cdef int n_eval = es_np.shape[0]
    cdef int heading_step
    if n_eval > 0:
        heading_step = es_np[n_eval - 1]
        if heading_step >= T_cand:
            heading_step = T_cand - 1
    else:
        heading_step = T_cand - 1

    cdef bint use_heading = not (ref_heading != ref_heading)  # NaN check

    for m in viable:
        if m == stop_idx:
            non_go.append(m)
            continue
        if not has_ref:
            go.append(m)
            continue

        # Corridor check at eval_steps
        in_corridor = True
        for ii in range(n_eval):
            step = es[ii]
            t = step if step < T_cand else (T_cand - 1)
            cx = wc[m, t, 0]
            cy = wc[m, t, 1]
            d = _point_to_polyline(cx, cy, poly_v, n_poly)
            if d > corridor_width:
                in_corridor = False
                break
        if not in_corridor:
            continue

        # Heading tolerance at last eval step
        if use_heading:
            cand_heading = wc[m, heading_step, 2]
            heading_diff = cand_heading - ref_heading
            # wrap to (-pi, pi]
            heading_diff = (heading_diff + M_PI) - \
                (((heading_diff + M_PI) // (2.0 * M_PI)) * 2.0 * M_PI) - M_PI
            if heading_diff > M_PI:
                heading_diff -= 2.0 * M_PI
            elif heading_diff <= -M_PI:
                heading_diff += 2.0 * M_PI
            if fabs(heading_diff) > heading_tolerance:
                continue

        go.append(m)

    # Fallback: target-direction check if all GO pruned
    if not go and has_ref and fallback_target_xy is not None:
        vx_f, vy_f = vehicle_pos
        tx, ty = fallback_target_xy
        target_dir = atan2(ty - vy_f, tx - vx_f)
        t_1s = 20 if 20 < T_cand else (T_cand - 1)
        for m in viable:
            if m == stop_idx:
                continue
            cx = wc[m, t_1s, 0]
            cy = wc[m, t_1s, 1]
            cand_dir = atan2(cy - vy_f, cx - vx_f)
            ad = cand_dir - target_dir
            # wrap to (-pi, pi]
            while ad > M_PI: ad -= 2.0 * M_PI
            while ad <= -M_PI: ad += 2.0 * M_PI
            if fabs(ad) <= fallback_angle_tol:
                go.append(m)

    # Ensure stop is always in non_go
    if stop_idx in viable and stop_idx not in non_go:
        non_go.append(stop_idx)

    return go, non_go


# ---------------------------------------------------------------------
# GO cost computation (batched across candidates)
# ---------------------------------------------------------------------
def compute_go_cost_batch(viable_go,
                           world_cands,
                           min_clearance,         # {m: float} dict
                           ref_path,              # [(x,y), ...] or None
                           eval_steps,            # [int]
                           double corridor_width,
                           other_agent_positions, # list of (x, y)
                           double agent_weight,
                           double ref_weight,
                           double bg_weight,
                           double bg_clearance_decay,
                           double agent_clearance_decay,
                           near_steps,            # [int]
                           near_weights,          # [float]
                           ):
    """Returns {m: cost} dict matching CsCandidateSelector._compute_go_cost
    semantics applied across all candidates in viable_go."""
    cdef cnp.ndarray[double, ndim=3] wc_np
    wc_np = np.ascontiguousarray(world_cands, dtype=np.float64)
    cdef double[:, :, :] wc = wc_np
    cdef int T_cand = wc.shape[1]
    cdef int T_max = T_cand - 1
    cdef bint has_ref = ref_path is not None and len(ref_path) >= 2

    cdef cnp.ndarray[double, ndim=2] poly_np
    cdef double[:, :] poly_v
    cdef int n_poly
    if has_ref:
        poly_np = np.asarray(ref_path, dtype=np.float64).reshape(-1, 2)
        poly_v = poly_np
        n_poly = poly_np.shape[0]
    else:
        poly_v = np.zeros((1, 2), dtype=np.float64)
        n_poly = 0

    cdef cnp.ndarray[int, ndim=1] es_np = np.asarray(
        eval_steps, dtype=np.intc) if eval_steps else np.empty(0, dtype=np.intc)
    cdef int[:] es = es_np
    cdef int n_eval = es_np.shape[0]

    cdef cnp.ndarray[int, ndim=1] ns_np = np.asarray(
        near_steps, dtype=np.intc) if near_steps else np.empty(0, dtype=np.intc)
    cdef cnp.ndarray[double, ndim=1] nw_np = np.asarray(
        near_weights, dtype=np.float64) if near_weights else \
        np.empty(0, dtype=np.float64)
    cdef int[:] ns = ns_np
    cdef double[:] nw = nw_np
    cdef int n_near = ns_np.shape[0]
    cdef double w_sum = 0.0
    cdef int ii
    for ii in range(n_near):
        w_sum += nw[ii]

    cdef int n_other
    cdef cnp.ndarray[double, ndim=2] others_np
    cdef double[:, :] others_v
    if other_agent_positions and agent_weight > 0 and n_near > 0:
        others_np = np.asarray(
            other_agent_positions, dtype=np.float64).reshape(-1, 2)
        others_v = others_np
        n_other = others_np.shape[0]
    else:
        others_v = np.zeros((1, 2), dtype=np.float64)
        n_other = 0

    result = {}
    cdef int m, step, t, j
    cdef double total_d, d, cx, cy
    cdef double j_ref, j_bg, j_agent, closest, ox, oy
    cdef double clearance

    for m in viable_go:
        # J_ref
        j_ref = 0.0
        if has_ref and n_eval > 0:
            total_d = 0.0
            for ii in range(n_eval):
                step = es[ii]
                t = step if step < T_cand else T_max
                cx = wc[m, t, 0]; cy = wc[m, t, 1]
                total_d += _point_to_polyline(cx, cy, poly_v, n_poly)
            j_ref = total_d / n_eval / corridor_width
            if j_ref > 1.0: j_ref = 1.0

        # J_bg
        clearance = min_clearance.get(m, float("inf"))
        if clearance == float("inf"):
            j_bg = 0.0
        else:
            j_bg = exp(-bg_clearance_decay * clearance)

        # J_agent
        j_agent = 0.0
        if n_other > 0 and n_near > 0:
            for ii in range(n_near):
                t = ns[ii] if ns[ii] < T_cand else T_max
                cx = wc[m, t, 0]; cy = wc[m, t, 1]
                closest = 1.0e300
                for j in range(n_other):
                    ox = others_v[j, 0]; oy = others_v[j, 1]
                    d = hypot(cx - ox, cy - oy)
                    if d < closest: closest = d
                j_agent += nw[ii] * exp(-agent_clearance_decay * closest)
            j_agent /= w_sum

        result[m] = ref_weight * j_ref + bg_weight * j_bg + agent_weight * j_agent

    return result
