# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython port of CsCandidateSelector._build_conflict_table.

Builds the pairwise 4-state conflict table and updates the persistent
yield-rule dictionary in-place.  Output is a (conflict_idx,
conflicting_pairs, counts) triple matching what the Python original
would produce.  Inlines OBB/SAT/AABB geometric checks to avoid the
per-call Python-marshalling overhead seen when calling the Phase-2
kernel functions through their Python wrappers.

Public entry point:
    build_conflict_table(
        agents,
        all_candidates,
        world_candidates,
        half_l, half_w,
        margin, margin_rate,
        conflict_check_steps, yield_min_gap,
        yield_to,        # in/out dict
        go_costs)
"""
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, cos, sin


# State codes (must match the Python PAIR_* constants ordering only by
# value uniqueness; we map back to strings in the caller via constants).
DEF S_INDEPENDENT = 0
DEF S_COLLISION   = 1
DEF S_A_YIELDS    = 2
DEF S_B_YIELDS    = 3


# ---------------------------------------------------------------------
# Inline geometric helpers — duplicated from _geom.pyx to avoid
# cross-module .pxd setup and per-call overhead.
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


# Phase 1: same-timestep OBB-SAT. Returns first colliding t, or -1.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _phase1_collision(double[:, :] traj_a, double[:, :] traj_b,
                                   double hl, double hw,
                                   double margin, double margin_rate,
                                   int T,
                                   double[:, :] obb_a,
                                   double[:, :] obb_b) noexcept nogil:
    cdef int t
    cdef double m_t, threshold, dx, dy, dist
    cdef double half_diag = sqrt(hl * hl + hw * hw)
    for t in range(T):
        m_t = margin + t * margin_rate
        threshold = 2.0 * (half_diag + m_t)
        dx = traj_a[t, 0] - traj_b[t, 0]
        dy = traj_a[t, 1] - traj_b[t, 1]
        dist = sqrt(dx * dx + dy * dy)
        if dist > threshold:
            continue
        _build_obb(traj_a[t, 0], traj_a[t, 1], traj_a[t, 2], hl, hw, m_t, obb_a)
        _build_obb(traj_b[t, 0], traj_b[t, 1], traj_b[t, 2], hl, hw, m_t, obb_b)
        if _sat_overlap(obb_a, obb_b):
            return t
    return -1


# Phase 2: AABB-corridor yield direction.
# Returns (direction_code, t_enter_a, t_enter_b).
# direction_code: 0=independent, 1=collision-tie, 2=a_yields, 3=b_yields.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _phase2_yield_aabb(double[:, :] traj_a, double[:, :] traj_b,
                                     double hl, double hw,
                                     double margin, double margin_rate,
                                     int T,
                                     int* direction, int* t_a_out,
                                     int* t_b_out) noexcept nogil:
    cdef double max_margin = margin + (T - 1) * margin_rate
    cdef double half_diag = sqrt(
        (hl + max_margin) * (hl + max_margin) +
        (hw + max_margin) * (hw + max_margin))

    cdef double a_xmin = traj_a[0, 0], a_xmax = traj_a[0, 0]
    cdef double a_ymin = traj_a[0, 1], a_ymax = traj_a[0, 1]
    cdef double b_xmin = traj_b[0, 0], b_xmax = traj_b[0, 0]
    cdef double b_ymin = traj_b[0, 1], b_ymax = traj_b[0, 1]
    cdef double v
    cdef int t
    for t in range(1, T):
        v = traj_a[t, 0]
        if v < a_xmin: a_xmin = v
        if v > a_xmax: a_xmax = v
        v = traj_a[t, 1]
        if v < a_ymin: a_ymin = v
        if v > a_ymax: a_ymax = v
        v = traj_b[t, 0]
        if v < b_xmin: b_xmin = v
        if v > b_xmax: b_xmax = v
        v = traj_b[t, 1]
        if v < b_ymin: b_ymin = v
        if v > b_ymax: b_ymax = v

    a_xmin -= half_diag; a_xmax += half_diag
    a_ymin -= half_diag; a_ymax += half_diag
    b_xmin -= half_diag; b_xmax += half_diag
    b_ymin -= half_diag; b_ymax += half_diag

    if a_xmax < b_xmin or b_xmax < a_xmin or \
       a_ymax < b_ymin or b_ymax < a_ymin:
        direction[0] = 0
        t_a_out[0] = -1
        t_b_out[0] = -1
        return

    cdef double ix_min = a_xmin if a_xmin > b_xmin else b_xmin
    cdef double ix_max = a_xmax if a_xmax < b_xmax else b_xmax
    cdef double iy_min = a_ymin if a_ymin > b_ymin else b_ymin
    cdef double iy_max = a_ymax if a_ymax < b_ymax else b_ymax

    cdef int t_a = T
    cdef int t_b = T
    for t in range(T):
        if ix_min <= traj_a[t, 0] <= ix_max and iy_min <= traj_a[t, 1] <= iy_max:
            t_a = t
            break
    for t in range(T):
        if ix_min <= traj_b[t, 0] <= ix_max and iy_min <= traj_b[t, 1] <= iy_max:
            t_b = t
            break
    if t_a == T and t_b == T:
        direction[0] = 0
        t_a_out[0] = -1
        t_b_out[0] = -1
        return
    if t_a < t_b:
        direction[0] = 3  # b yields
    elif t_b < t_a:
        direction[0] = 2  # a yields
    else:
        direction[0] = 1  # collision (tie)
    t_a_out[0] = t_a
    t_b_out[0] = t_b


# Full 4-state classification (used inside the agent-pair inner loop).
# Returns S_INDEPENDENT / S_COLLISION / S_A_YIELDS / S_B_YIELDS.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _classify(double[:, :] traj_a, double[:, :] traj_b,
                           double hl, double hw,
                           double margin, double margin_rate,
                           int T, int yield_min_gap,
                           double[:, :] obb_a,
                           double[:, :] obb_b) noexcept nogil:
    cdef int direction, t_a, t_b, gap
    # Phase 1
    if _phase1_collision(traj_a, traj_b, hl, hw, margin, margin_rate, T,
                          obb_a, obb_b) >= 0:
        return S_COLLISION
    # Phase 2
    _phase2_yield_aabb(traj_a, traj_b, hl, hw, margin, margin_rate, T,
                        &direction, &t_a, &t_b)
    if direction == 0:
        return S_INDEPENDENT
    gap = t_a - t_b
    if gap < 0:
        gap = -gap
    if gap < yield_min_gap:
        return S_COLLISION
    if direction == 2:
        return S_A_YIELDS
    elif direction == 3:
        return S_B_YIELDS
    else:
        return S_COLLISION  # tie


# Yield-direction only (used for first-encounter rule establishment).
# Skips Phase 1; returns S_INDEPENDENT/S_COLLISION/S_A_YIELDS/S_B_YIELDS
# (collision means tie here).
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _yield_direction_only(double[:, :] traj_a, double[:, :] traj_b,
                                       double hl, double hw,
                                       double margin, double margin_rate,
                                       int T) noexcept nogil:
    cdef int direction, t_a, t_b
    _phase2_yield_aabb(traj_a, traj_b, hl, hw, margin, margin_rate, T,
                        &direction, &t_a, &t_b)
    if direction == 2:
        return S_A_YIELDS
    elif direction == 3:
        return S_B_YIELDS
    elif direction == 0:
        return S_INDEPENDENT
    else:
        return S_COLLISION


# ---------------------------------------------------------------------
# Public Python entry point
# ---------------------------------------------------------------------
def build_conflict_table(agents,
                          all_candidates,
                          world_candidates,
                          double half_l,
                          double half_w,
                          double margin,
                          double margin_rate,
                          int conflict_check_steps,
                          int yield_min_gap,
                          yield_to,
                          go_costs):
    """Build conflict table + establish/update yield rules.

    Args:
        agents: list of agent_ids.
        all_candidates: {agent_id: list of candidate indices}.
        world_candidates: {agent_id: (M+1, T, 3) numpy.float64 array}.
        half_l, half_w: OBB half-dimensions.
        margin, margin_rate: time-growing collision margin.
        conflict_check_steps: T parameter (1.5s default).
        yield_min_gap: minimum entry-time gap to allow yield (1.0s default).
        yield_to: in/out dict {yielder_id: set of passer_ids}; mutated.
        go_costs: {agent_id: {cand_idx: cost}} — used to determine best-GO.

    Returns:
        (conflict_idx, conflicting_pairs, n_checks, n_collisions,
         n_yields, n_new_rules)
    """
    cdef int n_agents = len(agents)
    cdef int n_checks = 0
    cdef int n_collisions = 0
    cdef int n_yields = 0
    cdef int n_new_rules = 0

    conflict_idx = {}
    conflicting_pairs = set()

    # Best-GO per agent.
    best_go = {}
    for agent_id in agents:
        gc = go_costs.get(agent_id, {})
        if gc:
            best_go[agent_id] = min(gc, key=gc.get)

    # Pre-convert world_candidates to ensure each entry is a
    # contiguous float64 array; build a lookup of per-(agent, m) memoryviews.
    # Doing this once avoids per-pair conversion inside the hot loop.
    cand_views = {}  # (agent_id, m) -> 2D memoryview of shape (T_cand, 3)
    for agent_id in agents:
        arr = np.ascontiguousarray(world_candidates[agent_id], dtype=np.float64)
        # arr shape: (M+1, T_cand, 3)
        cand_views[agent_id] = arr  # store the 3D array; slice in loop

    # Pre-allocated OBB scratch.
    cdef cnp.ndarray[double, ndim=2] obb_a_np = np.empty((4, 2), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] obb_b_np = np.empty((4, 2), dtype=np.float64)
    cdef double[:, :] obb_a = obb_a_np
    cdef double[:, :] obb_b = obb_b_np

    cdef int i, j, T_local
    cdef int state
    cdef int direction
    cdef double[:, :] traj_a
    cdef double[:, :] traj_b
    cdef int m, n
    cdef bint has_rule_a_to_b, has_rule_b_to_a

    for i in range(n_agents):
        ai = agents[i]
        arr_a = cand_views[ai]
        for j in range(i + 1, n_agents):
            aj = agents[j]
            arr_b = cand_views[aj]
            pair_key = (ai, aj) if ai < aj else (aj, ai)

            # --- Yield rule establishment if first encounter ---
            has_rule_a_to_b = ai in yield_to and aj in yield_to[ai]
            has_rule_b_to_a = aj in yield_to and ai in yield_to[aj]
            if (not (has_rule_a_to_b or has_rule_b_to_a)
                    and ai in best_go and aj in best_go):
                m_best = best_go[ai]
                n_best = best_go[aj]
                # Slice trajectories for the best-GO pair.
                T_local = min(conflict_check_steps,
                              arr_a.shape[1], arr_b.shape[1])
                traj_a = arr_a[m_best]
                traj_b = arr_b[n_best]
                direction = _yield_direction_only(
                    traj_a, traj_b, half_l, half_w,
                    margin, margin_rate, T_local)
                if direction == S_A_YIELDS:
                    yield_to.setdefault(ai, set()).add(aj)
                    n_new_rules += 1
                elif direction == S_B_YIELDS:
                    yield_to.setdefault(aj, set()).add(ai)
                    n_new_rules += 1
                # else: COLLISION (tie) or INDEPENDENT — no rule

            # --- Classify all candidate pairs ---
            T_local = min(conflict_check_steps,
                          arr_a.shape[1], arr_b.shape[1])
            yt_ai = yield_to.get(ai)
            yt_aj = yield_to.get(aj)
            ai_yields_aj = (yt_ai is not None) and (aj in yt_ai)
            aj_yields_ai = (yt_aj is not None) and (ai in yt_aj)

            for m in all_candidates[ai]:
                traj_a = arr_a[m]
                for n in all_candidates[aj]:
                    traj_b = arr_b[n]
                    n_checks += 1
                    state = _classify(
                        traj_a, traj_b, half_l, half_w,
                        margin, margin_rate, T_local, yield_min_gap,
                        obb_a, obb_b)

                    if state == S_INDEPENDENT:
                        continue
                    if state == S_COLLISION:
                        _add_conflict(conflict_idx, ai, m, aj, n)
                        n_collisions += 1
                        conflicting_pairs.add(pair_key)
                        continue
                    if state == S_A_YIELDS:
                        if ai_yields_aj:
                            n_yields += 1
                            conflicting_pairs.add(pair_key)
                        elif aj_yields_ai:
                            _add_conflict(conflict_idx, ai, m, aj, n)
                            n_collisions += 1
                            conflicting_pairs.add(pair_key)
                        else:
                            _add_conflict(conflict_idx, ai, m, aj, n)
                            n_collisions += 1
                            conflicting_pairs.add(pair_key)
                    elif state == S_B_YIELDS:
                        if aj_yields_ai:
                            n_yields += 1
                            conflicting_pairs.add(pair_key)
                        elif ai_yields_aj:
                            _add_conflict(conflict_idx, ai, m, aj, n)
                            n_collisions += 1
                            conflicting_pairs.add(pair_key)
                        else:
                            _add_conflict(conflict_idx, ai, m, aj, n)
                            n_collisions += 1
                            conflicting_pairs.add(pair_key)

    # --- Cleanup stale yield rules for pairs no longer in conflict ---
    # Mirrors CsCandidateSelector._cleanup_stale_yield_rules.
    for a in list(yield_to):
        if a not in yield_to:
            continue
        for b in list(yield_to.get(a, ())):
            pair_key_ab = (a, b) if a < b else (b, a)
            if pair_key_ab not in conflicting_pairs:
                yield_to[a].discard(b)
        if not yield_to.get(a):
            yield_to.pop(a, None)

    return (conflict_idx, conflicting_pairs,
            n_checks, n_collisions, n_yields, n_new_rules)


cdef inline void _add_conflict(dict conflict_idx, ai, int m, aj, int n):
    """Symmetric insert into the conflict_idx dict."""
    key_a = (ai, m)
    key_b = (aj, n)
    s = conflict_idx.get(key_a)
    if s is None:
        s = set()
        conflict_idx[key_a] = s
    s.add(key_b)
    s = conflict_idx.get(key_b)
    if s is None:
        s = set()
        conflict_idx[key_b] = s
    s.add(key_a)
