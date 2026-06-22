# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython port of CsCandidateSelector._bnb_select.

DFS over joint candidate assignments with cost-bound and conflict
pruning. Algorithm is byte-identical to the Python original; only the
inner loop is compiled and operates on typed memoryviews instead of
Python dicts.

Public entry point: bnb_select(agents, all_candidates, all_costs,
                               conflict_idx, stop_index)
Returns: {agent_id: (chosen_cand, chosen_cost)} dict, same as the
         original Python function.
"""
import numpy as np
cimport numpy as cnp
cimport cython


# ---------------------------------------------------------------------
# Inner DFS — pure C, no GIL needed but kept with-GIL to allow Python
# refcount safety in the caller's loop (not the bottleneck).
# ---------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _dfs(int depth,
               double partial_cost,
               int n,
               int[:] offsets,
               int[:] n_cands_per_agent,
               double[:] costs_flat,
               unsigned char[:, :] conflict_mat,
               int[:] cur_assign,
               int[:] best_assign,
               double[:] best_cost_box) noexcept nogil:
    cdef int local_idx
    cdef int g_cur, g_prev
    cdef int prev_pos
    cdef double new_cost
    cdef bint conflicted

    if depth == n:
        if partial_cost < best_cost_box[0]:
            best_cost_box[0] = partial_cost
            for prev_pos in range(n):
                best_assign[prev_pos] = cur_assign[prev_pos]
        return

    for local_idx in range(n_cands_per_agent[depth]):
        g_cur = offsets[depth] + local_idx
        new_cost = partial_cost + costs_flat[g_cur]
        # Cost prune (sorted ascending → could break, but match Python's continue)
        if new_cost >= best_cost_box[0]:
            continue
        # Conflict prune
        conflicted = False
        for prev_pos in range(depth):
            g_prev = offsets[prev_pos] + cur_assign[prev_pos]
            if conflict_mat[g_cur, g_prev]:
                conflicted = True
                break
        if conflicted:
            continue
        cur_assign[depth] = local_idx
        _dfs(depth + 1, new_cost, n, offsets, n_cands_per_agent,
             costs_flat, conflict_mat,
             cur_assign, best_assign, best_cost_box)


# ---------------------------------------------------------------------
# Public Python entry point. Signature mirrors the original Python
# _bnb_select. Handles dict→flat-array marshalling and back.
# ---------------------------------------------------------------------
def bnb_select(agents, all_candidates, all_costs, conflict_idx, stop_index):
    """Cython implementation; output equivalent to the Python original.

    Args:
        agents: iterable of agent_ids.
        all_candidates: {agent_id: list of candidate indices}.
        all_costs: {agent_id: {cand_idx: cost}}.
        conflict_idx: {(agent, cand): set of (other_agent, other_cand)}.
        stop_index: {agent_id: stop candidate index}.

    Returns:
        (result_dict, best_cost, feasible)
        result_dict: {agent_id: (chosen_cand_idx, chosen_cost)}.
        best_cost: total cost of returned assignment.
        feasible: True if a strictly cheaper-than-all-stop solution was
                  found; False if returned solution is the all-stop
                  fallback (matches the Python original's two-branch
                  output behavior).
    """
    # ---- 1. Order agents by MRV (fewer candidates first) ----
    agents_list = list(agents)
    # Build the same pool the Python original would use: include stop.
    pool_per_agent_raw = {}
    for a in agents_list:
        cands = list(all_candidates.get(a, []))
        if stop_index[a] not in cands:
            cands.append(stop_index[a])
        pool_per_agent_raw[a] = cands

    agent_order = sorted(agents_list, key=lambda a: len(pool_per_agent_raw[a]))
    cdef int n = len(agent_order)

    # All cdef declarations up-front (Cython requires it).
    cdef int agent_pos, local_idx, off
    cdef int total
    cdef int g_cur, g_other
    cdef double init_cost = 0.0
    cdef double chosen_cost
    cdef int chosen_local

    # ---- 2. Sort each agent's candidates by cost ascending ----
    sorted_cands = []  # list of lists, indexed by agent_pos
    sorted_costs = []
    stop_local_per_agent = []
    for agent_pos in range(n):
        a = agent_order[agent_pos]
        cands = pool_per_agent_raw[a]
        cost_map = all_costs.get(a, {})
        cands_sorted = sorted(cands, key=lambda m: cost_map.get(m, 0.0))
        costs_sorted = [cost_map.get(m, 0.0) for m in cands_sorted]
        sorted_cands.append(cands_sorted)
        sorted_costs.append(costs_sorted)
        stop_local_per_agent.append(cands_sorted.index(stop_index[a]))

    # ---- 3. Build flat arrays ----
    total = sum(len(c) for c in sorted_cands)
    cdef cnp.ndarray[int, ndim=1] n_cands_np = np.empty(n, dtype=np.intc)
    cdef cnp.ndarray[int, ndim=1] offsets_np = np.empty(n, dtype=np.intc)
    cdef cnp.ndarray[double, ndim=1] costs_flat_np = np.empty(total, dtype=np.float64)
    off = 0
    for agent_pos in range(n):
        n_cands_np[agent_pos] = len(sorted_cands[agent_pos])
        offsets_np[agent_pos] = off
        for local_idx in range(n_cands_np[agent_pos]):
            costs_flat_np[off + local_idx] = sorted_costs[agent_pos][local_idx]
        off += n_cands_np[agent_pos]

    # ---- 4. Map (agent_id, cand_idx) → global id, build conflict matrix ----
    # global_id[(agent_id, cand_idx)] = int
    global_id = {}
    for agent_pos in range(n):
        a = agent_order[agent_pos]
        for local_idx in range(n_cands_np[agent_pos]):
            m = sorted_cands[agent_pos][local_idx]
            global_id[(a, m)] = offsets_np[agent_pos] + local_idx

    cdef cnp.ndarray[unsigned char, ndim=2] conflict_mat_np = np.zeros(
        (total, total), dtype=np.uint8)
    for agent_pos in range(n):
        a = agent_order[agent_pos]
        for local_idx in range(n_cands_np[agent_pos]):
            m = sorted_cands[agent_pos][local_idx]
            cset = conflict_idx.get((a, m))
            if not cset:
                continue
            g_cur = offsets_np[agent_pos] + local_idx
            for (other_a, other_m) in cset:
                g_other = global_id.get((other_a, other_m), -1)
                if g_other >= 0:
                    conflict_mat_np[g_cur, g_other] = 1
                    conflict_mat_np[g_other, g_cur] = 1

    # ---- 5. Initial incumbent: all-stop ----
    cdef cnp.ndarray[int, ndim=1] best_assign_np = np.empty(n, dtype=np.intc)
    cdef cnp.ndarray[int, ndim=1] cur_assign_np = np.zeros(n, dtype=np.intc)
    cdef cnp.ndarray[double, ndim=1] best_cost_box = np.empty(1, dtype=np.float64)
    for agent_pos in range(n):
        best_assign_np[agent_pos] = stop_local_per_agent[agent_pos]
        init_cost += costs_flat_np[
            offsets_np[agent_pos] + stop_local_per_agent[agent_pos]]
    best_cost_box[0] = init_cost

    # ---- 6. Run DFS ----
    cdef int[:] offsets_v = offsets_np
    cdef int[:] n_cands_v = n_cands_np
    cdef double[:] costs_flat_v = costs_flat_np
    cdef unsigned char[:, :] conflict_mat_v = conflict_mat_np
    cdef int[:] cur_assign_v = cur_assign_np
    cdef int[:] best_assign_v = best_assign_np
    cdef double[:] best_cost_v = best_cost_box

    with nogil:
        _dfs(0, 0.0, n, offsets_v, n_cands_v, costs_flat_v,
             conflict_mat_v, cur_assign_v, best_assign_v, best_cost_v)

    # ---- 7. Decide feasibility (strict improvement over all-stop) ----
    feasible = best_cost_box[0] < init_cost

    # ---- 8. Marshal results back ----
    result = {}
    for agent_pos in range(n):
        a = agent_order[agent_pos]
        chosen_local = best_assign_np[agent_pos]
        chosen_cand = sorted_cands[agent_pos][chosen_local]
        chosen_cost = sorted_costs[agent_pos][chosen_local]
        result[a] = (chosen_cand, chosen_cost)

    return result, float(best_cost_box[0]), bool(feasible)
