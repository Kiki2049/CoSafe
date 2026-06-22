"""Phase 4: Route-aware candidate trajectory selection.

Given M=128 WoTE candidates per flagged ADS vehicle, jointly selects one
trajectory per agent avoiding inter-agent conflicts.

Step A: Hard pruning (CollisionFree, Drivable) + Route Corridor Filtering.
Step B: Pre-compute 4-state pairwise conflict table (INDEPENDENT, COLLISION,
        A_YIELDS, B_YIELDS) with persistent yield rules (established on first
        conflict encounter, cleared when conflict resolves) to prevent
        oscillation, then branch-and-bound optimal joint search.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import carla

from cs_cosafe_c.cs_config import CsRiskConfig, CsSelectorConfig
from cs_cosafe_c.geometry.cs_collision_utils import (
    _build_obb_2d,
    _sat_overlap_2d,
    check_same_timestep_collision,
    determine_yield_aabb,
)

try:
    from cs_cosafe_c.kernels import _bnb as _cy_bnb
    _CY_BNB_AVAILABLE = True
except ImportError:
    _cy_bnb = None
    _CY_BNB_AVAILABLE = False

try:
    from cs_cosafe_c.kernels import _conflict as _cy_conflict
    _CY_CONFLICT_AVAILABLE = True
except ImportError:
    _cy_conflict = None
    _CY_CONFLICT_AVAILABLE = False

try:
    from cs_cosafe_c.kernels import _select_helpers as _cy_sh
    _CY_SH_AVAILABLE = True
except ImportError:
    _cy_sh = None
    _CY_SH_AVAILABLE = False


PAIR_INDEPENDENT = 'independent'
PAIR_COLLISION = 'collision'
PAIR_A_YIELDS = 'a_yields'
PAIR_B_YIELDS = 'b_yields'


@dataclass
class CsSelectionResult:
    """Output of Phase 4 candidate selection."""

    assignments: Dict[int, int]
    trajectories_local: Dict[int, np.ndarray]
    trajectories_world: Dict[int, np.ndarray]
    costs: Dict[int, float]
    priority_order: List[int]
    stop_agents: set = field(default_factory=set)
    reference_paths: Optional[Dict[int, List[Tuple[float, float]]]] = None
    pool_info: Optional[Dict[int, Dict]] = None


def _transform_to_world(candidates_local, vehicle_state):
    """Transform candidates from ego-local to world frame.

    Args:
        candidates_local: (M, T, 3) ego-local [x_fwd, y_lat, heading].
        vehicle_state: CsVehicleState with position_world and heading_rad.

    Returns:
        (M, T, 3) world frame [x_world, y_world, heading_world].
    """
    wx, wy = vehicle_state.position_world
    h = vehicle_state.heading_rad
    cos_h = math.cos(h)
    sin_h = math.sin(h)

    world = np.empty_like(candidates_local)
    world[:, :, 0] = wx + candidates_local[:, :, 0] * cos_h - candidates_local[:, :, 1] * sin_h
    world[:, :, 1] = wy + candidates_local[:, :, 0] * sin_h + candidates_local[:, :, 1] * cos_h
    world[:, :, 2] = h + candidates_local[:, :, 2]
    return world


def _point_to_polyline_dist(px, py, polyline):
    """Minimum distance from point (px, py) to a polyline [(x, y), ...]."""
    min_d = float('inf')
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        abx, aby = bx - ax, by - ay
        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-9:
            d = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
            proj_x = ax + t * abx
            proj_y = ay + t * aby
            d = math.hypot(px - proj_x, py - proj_y)
        if d < min_d:
            min_d = d
    if len(polyline) == 1:
        min_d = math.hypot(px - polyline[0][0], py - polyline[0][1])
    return min_d


def _extend_polyline(polyline, min_length):
    """Extend a polyline so its total length is at least min_length meters.

    If already long enough, returns as-is. Otherwise appends a straight-line
    extension along the last segment's heading.

    Args:
        polyline: [(x, y), ...] at least 2 points.
        min_length: minimum total polyline length in meters.

    Returns:
        Original or extended polyline.
    """
    if len(polyline) < 2:
        return polyline
    total = 0.0
    for i in range(len(polyline) - 1):
        total += math.hypot(polyline[i + 1][0] - polyline[i][0],
                            polyline[i + 1][1] - polyline[i][1])
    if total >= min_length:
        return polyline
    ax, ay = polyline[-2]
    bx, by = polyline[-1]
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return polyline
    ux, uy = dx / d, dy / d
    gap = min_length - total
    ext = (bx + ux * gap, by + uy * gap)
    return list(polyline) + [ext]


def _walk_polyline(polyline, dist):
    """Find a point at given distance along a polyline from its start.

    If dist exceeds polyline length, returns the last point.

    Returns:
        (x, y) tuple.
    """
    remaining = dist
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        if remaining <= seg_len:
            t = remaining / seg_len if seg_len > 1e-9 else 0.0
            return (ax + t * (bx - ax), ay + t * (by - ay))
        remaining -= seg_len
    return polyline[-1]


def _polyline_heading(polyline):
    """Heading direction of the last segment of a polyline (radians)."""
    if len(polyline) < 2:
        return 0.0
    ax, ay = polyline[-2]
    bx, by = polyline[-1]
    return math.atan2(by - ay, bx - ax)


class CsCandidateSelector:
    """Route-aware candidate selector (Phase 4) with branch-and-bound joint optimization."""

    def __init__(self, selector_config: CsSelectorConfig,
                 risk_config: CsRiskConfig):
        self._cfg = selector_config
        self._risk_cfg = risk_config
        self._carla_map = None
        self._yield_to = {}  # type: Dict[int, Set[int]]
        self._last_core_time = 0.0

    @staticmethod
    def _build_brake_trajectory(speed, T, dt=0.05, a_brake=None):
        """Build a braking trajectory in ego-local frame.

        Instead of an instant-stop (all zeros), generates a realistic
        deceleration curve so that collision checks cover the actual
        braking path.

        Args:
            speed: current vehicle speed (m/s).
            T: number of timesteps.
            dt: time per step (s).
            a_brake: deceleration magnitude (m/s²).  Falls back to 3.0.

        Returns:
            (1, T, 3) array [x_fwd, y_lat=0, heading=0].
        """
        if a_brake is None:
            a_brake = 3.0
        traj = np.zeros((1, T, 3), dtype=np.float32)
        v = float(speed)
        x = 0.0
        for t in range(T):
            if v > 0:
                t_stop = v / a_brake
                if t_stop <= dt:
                    x += v * t_stop - 0.5 * a_brake * t_stop * t_stop
                    v = 0.0
                else:
                    x += v * dt - 0.5 * a_brake * dt * dt
                    v -= a_brake * dt
            traj[0, t, 0] = x
        return traj

    def _get_carla_map(self):
        if self._carla_map is None:
            try:
                from srunner.scenariomanager.carla_data_provider import (
                    CarlaDataProvider)
                self._carla_map = CarlaDataProvider.get_map()
            except Exception:
                pass
        return self._carla_map

    def solve(self, wote_results, perception_output,
              risk_assessment, flagged_ads,
              vehicle_targets=None,
              vehicle_reference_paths=None) -> Optional[CsSelectionResult]:
        """Select one trajectory per flagged ADS agent.

        Args:
            wote_results: {agent_id: (candidates, scores, vehicle_state, command)}.
            perception_output: CsPerceptionOutput (for background vehicles).
            risk_assessment: CsRiskAssessment (for priority ordering).
            flagged_ads: Set[int] of ADS agent IDs to plan for.
            vehicle_targets: {agent_id: (x, y)} world-frame target points.
            vehicle_reference_paths: {agent_id: [(x, y), ...]} route waypoints.

        Returns:
            CsSelectionResult or None if no agents to plan for.
        """
        if not flagged_ads or not wote_results:
            return None

        agents = [a for a in flagged_ads if a in wote_results]
        if not agents:
            return None

        import time as _time
        _sel_times = {}
        _t0 = _time.perf_counter()

        self._cleanup_yield_memory(set(agents))

        if vehicle_targets is None:
            vehicle_targets = {}
        if vehicle_reference_paths is None:
            vehicle_reference_paths = {}

        half_l = self._risk_cfg.vehicle_length / 2
        half_w = self._risk_cfg.vehicle_width / 2
        col_margin = self._cfg.bg_collision_margin

        world_candidates = {}
        local_candidates = {}
        stop_index = {}
        vehicle_states = {}
        low_motion = {}
        filter_thresh = self._cfg.snap_static_threshold
        for agent_id in agents:
            cands, scores, vstate, command = wote_results[agent_id]
            vehicle_states[agent_id] = vstate
            T = cands.shape[1]
            excluded = set()
            if filter_thresh > 0:
                for m in range(cands.shape[0]):
                    max_dev = np.max(np.hypot(
                        cands[m, :, 0] - cands[m, 0, 0],
                        cands[m, :, 1] - cands[m, 0, 1]))
                    total_disp = math.hypot(
                        cands[m, -1, 0] - cands[m, 0, 0],
                        cands[m, -1, 1] - cands[m, 0, 1])
                    if max_dev < filter_thresh or total_disp < filter_thresh:
                        excluded.add(m)
            low_motion[agent_id] = excluded
            stop_traj = self._build_brake_trajectory(
                vstate.speed, T, dt=0.05,
                a_brake=self._cfg.brake_decel)
            local_candidates[agent_id] = np.concatenate([cands, stop_traj], axis=0)
            world_candidates[agent_id] = _transform_to_world(
                local_candidates[agent_id], vstate)
            stop_index[agent_id] = cands.shape[0]

        _sel_times['1_prep'] = _time.perf_counter() - _t0

        bg_vehicles = self._extract_background(perception_output, flagged_ads)

        viable = {}
        min_clearance = {}
        _t0 = _time.perf_counter()
        for agent_id in agents:
            v, c = self._prune_collision(
                world_candidates[agent_id],
                bg_vehicles, half_l, half_w, col_margin)
            viable[agent_id] = v
            min_clearance[agent_id] = c
        _sel_times['3a_collision'] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        for agent_id in agents:
            if self._cfg.enable_drivable_check:
                v = self._prune_drivable(world_candidates[agent_id],
                                         viable[agent_id],
                                         stop_index[agent_id])
            else:
                v = viable[agent_id]
            v = [m for m in v if m not in low_motion[agent_id]]
            viable[agent_id] = v
        _sel_times['3b_drivable'] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        viable_go = {}
        non_go = {}
        ref_paths_used = {}
        for agent_id in agents:
            ref_path = vehicle_reference_paths.get(agent_id)
            ref_paths_used[agent_id] = ref_path
            go, rest = self._filter_by_corridor(
                viable[agent_id],
                world_candidates[agent_id],
                local_candidates[agent_id],
                stop_index[agent_id],
                ref_path,
                vehicle_states[agent_id])
            viable_go[agent_id] = go
            non_go[agent_id] = rest
            print("[CoSafe][Corridor] agent={} viable={} go={} non_go={} "
                  "ref_pts={}".format(
                      agent_id, len(viable[agent_id]), len(go), len(rest),
                      len(ref_path) if ref_path else 0))

        _sel_times['4_corridor'] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        go_costs = {}
        pool_size = self._cfg.pool_size
        for agent_id in agents:
            ref_path = ref_paths_used.get(agent_id)
            other_positions = [
                vehicle_states[other_id].position_world
                for other_id in agents if other_id != agent_id
            ]
            if _CY_SH_AVAILABLE:
                costs = _cy_sh.compute_go_cost_batch(
                    list(viable_go[agent_id]),
                    world_candidates[agent_id],
                    min_clearance[agent_id],
                    ref_path,
                    list(self._cfg.corridor_eval_steps),
                    float(self._cfg.corridor_width),
                    other_positions,
                    float(self._cfg.agent_weight),
                    float(self._cfg.ref_weight),
                    float(self._cfg.bg_weight),
                    float(self._cfg.bg_clearance_decay),
                    float(self._cfg.agent_clearance_decay),
                    [5, 10],
                    [3.0, 1.0],
                )
            else:
                costs = {}
                for m in viable_go[agent_id]:
                    costs[m] = self._compute_go_cost(
                        m, world_candidates[agent_id],
                        min_clearance[agent_id], ref_path,
                        other_agent_positions=other_positions)
            if len(costs) > pool_size:
                diverse = self._select_diverse_pool(
                    costs, local_candidates[agent_id], pool_size)
                costs = {m: costs[m] for m in diverse}
                viable_go[agent_id] = diverse
            go_costs[agent_id] = costs

        priority_order = self._sort_by_urgency(agents, risk_assessment)

        all_candidates = {}
        all_costs = {}
        non_go_cost = self._cfg.non_go_cost
        for agent_id in agents:
            pool = list(viable_go[agent_id]) + list(non_go[agent_id])
            seen = set()
            deduped = []
            for m in pool:
                if m not in seen:
                    seen.add(m)
                    deduped.append(m)
            pool = deduped
            if stop_index[agent_id] not in seen:
                pool.append(stop_index[agent_id])
            all_candidates[agent_id] = pool

            agent_costs = dict(go_costs.get(agent_id, {}))
            for m in pool:
                if m not in agent_costs:
                    agent_costs[m] = non_go_cost
            all_costs[agent_id] = agent_costs

            print("[CoSafe][BnB] agent={} pool={} (go={} non_go={})".format(
                agent_id, len(pool), len(viable_go[agent_id]),
                len(non_go.get(agent_id, []))))

        _sel_times['5-7_cost_pool'] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        if self._cfg.selector_mode == 'independent':
            selected = self._independent_select(
                agents, all_candidates, all_costs, stop_index)
            _sel_times['8_conflict'] = 0.0
            _sel_times['9_select'] = _time.perf_counter() - _t0
        else:
            conflict_table = self._build_conflict_table(
                agents, all_candidates, world_candidates, half_l, half_w,
                viable_go, go_costs)
            if self._yield_to:
                yield_str = ", ".join("{}->{}".format(b, sorted(a_set))
                                      for b, a_set in sorted(self._yield_to.items()))
                print("[CoSafe][Yield] rules: {}".format(yield_str))
            _sel_times['8_conflict'] = _time.perf_counter() - _t0
            _t0 = _time.perf_counter()
            selected = self._bnb_select(
                agents, all_candidates, all_costs, conflict_table, stop_index)
            _sel_times['9_select'] = _time.perf_counter() - _t0

        self._last_core_time = (
            _sel_times.get('4_corridor', 0) +
            _sel_times.get('5-7_cost_pool', 0) +
            _sel_times.get('8_conflict', 0) +
            _sel_times.get('9_select', 0))

        parts = " ".join("{}={:.1f}ms".format(k, v * 1000)
                         for k, v in _sel_times.items())
        print("[CoSafe][SelTiming] {} | core={:.1f}ms".format(parts, self._last_core_time * 1000))

        pool_info = {}
        for agent_id in agents:
            pool_info[agent_id] = {
                'pool': list(all_candidates[agent_id]),
                'go': list(viable_go[agent_id]),
                'non_go': list(non_go.get(agent_id, [])),
                'stop_idx': stop_index[agent_id],
                'costs': dict(all_costs.get(agent_id, {})),
                'local_candidates': local_candidates[agent_id],
            }

        assignments = {}
        trajectories_local = {}
        trajectories_world = {}
        costs_out = {}
        _stop_agents = set()
        for agent_id, (chosen_idx, chosen_cost) in selected.items():
            assignments[agent_id] = chosen_idx
            if chosen_idx == stop_index[agent_id]:
                _stop_agents.add(agent_id)
                T = local_candidates[agent_id].shape[1]
                trajectories_local[agent_id] = np.zeros((T, 3), dtype=np.float32)
                wx, wy = vehicle_states[agent_id].position_world
                h = vehicle_states[agent_id].heading_rad
                mpc_world = np.zeros((T, 3), dtype=np.float32)
                mpc_world[:, 0] = wx
                mpc_world[:, 1] = wy
                mpc_world[:, 2] = h
                trajectories_world[agent_id] = mpc_world
            else:
                trajectories_local[agent_id] = local_candidates[agent_id][chosen_idx]
                trajectories_world[agent_id] = world_candidates[agent_id][chosen_idx]
            costs_out[agent_id] = chosen_cost

        return CsSelectionResult(
            assignments=assignments,
            trajectories_local=trajectories_local,
            trajectories_world=trajectories_world,
            costs=costs_out,
            priority_order=priority_order,
            stop_agents=_stop_agents,
            reference_paths=ref_paths_used,
            pool_info=pool_info,
        )


    def _extract_background(self, perception_output, flagged_ads):
        """Extract non-flagged vehicle states from perception output."""
        bg = []
        if perception_output is None or not perception_output.scenes:
            return bg
        scene = perception_output.scenes[-1]
        for v in scene:
            if v.instance_id in flagged_ads:
                continue
            traj = [(v.position_world[0], v.position_world[1])]
            for wp in v.trajectory_world:
                traj.append((wp[0], wp[1]))
            bg.append((v.position_world[0], v.position_world[1],
                        v.heading_rad, traj))
        return bg


    def _prune_collision(self, world_cands, bg_vehicles,
                         half_l, half_w, col_margin):
        """Prune candidates that collide with background vehicles.

        Background trajectories are at frame_dt (0.25s) intervals while
        candidates are at dt_plan (0.05s).  We only check at bg timesteps
        to ensure correct temporal alignment.

        Returns:
            viable: list of candidate indices.
            min_clearance: dict {m: float}.
        """
        N = world_cands.shape[0]
        stop_idx = N - 1
        T_cand = world_cands.shape[1]
        check_limit = min(self._cfg.bg_check_steps, T_cand)

        step_ratio = max(1, int(round(self._risk_cfg.frame_dt / 0.05)))

        viable = []
        min_clearance = {}

        if not bg_vehicles:
            for m in range(N):
                viable.append(m)
                min_clearance[m] = float('inf')
            return viable, min_clearance

        half_diag = math.sqrt(half_l * half_l + half_w * half_w)
        dist_threshold = 2.0 * (half_diag + col_margin)

        for m in range(N):
            collided = False
            closest = float('inf')
            for bx, by, bh, btraj in bg_vehicles:
                n_bg = len(btraj)
                for b in range(n_bg):
                    t = b * step_ratio
                    if t >= check_limit:
                        break
                    cx = world_cands[m, t, 0]
                    cy = world_cands[m, t, 1]
                    ch = world_cands[m, t, 2]
                    bgx, bgy = btraj[b]
                    dist = math.hypot(cx - bgx, cy - bgy)
                    if dist < closest:
                        closest = dist
                    if dist > dist_threshold:
                        continue
                    obb_c = _build_obb_2d(cx, cy, ch, half_l, half_w, col_margin)
                    if b + 1 < n_bg:
                        dx = btraj[b + 1][0] - bgx
                        dy = btraj[b + 1][1] - bgy
                        bg_heading = math.atan2(dy, dx) if dx * dx + dy * dy > 0.01 else bh
                    elif b > 0:
                        dx = bgx - btraj[b - 1][0]
                        dy = bgy - btraj[b - 1][1]
                        bg_heading = math.atan2(dy, dx) if dx * dx + dy * dy > 0.01 else bh
                    else:
                        bg_heading = bh
                    obb_b = _build_obb_2d(bgx, bgy, bg_heading,
                                          half_l, half_w, col_margin)
                    if _sat_overlap_2d(obb_c, obb_b):
                        collided = True
                        break
                if collided:
                    break
            if not collided:
                viable.append(m)
                min_clearance[m] = closest

        if not viable:
            viable.append(stop_idx)
            min_clearance[stop_idx] = 0.0

        return viable, min_clearance


    def _prune_drivable(self, world_cands, viable, stop_idx):
        """Prune candidates that leave drivable road surface.

        Uses CARLA map binary query (project_to_road=False): returns None
        if the point is not within any Driving lane geometry.  No distance
        tolerance needed.

        The stop trajectory (stop_idx) is always kept — stopping at the
        current position is always acceptable.

        Skips the first ``drivable_skip_steps`` timesteps so that a vehicle
        currently off-road can still select candidates that steer back onto
        the road.

        Returns:
            filtered viable list.  If all GO pruned, only stop remains.
        """
        carla_map = self._get_carla_map()
        if carla_map is None:
            return viable

        skip = self._cfg.drivable_skip_steps
        check_steps = min(self._cfg.road_check_steps, world_cands.shape[1])

        result = []
        for m in viable:
            if m == stop_idx:
                result.append(m)
                continue
            on_road = True
            for t in range(skip, check_steps):
                wx = float(world_cands[m, t, 0])
                wy = float(world_cands[m, t, 1])
                try:
                    wp = carla_map.get_waypoint(
                        carla.Location(x=wx, y=wy, z=0.0),
                        project_to_road=False,
                        lane_type=carla.LaneType.Driving)
                except Exception:
                    wp = None
                if wp is None:
                    on_road = False
                    break
            if on_road:
                result.append(m)

        return result


    def _filter_by_corridor(self, viable, world_cands, local_cands,
                            stop_idx, ref_path, vehicle_state):
        """Split viable candidates into GO and non-GO using route corridor.

        Dispatches to the Cython kernel when available; falls back to
        the Python reference implementation otherwise.
        """
        if _CY_SH_AVAILABLE:
            cfg = self._cfg
            rp = _extend_polyline(ref_path, cfg.ref_path_extend_m) \
                if ref_path and len(ref_path) >= 2 else None
            ref_heading = _polyline_heading(rp) \
                if rp and len(rp) >= 2 else float("nan")
            fb_target = _walk_polyline(rp, cfg.fallback_target_dist) \
                if rp and len(rp) >= 2 else None
            return _cy_sh.filter_by_corridor(
                list(viable), world_cands, int(stop_idx),
                rp, tuple(vehicle_state.position_world),
                list(cfg.corridor_eval_steps),
                float(cfg.corridor_width),
                float(cfg.heading_tolerance),
                float(ref_heading),
                float(cfg.fallback_target_dist),
                float(cfg.fallback_angle_tol),
                fb_target,
            )
        return self._filter_by_corridor_python(
            viable, world_cands, local_cands, stop_idx, ref_path,
            vehicle_state)

    def _filter_by_corridor_python(self, viable, world_cands, local_cands,
                                    stop_idx, ref_path, vehicle_state):
        """Python reference implementation; kept as Cython fallback."""
        cfg = self._cfg
        corridor_w = cfg.corridor_width

        go = []
        non_go = []

        if ref_path and len(ref_path) >= 2:
            ref_path = _extend_polyline(ref_path, cfg.ref_path_extend_m)

        ref_heading = None
        if ref_path and len(ref_path) >= 2:
            ref_heading = _polyline_heading(ref_path)

        eval_steps = cfg.corridor_eval_steps
        T_cand = world_cands.shape[1]
        heading_step = min(eval_steps[-1], T_cand - 1) if eval_steps else T_cand - 1

        for m in viable:
            if m == stop_idx:
                non_go.append(m)
                continue

            if not ref_path or len(ref_path) < 2:
                go.append(m)
                continue

            in_corridor = True
            for step in eval_steps:
                t = min(step, T_cand - 1)
                cx = float(world_cands[m, t, 0])
                cy = float(world_cands[m, t, 1])
                d = _point_to_polyline_dist(cx, cy, ref_path)
                if d > corridor_w:
                    in_corridor = False
                    break

            if not in_corridor:
                continue

            if ref_heading is not None:
                cand_heading = float(world_cands[m, heading_step, 2])
                heading_diff = cand_heading - ref_heading
                heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
                if abs(heading_diff) > cfg.heading_tolerance:
                    continue

            go.append(m)

        if not go and ref_path and len(ref_path) >= 2:
            non_stop_viable = [m for m in viable if m != stop_idx]
            if non_stop_viable:
                vx, vy = vehicle_state.position_world
                target = _walk_polyline(ref_path, cfg.fallback_target_dist)
                target_dir = math.atan2(target[1] - vy, target[0] - vx)
                t_1s = min(20, T_cand - 1)
                for m in non_stop_viable:
                    cx = float(world_cands[m, t_1s, 0])
                    cy = float(world_cands[m, t_1s, 1])
                    cand_dir = math.atan2(cy - vy, cx - vx)
                    angle_diff = (cand_dir - target_dir + math.pi) % (2 * math.pi) - math.pi
                    if abs(angle_diff) <= cfg.fallback_angle_tol:
                        go.append(m)

        if stop_idx in viable and stop_idx not in non_go:
            non_go.append(stop_idx)

        return go, non_go


    def _compute_go_cost(self, m, world_cands, min_clearance, ref_path,
                         other_agent_positions=None):
        """Compute cost for a single GO candidate.

        J = w_ref * J_ref + w_bg * J_bg + w_agent * J_agent

        All terms in [0, 1]:
        - J_ref: mean distance to ref polyline / corridor_width
        - J_bg: exp(-decay * clearance_to_background)
        - J_agent: exp(-decay * clearance_to_flagged_agents)
        """
        cfg = self._cfg
        eval_steps = cfg.corridor_eval_steps
        T_max = world_cands.shape[1] - 1

        j_ref = 0.0
        if ref_path and len(ref_path) >= 2:
            total_d = 0.0
            n_pts = 0
            for step in eval_steps:
                t = min(step, T_max)
                cx = float(world_cands[m, t, 0])
                cy = float(world_cands[m, t, 1])
                d = _point_to_polyline_dist(cx, cy, ref_path)
                total_d += d
                n_pts += 1
            if n_pts > 0:
                j_ref = min(total_d / n_pts / cfg.corridor_width, 1.0)

        clearance = min_clearance.get(m, float('inf'))
        j_bg = math.exp(-cfg.bg_clearance_decay * clearance)

        j_agent = 0.0
        if other_agent_positions and cfg.agent_weight > 0:
            near_steps = [5, 10]
            near_weights = [3.0, 1.0]
            w_sum = sum(near_weights)
            for ns, nw in zip(near_steps, near_weights):
                t = min(ns, T_max)
                cx = float(world_cands[m, t, 0])
                cy = float(world_cands[m, t, 1])
                closest = float('inf')
                for ox, oy in other_agent_positions:
                    d = math.hypot(cx - ox, cy - oy)
                    if d < closest:
                        closest = d
                j_agent += nw * math.exp(-cfg.agent_clearance_decay * closest)
            j_agent /= w_sum

        return (cfg.ref_weight * j_ref + cfg.bg_weight * j_bg
                + cfg.agent_weight * j_agent)


    def _select_diverse_pool(self, costs, local_cands, pool_size):
        """Greedy diversity selection from cost-sorted GO candidates.

        1. Sort by cost (lowest first).
        2. Always pick the best-cost candidate.
        3. For each subsequent candidate, accept only if its minimum
           trajectory distance to all already-selected candidates exceeds
           diversity_min_dist.
        4. If not enough, relax threshold by half and scan remaining.

        Args:
            costs: {cand_idx: cost_value}.
            local_cands: (M+1, T, 3) ego-local candidates.
            pool_size: max number of GO candidates to select.

        Returns:
            list of selected candidate indices.
        """
        eval_steps = self._cfg.diversity_eval_steps
        d_min = self._cfg.diversity_min_dist
        T_max = local_cands.shape[1] - 1

        sorted_cands = sorted(costs.keys(), key=lambda m: costs[m])

        eval_pts = {}
        for m in sorted_cands:
            pts = []
            for s in eval_steps:
                t = min(s, T_max)
                pts.append((float(local_cands[m, t, 0]),
                            float(local_cands[m, t, 1])))
            eval_pts[m] = pts

        def traj_dist(m_a, m_b):
            """Multi-point Euclidean distance between two candidates."""
            d_sum = 0.0
            for (ax, ay), (bx, by) in zip(eval_pts[m_a], eval_pts[m_b]):
                d_sum += math.hypot(ax - bx, ay - by)
            return d_sum / len(eval_steps)

        selected = [sorted_cands[0]]
        remaining = sorted_cands[1:]

        thresholds = [d_min * 1.5, d_min, d_min * 0.5]
        for thresh in thresholds:
            if len(selected) >= pool_size:
                break
            next_remaining = []
            for m in remaining:
                if len(selected) >= pool_size:
                    next_remaining.append(m)
                    continue
                min_d = min(traj_dist(m, s) for s in selected)
                if min_d >= thresh:
                    selected.append(m)
                else:
                    next_remaining.append(m)
            remaining = next_remaining

        if len(selected) < pool_size:
            selected_set = set(selected)
            for m in sorted_cands:
                if len(selected) >= pool_size:
                    break
                if m not in selected_set:
                    selected.append(m)
                    selected_set.add(m)

        return selected


    def _sort_by_urgency(self, agents, risk_assessment):
        """Sort agents by risk urgency (most urgent first)."""
        def urgency_key(agent_id):
            earliest_t = float('inf')
            max_severity = 0.0
            if risk_assessment is not None:
                for r in risk_assessment.risks:
                    if r.agent_i == agent_id or r.agent_j == agent_id:
                        if r.time_seconds < earliest_t:
                            earliest_t = r.time_seconds
                        if r.severity > max_severity:
                            max_severity = r.severity
            return (earliest_t, -max_severity, agent_id)

        return sorted(agents, key=urgency_key)


    def _classify_pair(self, traj_a, traj_b, half_l, half_w, margin):
        """Classify the relationship between two candidate trajectories.

        Returns one of PAIR_INDEPENDENT, PAIR_COLLISION, PAIR_A_YIELDS,
        PAIR_B_YIELDS.

        Phase 1: Same-timestep OBB collision with distance pre-filter (exact).
        Phase 2: AABB corridor yield direction in O(T) (conservative approx).
        """
        cfg = self._cfg
        T = min(cfg.conflict_check_steps, traj_a.shape[0], traj_b.shape[0])
        margin_rate = cfg.conflict_margin_rate

        if check_same_timestep_collision(
                traj_a, traj_b, half_l, half_w,
                margin, margin_rate, T) >= 0:
            return PAIR_COLLISION

        result, t_a, t_b = determine_yield_aabb(
            traj_a, traj_b, half_l, half_w,
            margin, margin_rate, T)

        if result == 'independent':
            return PAIR_INDEPENDENT

        gap = abs(t_a - t_b)
        if gap < cfg.yield_min_gap:
            return PAIR_COLLISION

        if result == 'a_yields':
            return PAIR_A_YIELDS
        elif result == 'b_yields':
            return PAIR_B_YIELDS
        else:
            return PAIR_COLLISION


    def _determine_yield_direction(self, traj_a, traj_b, half_l, half_w):
        """Determine yield direction via AABB corridor entry-time comparison.

        Skips Phase 1 (same-timestep collision) because for best GO candidates
        that typically collide at same t, we still need to know who reaches the
        conflict zone first to establish a stable yield rule.

        Returns PAIR_A_YIELDS, PAIR_B_YIELDS, PAIR_COLLISION (tie), or
        PAIR_INDEPENDENT.
        """
        cfg = self._cfg
        T = min(cfg.conflict_check_steps, traj_a.shape[0], traj_b.shape[0])
        margin = self._cfg.conflict_margin
        margin_rate = cfg.conflict_margin_rate

        result, _, _ = determine_yield_aabb(
            traj_a, traj_b, half_l, half_w, margin, margin_rate, T)

        if result == 'a_yields':
            return PAIR_A_YIELDS
        elif result == 'b_yields':
            return PAIR_B_YIELDS
        elif result == 'independent':
            return PAIR_INDEPENDENT
        else:
            return PAIR_COLLISION

    def _build_conflict_table(self, agents, all_candidates,
                              world_candidates, half_l, half_w,
                              viable_go, go_costs):
        """Pre-compute pairwise conflict table with persistent yield rules.

        Dispatches to the Cython kernel when available; falls back to
        the Python reference implementation otherwise.  Both paths
        produce numerically identical conflict_idx and identical
        mutations to self._yield_to (verified by
        cs_cosafe_c/tests/test_conflict_kernel.py).
        """
        if _CY_CONFLICT_AVAILABLE:
            margin = self._cfg.conflict_margin
            (conflict_idx, conflicting_pairs,
             n_checks, n_collisions, n_yields, n_new_rules) = \
                _cy_conflict.build_conflict_table(
                    list(agents), all_candidates, world_candidates,
                    float(half_l), float(half_w),
                    float(margin),
                    float(self._cfg.conflict_margin_rate),
                    int(self._cfg.conflict_check_steps),
                    int(self._cfg.yield_min_gap),
                    self._yield_to, go_costs)
            if self._yield_to:
                yield_str = ", ".join("{}->{}".format(b, sorted(a_set))
                                      for b, a_set
                                      in sorted(self._yield_to.items()))
                print("[CoSafe][Yield] rules: {}".format(yield_str))
            print("[CoSafe][BnB] conflict table: {} checks, {} collisions, "
                  "{} yields, {} new_rules".format(
                      n_checks, n_collisions, n_yields, n_new_rules))
            return conflict_idx
        return self._build_conflict_table_python(
            agents, all_candidates, world_candidates,
            half_l, half_w, viable_go, go_costs)

    def _build_conflict_table_python(self, agents, all_candidates,
                                      world_candidates, half_l, half_w,
                                      viable_go, go_costs):
        """Python reference implementation of conflict-table construction.

        Kept as a fallback when the Cython kernel is unavailable.
        Identical to the original implementation in cs_cosafe/.
        """
        margin = self._cfg.conflict_margin
        conflict_idx = {}
        n_checks = 0
        n_collisions = 0
        n_yields = 0
        n_new_rules = 0

        best_go = {}
        for agent_id in agents:
            gc = go_costs.get(agent_id, {})
            if gc:
                best_go[agent_id] = min(gc, key=gc.get)

        conflicting_pairs = set()

        for i in range(len(agents)):
            ai = agents[i]
            for j in range(i + 1, len(agents)):
                aj = agents[j]

                pair_key = (min(ai, aj), max(ai, aj))
                has_rule = (ai in self._yield_to and aj in self._yield_to[ai]) or \
                           (aj in self._yield_to and ai in self._yield_to[aj])

                if not has_rule and ai in best_go and aj in best_go:
                    direction = self._determine_yield_direction(
                        world_candidates[ai][best_go[ai]],
                        world_candidates[aj][best_go[aj]],
                        half_l, half_w)
                    if direction == PAIR_A_YIELDS:
                        self._yield_to.setdefault(ai, set()).add(aj)
                        n_new_rules += 1
                    elif direction == PAIR_B_YIELDS:
                        self._yield_to.setdefault(aj, set()).add(ai)
                        n_new_rules += 1

                for m in all_candidates[ai]:
                    for n in all_candidates[aj]:
                        n_checks += 1
                        state = self._classify_pair(
                            world_candidates[ai][m],
                            world_candidates[aj][n],
                            half_l, half_w, margin)

                        if state == PAIR_INDEPENDENT:
                            continue

                        if state == PAIR_COLLISION:
                            conflict_idx.setdefault((ai, m), set()).add((aj, n))
                            conflict_idx.setdefault((aj, n), set()).add((ai, m))
                            n_collisions += 1
                            conflicting_pairs.add(pair_key)
                            continue

                        if state == PAIR_A_YIELDS:
                            if aj in self._yield_to.get(ai, set()):
                                n_yields += 1
                                conflicting_pairs.add(pair_key)
                            elif ai in self._yield_to.get(aj, set()):
                                conflict_idx.setdefault((ai, m), set()).add((aj, n))
                                conflict_idx.setdefault((aj, n), set()).add((ai, m))
                                n_collisions += 1
                                conflicting_pairs.add(pair_key)
                            else:
                                conflict_idx.setdefault((ai, m), set()).add((aj, n))
                                conflict_idx.setdefault((aj, n), set()).add((ai, m))
                                n_collisions += 1
                                conflicting_pairs.add(pair_key)
                        elif state == PAIR_B_YIELDS:
                            if ai in self._yield_to.get(aj, set()):
                                n_yields += 1
                                conflicting_pairs.add(pair_key)
                            elif aj in self._yield_to.get(ai, set()):
                                conflict_idx.setdefault((ai, m), set()).add((aj, n))
                                conflict_idx.setdefault((aj, n), set()).add((ai, m))
                                n_collisions += 1
                                conflicting_pairs.add(pair_key)
                            else:
                                conflict_idx.setdefault((ai, m), set()).add((aj, n))
                                conflict_idx.setdefault((aj, n), set()).add((ai, m))
                                n_collisions += 1
                                conflicting_pairs.add(pair_key)

        self._cleanup_stale_yield_rules(agents, conflicting_pairs)

        print("[CoSafe][BnB] conflict table: {} checks, {} collisions, "
              "{} yields, {} new_rules".format(
                  n_checks, n_collisions, n_yields, n_new_rules))
        return conflict_idx


    def _cleanup_yield_memory(self, current_agents):
        """Remove yield entries for agents no longer in the flagged set."""
        stale = [a for a in self._yield_to if a not in current_agents]
        for a in stale:
            del self._yield_to[a]
        for a in list(self._yield_to):
            self._yield_to[a] = self._yield_to[a] & current_agents
            if not self._yield_to[a]:
                del self._yield_to[a]

    def _cleanup_stale_yield_rules(self, agents, conflicting_pairs):
        """Remove yield rules for agent pairs that no longer have conflicts.

        If a pair had a yield rule but this frame shows no conflict between
        them (not in conflicting_pairs), the conflict is resolved and the
        rule should be cleared so it doesn't incorrectly constrain future
        independent encounters.
        """
        for a in list(self._yield_to):
            if a not in self._yield_to:
                continue
            for b in list(self._yield_to.get(a, set())):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in conflicting_pairs:
                    self._yield_to[a].discard(b)
            if not self._yield_to.get(a):
                self._yield_to.pop(a, None)


    def _independent_select(self, agents, all_candidates, all_costs,
                            stop_index):
        """Per-agent argmin selection (ablation: no joint optimization).

        Each agent independently picks min-cost candidate from its pool.
        Hard pruning, corridor filtering, and per-agent cost
        (J_ref + J_bg + J_agent) are all preserved from solve(); only the
        inter-agent conflict resolution and BnB joint search are skipped.
        """
        selected = {}
        for agent_id in agents:
            pool = all_candidates[agent_id]
            costs = all_costs[agent_id]
            if not pool:
                selected[agent_id] = (
                    stop_index[agent_id], self._cfg.non_go_cost)
                continue
            best_idx = min(
                pool,
                key=lambda m: costs.get(m, self._cfg.non_go_cost))
            selected[agent_id] = (
                best_idx, costs.get(best_idx, self._cfg.non_go_cost))
        return selected


    def _bnb_select(self, agents, all_candidates, all_costs,
                    conflict_idx, stop_index):
        """Optimal joint selection via branch-and-bound.

        Dispatches to the compiled Cython kernel when available; falls
        back to the Python implementation otherwise.  Both paths produce
        numerically identical results (verified by the equivalence test
        in cs_cosafe_c/tests/test_bnb_kernel.py).
        """
        if _CY_BNB_AVAILABLE:
            result, best_cost, feasible = _cy_bnb.bnb_select(
                agents, all_candidates, all_costs, conflict_idx, stop_index)
            if feasible:
                print("[CoSafe][BnB] optimal cost={:.4f}".format(best_cost))
            else:
                print("[CoSafe][BnB] no feasible solution, all agents stop")
            return result
        return self._bnb_select_python(
            agents, all_candidates, all_costs, conflict_idx, stop_index)

    def _bnb_select_python(self, agents, all_candidates, all_costs,
                           conflict_idx, stop_index):
        """Python reference implementation of branch-and-bound joint select.

        Kept as a fallback when the Cython kernel is unavailable.
        Identical to the original implementation in cs_cosafe/.
        """
        agent_order = sorted(agents, key=lambda a: len(all_candidates.get(a, [])))
        n = len(agent_order)

        sorted_cands = {}
        for a in agents:
            cand_list = list(all_candidates.get(a, []))
            if stop_index[a] not in cand_list:
                cand_list.append(stop_index[a])
            sorted_cands[a] = sorted(
                cand_list, key=lambda m: all_costs.get(a, {}).get(m, 0.0))

        best = [float('inf')]
        best_assign = [None]

        def dfs(depth, partial, partial_cost):
            if depth == n:
                if partial_cost < best[0]:
                    best[0] = partial_cost
                    best_assign[0] = dict(partial)
                return

            agent = agent_order[depth]
            for m in sorted_cands[agent]:
                c = all_costs.get(agent, {}).get(m, 0.0)
                new_cost = partial_cost + c
                if new_cost >= best[0]:
                    continue
                cset = conflict_idx.get((agent, m))
                if cset:
                    conflicted = False
                    for prev_a, prev_m in partial.items():
                        if (prev_a, prev_m) in cset:
                            conflicted = True
                            break
                    if conflicted:
                        continue
                partial[agent] = m
                dfs(depth + 1, partial, new_cost)
                del partial[agent]

        dfs(0, {}, 0.0)

        if best_assign[0] is not None:
            result = {}
            for a in agents:
                m = best_assign[0][a]
                result[a] = (m, all_costs.get(a, {}).get(m, 0.0))
            print("[CoSafe][BnB] optimal cost={:.4f}".format(best[0]))
            return result

        print("[CoSafe][BnB] no feasible solution, all agents stop")
        result = {}
        for a in agents:
            result[a] = (stop_index[a],
                         all_costs.get(a, {}).get(stop_index[a], self._cfg.non_go_cost))
        return result
