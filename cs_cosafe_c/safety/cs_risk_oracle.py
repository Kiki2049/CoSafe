"""Phase 2 risk assessment oracles.

Consumes CsPerceptionOutput (GT or K=6 VAE predictions) and detects:
- Collision risks via 2D SAT (Separating Axis Theorem)
- Traffic rule violations (red light only)
- Deadlock via spatial conflict graph + wait-for graph cycle detection
- Single stuck via isolated stall detection
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from cs_cosafe_c.cs_config import CsRiskConfig
from cs_cosafe_c.geometry.cs_collision_utils import (
    _build_obb_2d,
    _build_trajectory_corridor,
    _infer_heading_t0,
    _sat_overlap_2d,
)


@dataclass
class CsRiskEvent:
    """A single detected risk."""

    risk_type: str
    severity: float
    agent_i: int
    agent_j: int
    timestep: int
    time_seconds: float
    details: Dict = field(default_factory=dict)


@dataclass
class CsRiskAssessment:
    """Collection of all detected risks for one perception step."""

    risks: List[CsRiskEvent]
    step: int
    perception_mode: str
    num_scenes: int
    flagged_agents: Set[int] = field(default_factory=set)


def _get_agent_id(vehicle, idx: int) -> int:
    """Get a unique agent identifier for a vehicle.

    In GT mode instance_id=-1 for all vehicles, so use list index.
    In pred mode use instance_id.
    """
    if vehicle.instance_id >= 0:
        return vehicle.instance_id
    return idx


try:
    from cs_cosafe_c.kernels import _risk as _cy_risk
    _CY_RISK_AVAILABLE = True
except ImportError:
    _cy_risk = None
    _CY_RISK_AVAILABLE = False


class CsCollisionOracle:
    """Detects pairwise collision risks via 2D OBB overlap (SAT).

    Temporal collision: checks OBB overlap at the SAME timestep t.
    P_col(i,j) = (1/K) * sum_k max_t indicator[overlap].
    """

    def __init__(self, config: CsRiskConfig):
        self._cfg = config
        self._half_l = config.vehicle_length / 2.0
        self._half_w = config.vehicle_width / 2.0
        self._margin = config.collision_margin
        self._dt = config.frame_dt
        self._dist_threshold = 2.0 * (
            math.sqrt(self._half_l ** 2 + self._half_w ** 2) + self._margin)

    def assess(self, perception) -> List[CsRiskEvent]:
        scenes = perception.scenes
        K = len(scenes)
        if K == 0:
            return []

        future_frames = 0
        for scene in scenes:
            for v in scene:
                future_frames = max(future_frames, len(v.trajectory_world))
                break
            if future_frames > 0:
                break

        if _CY_RISK_AVAILABLE:
            return self._assess_cython(scenes, K, future_frames)
        return self._assess_python(scenes, K, future_frames)

    def _assess_cython(self, scenes, K, future_frames):
        """Pre-extract typed arrays and dispatch to the C kernel."""
        import numpy as np

        T_plus_1 = future_frames + 1
        V_max = max((len(s) for s in scenes), default=0)
        if V_max == 0:
            return []

        positions = np.zeros((K, V_max, T_plus_1, 2), dtype=np.float64)
        headings = np.zeros((K, V_max, T_plus_1), dtype=np.float64)
        n_per_scene = np.zeros(K, dtype=np.intc)

        agent_ids = []
        ref_scene = scenes[0]
        for i, v in enumerate(ref_scene):
            agent_ids.append(_get_agent_id(v, i))

        for k, scene in enumerate(scenes):
            n_per_scene[k] = len(scene)
            for i, v in enumerate(scene):
                for t in range(T_plus_1):
                    xy = self._pos_at_t(v, t)
                    positions[k, i, t, 0] = xy[0]
                    positions[k, i, t, 1] = xy[1]
                    headings[k, i, t] = self._heading_at_t(v, t)

        pair_hits, pair_earliest = _cy_risk.collision_pair_scan(
            positions, headings, n_per_scene,
            float(self._dist_threshold),
            float(self._half_l), float(self._half_w),
            float(self._margin))

        risks = []
        for (i, j), hits in pair_hits.items():
            p_col = hits / K
            t_earliest, _ = pair_earliest[(i, j)]
            risks.append(CsRiskEvent(
                risk_type='collision',
                severity=p_col,
                agent_i=agent_ids[i],
                agent_j=agent_ids[j],
                timestep=t_earliest,
                time_seconds=t_earliest * self._dt,
                details={'p_col': p_col, 'scenes_with_collision': hits},
            ))
        return risks

    def _assess_python(self, scenes, K, future_frames):
        """Python reference (preserved as Cython fallback)."""
        pair_hits = {}
        pair_earliest = {}

        for k, scene in enumerate(scenes):
            n = len(scene)
            for i in range(n):
                for j in range(i + 1, n):
                    vi, vj = scene[i], scene[j]
                    aid_i = _get_agent_id(vi, i)
                    aid_j = _get_agent_id(vj, j)
                    pair_key = (aid_i, aid_j)

                    for t in range(future_frames + 1):
                        xi, yi = self._pos_at_t(vi, t)
                        xj, yj = self._pos_at_t(vj, t)

                        if math.hypot(xi - xj, yi - yj) > self._dist_threshold:
                            continue

                        hi = self._heading_at_t(vi, t)
                        hj = self._heading_at_t(vj, t)

                        ci = _build_obb_2d(
                            xi, yi, hi, self._half_l, self._half_w,
                            self._margin)
                        cj = _build_obb_2d(
                            xj, yj, hj, self._half_l, self._half_w,
                            self._margin)

                        if _sat_overlap_2d(ci, cj):
                            pair_hits[pair_key] = pair_hits.get(
                                pair_key, 0) + 1
                            if (pair_key not in pair_earliest
                                    or t < pair_earliest[pair_key][0]):
                                pair_earliest[pair_key] = (t, k)
                            break

        risks = []
        for (aid_i, aid_j), hits in pair_hits.items():
            p_col = hits / K
            t_earliest, _ = pair_earliest[(aid_i, aid_j)]
            risks.append(CsRiskEvent(
                risk_type='collision',
                severity=p_col,
                agent_i=aid_i,
                agent_j=aid_j,
                timestep=t_earliest,
                time_seconds=t_earliest * self._dt,
                details={'p_col': p_col, 'scenes_with_collision': hits},
            ))
        return risks

    @staticmethod
    def _pos_at_t(vehicle, t: int) -> Tuple[float, float]:
        if t == 0:
            return vehicle.position_world
        idx = t - 1
        if idx < len(vehicle.trajectory_world):
            return vehicle.trajectory_world[idx]
        if vehicle.trajectory_world:
            return vehicle.trajectory_world[-1]
        return vehicle.position_world

    @staticmethod
    def _heading_at_t(vehicle, t: int) -> float:
        if t == 0:
            return _infer_heading_t0(vehicle)
        pos_prev = CsCollisionOracle._pos_at_t(vehicle, t - 1)
        pos_curr = CsCollisionOracle._pos_at_t(vehicle, t)
        dx = pos_curr[0] - pos_prev[0]
        dy = pos_curr[1] - pos_prev[1]
        if dx * dx + dy * dy > 0.01:
            return math.atan2(dy, dx)
        return vehicle.heading_rad


class CsTrafficRuleOracle:
    """Detects predictive red-light violations via trajectory-stopline intersection."""

    def __init__(self, config: CsRiskConfig):
        self._cfg = config
        self._dt = config.frame_dt
        self._traffic_lights = []
        self._setup_done = False

    def setup(self):
        """Cache traffic light actors and their stop line waypoints."""
        try:
            import carla
            from srunner.scenariomanager.carla_data_provider import (
                CarlaDataProvider)
            world = CarlaDataProvider.get_world()

            lights = world.get_actors().filter('*traffic_light*')
            for light in lights:
                trigger = light.trigger_volume
                light_loc = light.get_location()
                trigger_loc = carla.Location(
                    trigger.location.x, trigger.location.y,
                    trigger.location.z)
                light.get_transform().transform(trigger_loc)

                yaw = math.radians(light.get_transform().rotation.yaw)
                perp_x = -math.sin(yaw)
                perp_y = math.cos(yaw)
                half_width = max(trigger.extent.y, 3.0)

                stop_start = (
                    trigger_loc.x + perp_x * half_width,
                    trigger_loc.y + perp_y * half_width,
                )
                stop_end = (
                    trigger_loc.x - perp_x * half_width,
                    trigger_loc.y - perp_y * half_width,
                )

                self._traffic_lights.append({
                    'actor': light,
                    'location': (light_loc.x, light_loc.y),
                    'stop_start': stop_start,
                    'stop_end': stop_end,
                })

            self._setup_done = True
            print("[CoSafe Risk] Cached {} traffic lights".format(
                len(self._traffic_lights)))
        except Exception as e:
            print("[CoSafe Risk] Traffic light setup failed: {}".format(e))
            self._setup_done = True

    def get_traffic_lights(self):
        """Public accessor for cached traffic light data."""
        return self._traffic_lights

    def assess(self, perception) -> List[CsRiskEvent]:
        """Check if any vehicle's trajectory crosses a red-light stop line.

        Conservative: if ANY scene shows a violation, the agent is flagged.
        """
        if not self._setup_done or not self._traffic_lights:
            return []

        flagged = {}
        for scene in perception.scenes:
            for idx, vehicle in enumerate(scene):
                aid = _get_agent_id(vehicle, idx)
                if aid in flagged:
                    continue
                risk = self._check_red_light_violation(vehicle, aid)
                if risk is not None:
                    flagged[aid] = risk

        return list(flagged.values())

    def _check_red_light_violation(self, vehicle, agent_id
                                   ) -> Optional[CsRiskEvent]:
        """Check if vehicle trajectory crosses any red-light stop line."""
        import carla

        points = [vehicle.position_world] + list(vehicle.trajectory_world)

        for tl_info in self._traffic_lights:
            light = tl_info['actor']

            dx = vehicle.position_world[0] - tl_info['location'][0]
            dy = vehicle.position_world[1] - tl_info['location'][1]
            if dx * dx + dy * dy > self._cfg.traffic_light_range ** 2:
                continue

            try:
                if light.get_state() != carla.TrafficLightState.Red:
                    continue
            except Exception:
                continue

            for t in range(len(points) - 1):
                seg_dx = points[t + 1][0] - points[t][0]
                seg_dy = points[t + 1][1] - points[t][1]
                seg_speed = math.sqrt(seg_dx ** 2 + seg_dy ** 2) / self._dt
                if seg_speed < self._cfg.v_creep:
                    continue

                if self._segments_intersect(
                    points[t], points[t + 1],
                    tl_info['stop_start'], tl_info['stop_end'],
                ):
                    return CsRiskEvent(
                        risk_type='red_light',
                        severity=1.0,
                        agent_i=agent_id,
                        agent_j=-1,
                        timestep=t,
                        time_seconds=t * self._dt,
                        details={
                            'traffic_light_id': light.id,
                            'speed_at_crossing': seg_speed,
                        },
                    )
        return None

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        """Test if segment (p1,p2) intersects (p3,p4) via cross products."""
        def cross(o, a, b):
            return ((a[0] - o[0]) * (b[1] - o[1])
                    - (a[1] - o[1]) * (b[0] - o[0]))

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False


class CsDeadlockOracle:
    """Detects deadlocks via spatial conflict graph + stall + wait-for cycles.

    Internal state:
    - _history: per-agent speed history for stall detection
    - _spatial_edges: directed spatial conflict edges (A,B) -> timestamp
    - _active_deadlocks: persistent deadlock groups (survive edge expiry)

    Flow:
    1. Update spatial conflict graph (from trajectory corridors, non-symmetric)
    2. Red-light queue exclusion (forward path crosses red stop line → skip)
    3. Stall detection (w_stall window, rho_stall fraction, excludes red-light)
    4. Build wait-for graph (from spatial edges + stalled set)
    5. Persistent deadlock: new cycles added, existing groups re-evaluated
       when a member exits stall
    6. Isolated stalled agents (no edges) → single_stuck
    """

    def __init__(self, config: CsRiskConfig):
        self._cfg = config
        self._dt = config.frame_dt
        self._half_l = config.vehicle_length / 2.0
        self._half_w = config.vehicle_width / 2.0
        self._margin = config.collision_margin
        self._history: Dict[int, deque] = {}
        self._max_history = int(
            max(config.w_stall, config.w_conflict) / config.frame_dt) + 1
        self._spatial_edges: Dict[Tuple[int, int], float] = {}
        self._active_deadlocks: List[Set[int]] = []
        self._current_time: float = 0.0

    def assess(self, perception, traffic_oracle: CsTrafficRuleOracle,
               step: int) -> List[CsRiskEvent]:
        scene = perception.scenes[-1]
        self._current_time = step * self._dt

        vehicles = {}
        for idx, v in enumerate(scene):
            vehicles[_get_agent_id(v, idx)] = v

        self._update_spatial_conflicts(vehicles)

        red_light_queuing = self._find_red_light_queuing(
            vehicles, traffic_oracle)

        self._update_history(vehicles, step)
        stalled, stall_durations = self._detect_stalled(
            vehicles, red_light_queuing)

        wait_for = self._build_wait_for(stalled)

        risks = self._update_deadlocks(
            wait_for, stalled, stall_durations)

        deadlocked = set()
        for group in self._active_deadlocks:
            deadlocked.update(group)

        agents_in_graph = set(wait_for.keys())

        for aid in stalled:
            if aid in deadlocked or aid in agents_in_graph:
                continue
            risks.append(CsRiskEvent(
                risk_type='single_stuck',
                severity=1.0,
                agent_i=aid,
                agent_j=-1,
                timestep=-1,
                time_seconds=stall_durations.get(aid, 0),
                details={'stall_duration': stall_durations.get(aid, 0)},
            ))

        return risks


    def _update_spatial_conflicts(self, vehicles):
        """Compute directed spatial conflict edges from trajectory corridors.

        Two vehicles A and B have a spatial conflict if their trajectory
        corridors overlap — i.e. their intended paths intersect in space.
        Both directed edges A->B and B->A are added when any OBB from
        A's corridor overlaps any OBB from B's corridor.
        """
        corridors = {}
        for aid, v in vehicles.items():
            corridors[aid] = _build_trajectory_corridor(
                v, self._half_l, self._half_w, self._margin,
                self._cfg.d_proj)

        aids = list(corridors.keys())
        for i_idx in range(len(aids)):
            for j_idx in range(i_idx + 1, len(aids)):
                a, b = aids[i_idx], aids[j_idx]
                conflict = False
                for a_obb in corridors[a]:
                    for b_obb in corridors[b]:
                        if _sat_overlap_2d(a_obb, b_obb):
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    self._spatial_edges[(a, b)] = self._current_time
                    self._spatial_edges[(b, a)] = self._current_time

        expired = [k for k, t in self._spatial_edges.items()
                   if self._current_time - t > self._cfg.w_conflict]
        for k in expired:
            del self._spatial_edges[k]


    def _find_red_light_queuing(self, vehicles, traffic_oracle):
        """Find vehicles whose forward path crosses a currently-red stop line."""
        red_light_queuing = set()
        if not traffic_oracle._setup_done:
            return red_light_queuing

        try:
            import carla as carla_mod
        except ImportError:
            return red_light_queuing

        for aid, v in vehicles.items():
            if self._is_red_light_queuing(v, traffic_oracle, carla_mod):
                red_light_queuing.add(aid)
        return red_light_queuing

    def _is_red_light_queuing(self, vehicle, traffic_oracle, carla_mod):
        """Check if vehicle's forward projection crosses a red stop line."""
        heading = _infer_heading_t0(vehicle)
        x, y = vehicle.position_world
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        end_x = x + cos_h * self._cfg.d_proj
        end_y = y + sin_h * self._cfg.d_proj

        for tl_info in traffic_oracle.get_traffic_lights():
            dx = x - tl_info['location'][0]
            dy = y - tl_info['location'][1]
            if dx * dx + dy * dy > self._cfg.traffic_light_range ** 2:
                continue

            try:
                if (tl_info['actor'].get_state()
                        != carla_mod.TrafficLightState.Red):
                    continue
            except Exception:
                continue

            if CsTrafficRuleOracle._segments_intersect(
                (x, y), (end_x, end_y),
                tl_info['stop_start'], tl_info['stop_end'],
            ):
                return True
        return False


    def _update_history(self, vehicles, step):
        """Record speed history and clean up stale agents."""
        for aid, v in vehicles.items():
            if aid not in self._history:
                self._history[aid] = deque(maxlen=self._max_history)
            speed = v.speed
            self._history[aid].append({'step': step, 'speed': speed})

        stale = [aid for aid in self._history if aid not in vehicles]
        for aid in stale:
            if self._history[aid]:
                last_step = self._history[aid][-1]['step']
                if (step - last_step) * self._dt > self._cfg.w_conflict:
                    del self._history[aid]

    def _detect_stalled(self, vehicles, red_light_queuing):
        """Sliding window stall detection, excluding red-light queuers."""
        stalled = set()
        stall_durations = {}
        w_entries = int(self._cfg.w_stall / self._dt)

        for aid in vehicles:
            if aid in red_light_queuing:
                continue
            hist = self._history.get(aid)
            if hist is None or len(hist) < w_entries:
                continue
            recent = list(hist)[-w_entries:]
            slow = sum(1 for e in recent if e['speed'] < self._cfg.v_stall)
            fraction = slow / len(recent)
            if fraction >= self._cfg.rho_stall:
                stalled.add(aid)
                stall_durations[aid] = slow * self._dt
        return stalled, stall_durations


    def _build_wait_for(self, stalled):
        """Build directed wait-for graph: A->B iff A stalled + spatial edge A->B.

        Only the waiter (A) must be stalled.  The obstacle (B) does NOT need
        to be stalled — B is simply in A's path.  This prevents premature
        single_stuck classification: a stalled vehicle with a spatial conflict
        is "blocked" (in the graph), not "isolated stuck".

        Deadlock still requires a cycle (A->B and B->A), which naturally
        requires both parties to be stalled.
        """
        wait_for: Dict[int, Set[int]] = {}
        for (a, b), _t in self._spatial_edges.items():
            if a in stalled:
                wait_for.setdefault(a, set()).add(b)
        return wait_for


    def _update_deadlocks(self, wait_for, stalled, stall_durations):
        """Persistent deadlock: detect new cycles, re-evaluate existing groups."""
        risks = []

        new_cycles = self._find_cycles(wait_for)

        for cycle_set in new_cycles:
            cycle_frozen = frozenset(cycle_set)
            already = any(cycle_frozen <= existing
                          for existing in self._active_deadlocks)
            if not already:
                merged = set(cycle_set)
                remaining = []
                for existing in self._active_deadlocks:
                    if merged & existing:
                        merged |= existing
                    else:
                        remaining.append(existing)
                remaining.append(merged)
                self._active_deadlocks = remaining

        updated = []
        for group in self._active_deadlocks:
            still_stalled = group & stalled
            exited = group - stalled

            if not exited:
                updated.append(group)
            elif len(still_stalled) >= 2:
                sub_graph = {}
                for a in still_stalled:
                    neighbors = wait_for.get(a, set()) & still_stalled
                    if neighbors:
                        sub_graph[a] = neighbors
                sub_cycles = self._find_cycles(sub_graph)
                if sub_cycles:
                    cycle_members = set()
                    for c in sub_cycles:
                        cycle_members.update(c)
                    updated.append(cycle_members)

        self._active_deadlocks = updated

        for group in self._active_deadlocks:
            group_list = sorted(group)
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    risks.append(CsRiskEvent(
                        risk_type='deadlock',
                        severity=1.0,
                        agent_i=group_list[i],
                        agent_j=group_list[j],
                        timestep=-1,
                        time_seconds=-1.0,
                        details={
                            'cycle_agents': group_list,
                            'stall_duration_i': stall_durations.get(
                                group_list[i], 0),
                            'stall_duration_j': stall_durations.get(
                                group_list[j], 0),
                        },
                    ))
        return risks

    @staticmethod
    def _find_cycles(graph: Dict[int, Set[int]]) -> List[Set[int]]:
        """Find all cycles in directed graph via DFS."""
        visited = set()
        on_stack = set()
        cycles = []
        seen_cycles = set()

        def dfs(node, path):
            visited.add(node)
            on_stack.add(node)
            path.append(node)
            for neighbor in graph.get(node, set()):
                if neighbor in on_stack:
                    cycle_start = path.index(neighbor)
                    cycle_key = frozenset(path[cycle_start:])
                    if cycle_key not in seen_cycles:
                        seen_cycles.add(cycle_key)
                        cycles.append(set(cycle_key))
                elif neighbor not in visited:
                    dfs(neighbor, path)
            path.pop()
            on_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])
        return cycles

    def reset(self):
        """Clear all state."""
        self._history.clear()
        self._spatial_edges.clear()
        self._active_deadlocks.clear()
        self._current_time = 0.0


class CsRiskOracle:
    """Orchestrator: runs all sub-oracles and aggregates risk events."""

    def __init__(self, config: CsRiskConfig):
        self._config = config
        self._collision = CsCollisionOracle(config)
        self._traffic = CsTrafficRuleOracle(config)
        self._deadlock = CsDeadlockOracle(config)
        self._step = 0
        self._flagged_until: Dict[int, float] = {}

    def setup(self):
        """Initialize sub-oracles (call after CARLA world is ready)."""
        self._traffic.setup()

    def assess(self, perception) -> CsRiskAssessment:
        """Run all risk oracles (collision + traffic + deadlock)."""
        self._step += 1
        current_time = self._step * self._config.frame_dt

        collision_risks = (self._collision.assess(perception)
                           if self._config.enable_collision else [])
        traffic_risks = (self._traffic.assess(perception)
                         if self._config.enable_traffic else [])
        deadlock_risks = (self._deadlock.assess(
            perception, self._traffic, self._step)
            if self._config.enable_deadlock else [])

        all_risks = collision_risks + traffic_risks + deadlock_risks

        current_flagged = set()
        for r in all_risks:
            current_flagged.add(r.agent_i)
            if r.agent_j >= 0:
                current_flagged.add(r.agent_j)

        for aid in current_flagged:
            self._flagged_until[aid] = current_time + self._config.w_hold

        flagged = set(current_flagged)
        expired = []
        for aid, until in self._flagged_until.items():
            if current_time < until:
                flagged.add(aid)
            else:
                expired.append(aid)
        for aid in expired:
            del self._flagged_until[aid]

        if all_risks:
            self._log_risks(all_risks)

        return CsRiskAssessment(
            risks=all_risks,
            step=self._step,
            perception_mode=perception.mode,
            num_scenes=len(perception.scenes),
            flagged_agents=flagged,
        )

    def _log_risks(self, risks: List[CsRiskEvent]):
        by_type = {}
        for r in risks:
            by_type.setdefault(r.risk_type, []).append(r)
        parts = ["{}={}".format(rt, len(evts))
                 for rt, evts in by_type.items()]
        print("[CoSafe Risk] Step {}: {}".format(self._step, ", ".join(parts)))

    def destroy(self):
        self._deadlock.reset()
        self._flagged_until.clear()
