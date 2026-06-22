"""
Mixed-deployment wrapper agent for RQ5.

Wraps {TCP, LMDrive, UniAD} sub-agents under one AutonomousAgent interface so
that different ego AVs in the same scenario can run different ADSes. ADS
assignment per ego follows a round-robin rule with a route-hash offset (so the
same route gets the same assignment in baseline and CoSafe runs, making them
directly comparable).

Architecture
------------
- Wrapper exposes a UNION of sub-agent sensor specs:
    * Cameras / lidar / radar  → prefixed by ADS kind (`tcp_rgb`, `uniad_CAM_FRONT`)
    * IMU / GPS / speedometer / opendrive_map → SHARED across sub-agents
      (the wrapper publishes one of each; at run_step the same data dict is
      aliased under each sub-agent's expected sensor id).
- Sensor count exceeds the default `SENSORS_LIMITS` (challenge-submission rule).
  We monkey-patch that module attribute at import time. This is scoped to
  processes that import this wrapper; single-ADS runs are unaffected.
- SAVE_PATH per sub-agent is isolated by overriding each agent module's
  SAVE_PATH attribute (and the env var) before its setup, then restored.
- run_step splits incoming `input_data` into sub-agent-specific dicts (with
  local ego ids) and calls each sub-agent's run_step. The merged controls are
  returned in a list indexed by global ego id.

Compatibility
-------------
- Does not modify any existing source files.
- Existing single-ADS eval paths (eval_cosafe.sh tcp/lmdrive/uniad/...) are
  unaffected because they don't import this module.
"""
import datetime
import hashlib
import json
import os
import pathlib
import yaml

import carla
import numpy as np

from leaderboard.autoagents import autonomous_agent

import leaderboard.autoagents.agent_wrapper as _agent_wrapper
_agent_wrapper.SENSORS_LIMITS = {
    'sensor.camera.rgb': 30,
    'sensor.camera.semantic_segmentation': 10,
    'sensor.camera.depth': 10,
    'sensor.lidar.ray_cast': 3,
    'sensor.lidar.ray_cast_semantic': 1,
    'sensor.other.radar': 4,
    'sensor.other.gnss': 5,
    'sensor.other.imu': 5,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 5,
}

_orig_setup_sensors = _agent_wrapper.AgentWrapper.setup_sensors
_orig_del_ego_sensor = _agent_wrapper.AgentWrapper.del_ego_sensor


def _patched_setup_sensors(self, vehicle, vehicle_num=0, save_root=None, debug_mode=False):
    set_fn = getattr(self._agent, '_set_current_vehicle_num', None)
    if set_fn:
        set_fn(vehicle_num)
    try:
        return _orig_setup_sensors(self, vehicle, vehicle_num, save_root, debug_mode)
    finally:
        if set_fn:
            set_fn(None)


def _patched_del_ego_sensor(self, vehicle_num=0):
    set_fn = getattr(self._agent, '_set_current_vehicle_num', None)
    if set_fn:
        set_fn(vehicle_num)
    try:
        return _orig_del_ego_sensor(self, vehicle_num)
    finally:
        if set_fn:
            set_fn(None)


_agent_wrapper.AgentWrapper.setup_sensors = _patched_setup_sensors
_agent_wrapper.AgentWrapper.del_ego_sensor = _patched_del_ego_sensor

_SINGLETON_TYPES = (
    'sensor.other.imu',
    'sensor.other.gnss',
    'sensor.speedometer',
    'sensor.opendrive_map',
)

_FORWARDED_ATTRS = (
    'town_id',
    'sampled_scenarios',
    'scenario_cofing_name',
)


def get_entry_point():
    return 'MixedAgent'


def _import_tcp():
    from team_code.tcp_agent import TCPAgent
    import team_code.tcp_agent as mod
    return TCPAgent, mod


def _import_lmdrive():
    from team_code.lmdriver_agent import LMDriveAgent
    import team_code.lmdriver_agent as mod
    return LMDriveAgent, mod


def _import_uniad():
    from team_code.uniad_b2d_agent import UniadAgent
    import team_code.uniad_b2d_agent as mod
    return UniadAgent, mod


_KIND_FACTORIES = {
    'tcp':     _import_tcp,
    'lmdrive': _import_lmdrive,
    'uniad':   _import_uniad,
}


def _route_hash_offset(num_kinds: int) -> int:
    """Stable offset derived from the ROUTES env var. Same route → same offset
    across baseline and CoSafe runs, so the assignment is directly comparable.
    """
    routes = os.environ.get('ROUTES', '')
    h = hashlib.md5(routes.encode('utf-8')).hexdigest()
    return int(h, 16) % max(num_kinds, 1)


class _AggregatedRoutePlanner(object):
    def __init__(self, wrapper):
        self._wrapper = wrapper

    @property
    def route(self):
        out = []
        for global_id in range(self._wrapper.ego_vehicles_num):
            sub_route = []
            if global_id < len(self._wrapper.assignment):
                kind = self._wrapper.assignment[global_id]
                sub = self._wrapper.sub_agents.get(kind)
                if sub is not None:
                    sub_rp = getattr(sub, '_route_planner', None)
                    if sub_rp is not None:
                        sub_routes = getattr(sub_rp, 'route', None)
                        if sub_routes is not None and global_id < len(sub_routes):
                            sub_route = sub_routes[global_id]
            out.append(sub_route)
        return out


class MixedAgent(autonomous_agent.AutonomousAgent):
    """Multi-ADS wrapper. See module docstring for details."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in _FORWARDED_ATTRS:
            subs = self.__dict__.get('sub_agents', None)
            if subs:
                for sub in subs.values():
                    try:
                        setattr(sub, name, value)
                    except Exception:
                        pass

    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        self.track = autonomous_agent.Track.SENSORS
        self.ego_vehicles_num = ego_vehicles_num
        self.step = -1

        with open(path_to_conf_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        self._mixed_cfg = cfg
        self.agent_name = cfg.get('agent_name', 'lmdrive')
        rotation = cfg['rotation']
        self._rotation_kinds = [r['kind'] for r in rotation]
        self._kind_to_subconfig = {r['kind']: r['config'] for r in rotation}
        self._kind_to_prefix = {r['kind']: r['sensor_prefix'] for r in rotation}

        offset_strategy = cfg.get('offset_strategy', 'route_hash')
        if offset_strategy == 'route_hash':
            offset = _route_hash_offset(len(self._rotation_kinds))
        else:
            offset = 0
        self._offset = offset

        self.assignment = [
            self._rotation_kinds[(i + offset) % len(self._rotation_kinds)]
            for i in range(ego_vehicles_num)
        ]

        self.kind_to_globals = {}
        kind_counts = {}
        for global_id, kind in enumerate(self.assignment):
            self.kind_to_globals.setdefault(kind, []).append(global_id)
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        self.k_per_kind = kind_counts

        save_root = os.environ.get('SAVE_PATH', None)
        if save_root:
            now = datetime.datetime.now()
            ts = '_'.join('%02d' % x for x in (now.month, now.day, now.hour, now.minute, now.second))
            route_stem = pathlib.Path(os.environ.get('ROUTES', '')).stem
            self.save_path = pathlib.Path(save_root) / f'{route_stem}_{ts}_mixed'
            self.save_path.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None

        self.sub_agents = {}
        for kind in self._rotation_kinds:
            if kind not in kind_counts:
                continue
            self.sub_agents[kind] = self._instantiate_sub(
                kind, self._kind_to_subconfig[kind], ego_vehicles_num,
            )

        self._route_planner = _AggregatedRoutePlanner(self)
        self._build_sensor_list()
        self._write_assignment_log()

        print(
            f"[MixedAgent] route={pathlib.Path(os.environ.get('ROUTES','?')).stem} "
            f"ego_num={ego_vehicles_num} offset={offset} "
            f"assignment={self.assignment} counts={kind_counts}"
        )

    def _instantiate_sub(self, kind, sub_conf_path, ego_vehicles_num):
        """Sub-agents are instantiated with the WRAPPER's full ego count so
        their internal arrays / hero-actor lookups can be indexed by global
        ego_id (they iterate range(ego_vehicles_num) and call
        get_hero_actor(hero_id=ego_id) — passing local counts breaks that).
        Per-sub-agent SAVE_PATH is forced to None permanently so they skip
        image dumps; the wrapper handles save_path centrally.
        """
        if kind not in _KIND_FACTORIES:
            raise ValueError(f"Unknown ADS kind: {kind}")
        cls, mod = _KIND_FACTORIES[kind]()

        prev_env = os.environ.get('SAVE_PATH', None)
        os.environ.pop('SAVE_PATH', None)
        mod.SAVE_PATH = None
        try:
            sub = cls(sub_conf_path, ego_vehicles_num)
        finally:
            if prev_env is None:
                os.environ.pop('SAVE_PATH', None)
            else:
                os.environ['SAVE_PATH'] = prev_env
        return sub


    def _build_sensor_list(self):
        """Build the sensor universe AND per-kind subsets used for per-ego mounting.

        Cameras/lidar/radar: prefixed by ADS kind. Singletons (imu/gps/speed):
        ONE published sensor per ego, with a single canonical id (the first
        rotation kind's). At run_step we alias the data under each sub-agent's
        expected sensor id. The validation-time `sensors()` returns the full
        union; at per-ego setup_sensors time we return only the assigned kind's
        sensors so each ego only mounts what it actually consumes.
        """
        merged = []
        per_kind = {}
        singletons = {}

        for kind in self._rotation_kinds:
            if kind not in self.sub_agents:
                continue
            sub = self.sub_agents[kind]
            kind_specs = []
            for spec in sub.sensors():
                t = spec['type']
                if t in _SINGLETON_TYPES:
                    if t not in singletons:
                        new_spec = dict(spec)
                        merged.append(new_spec)
                        singletons[t] = {
                            'src_id': spec['id'],
                            'src_kind': kind,
                            'aliases': {kind: spec['id']},
                        }
                    else:
                        singletons[t]['aliases'][kind] = spec['id']
                    kind_specs.append({
                        'type': t,
                        'singleton_type': t,
                    })
                else:
                    new_spec = dict(spec)
                    new_spec['id'] = f"{self._kind_to_prefix[kind]}_{spec['id']}"
                    merged.append(new_spec)
                    kind_specs.append(new_spec)
            per_kind[kind] = kind_specs

        self._cached_sensors = merged
        self._sensors_per_kind = per_kind
        self._singletons = singletons

        per_kind_published = {}
        for kind, specs in per_kind.items():
            out = []
            for spec in specs:
                if 'singleton_type' in spec:
                    pub = dict(self._singleton_spec(spec['singleton_type']))
                    out.append(pub)
                else:
                    out.append(spec)
            per_kind_published[kind] = out
        self._sensors_per_kind_published = per_kind_published

    def _singleton_spec(self, t):
        for spec in self._cached_sensors:
            if spec['type'] == t:
                return spec
        return None

    def _set_current_vehicle_num(self, vehicle_num):
        self._current_vehicle_num = vehicle_num

    def sensors(self):
        vnum = getattr(self, '_current_vehicle_num', None)
        if vnum is None or vnum >= len(self.assignment):
            return self._cached_sensors
        kind = self.assignment[vnum]
        return self._sensors_per_kind_published.get(kind, self._cached_sensors)


    def run_step(self, input_data, timestamp):
        """Dispatch input_data to each sub-agent and merge controls.

        Bypasses the buggy `for ego_id in range(ego_vehicles_num)` loop in each
        sub-agent (which calls get_hero_actor(local_id) treating local_id as
        global) by calling run_step_single_vehicle(global_id, ...) directly.
        Sensor keys are kept under their global ego ids so get_hero_actor and
        input_data['<id>_<global_id>'] both resolve to the right ego.
        """
        self.step += 1
        controls = [None] * self.ego_vehicles_num
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        for kind, sub in self.sub_agents.items():
            sub.step += 1
            sub_input = self._build_sub_input(kind, input_data)
            for global_id in self.kind_to_globals[kind]:
                hero = CarlaDataProvider.get_hero_actor(hero_id=global_id)
                if hero is None or not hero.is_alive:
                    continue
                try:
                    control = sub.run_step_single_vehicle(global_id, sub_input, timestamp)
                    controls[global_id] = control
                except Exception as e:
                    print(f"[MixedAgent] {kind} ego {global_id} run_step_single_vehicle failed: {e}")
        for i, c in enumerate(controls):
            if c is None:
                fallback = carla.VehicleControl()
                fallback.steer = 0.0
                fallback.throttle = 0.0
                fallback.brake = 1.0
                controls[i] = fallback
        return controls

    def _build_sub_input(self, kind, input_data):
        """Build the input_data dict each sub-agent expects, keyed by global
        ego_id so that sub.run_step_single_vehicle(global_id, ...) finds its
        sensor data at `<sub_id>_<global_id>`.
        """
        sub = self.sub_agents[kind]
        prefix = self._kind_to_prefix[kind]
        sub_input = {}
        for spec in sub.sensors():
            sub_id = spec['id']
            t = spec['type']
            for global_id in self.kind_to_globals[kind]:
                if t in _SINGLETON_TYPES:
                    src_id = self._singletons[t]['src_id']
                    src_key = f"{src_id}_{global_id}"
                else:
                    src_key = f"{prefix}_{sub_id}_{global_id}"
                if src_key in input_data:
                    sub_input[f"{sub_id}_{global_id}"] = input_data[src_key]
        return sub_input


    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Forward the FULL multi-ego plan to every sub-agent so their
        per-ego route planners can be indexed by global ego_id (matching the
        global-id dispatch in run_step). Also populate the wrapper's own
        _global_plan_world_coord_all / _global_plan_all so CoSafe can pull
        per-ego reference paths from the wrapper directly.
        """
        try:
            super().set_global_plan(global_plan_gps, global_plan_world_coord)
        except Exception:
            pass

        if not getattr(self, '_global_plan_world_coord_all', None):
            from leaderboard.utils.route_manipulation import downsample_route
            self._global_plan_world_coord = []
            self._global_plan = []
            self._global_plan_world_coord_all = []
            self._global_plan_all = []
            distance = 50
            for vehicle_id in range(len(global_plan_gps)):
                ds_ids = downsample_route(global_plan_world_coord[vehicle_id], distance)
                ds_world = [(global_plan_world_coord[vehicle_id][x][0], global_plan_world_coord[vehicle_id][x][1]) for x in ds_ids]
                ds_gps = [global_plan_gps[vehicle_id][x] for x in ds_ids]
                full_world = [(global_plan_world_coord[vehicle_id][x][0], global_plan_world_coord[vehicle_id][x][1]) for x in range(len(global_plan_world_coord[vehicle_id]))]
                full_gps = global_plan_gps[vehicle_id]
                self._global_plan_world_coord.append(ds_world)
                self._global_plan.append(ds_gps)
                self._global_plan_world_coord_all.append(full_world)
                self._global_plan_all.append(full_gps)

        for sub in self.sub_agents.values():
            sub.set_global_plan(global_plan_gps, global_plan_world_coord)


    def get_save_path(self):
        return self.save_path

    def destroy(self):
        for sub in self.sub_agents.values():
            try:
                sub.destroy()
            except Exception as e:
                print(f"[MixedAgent] sub-agent destroy raised: {e}")
        self.sub_agents = {}


    def _write_assignment_log(self):
        """Write per-scenario ADS assignment to a JSON file alongside results.
        Aggregator script (cs_mixed_assignment_stats.py) consumes these.
        """
        endpoint = os.environ.get('CHECKPOINT_ENDPOINT', '')
        if not endpoint:
            return
        out_dir = os.path.dirname(endpoint)
        os.makedirs(out_dir, exist_ok=True)
        record = {
            'route': pathlib.Path(os.environ.get('ROUTES', '')).stem,
            'scenarios': os.environ.get('EVAL_SETTING', ''),
            'ego_vehicles_num': self.ego_vehicles_num,
            'rotation': self._rotation_kinds,
            'offset': self._offset,
            'assignment': {str(i): k for i, k in enumerate(self.assignment)},
            'counts': {k: self.k_per_kind.get(k, 0) for k in self._rotation_kinds},
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        out_path = os.path.join(out_dir, 'mixed_assignment.json')
        try:
            with open(out_path, 'w') as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            print(f"[MixedAgent] failed to write assignment log: {e}")
