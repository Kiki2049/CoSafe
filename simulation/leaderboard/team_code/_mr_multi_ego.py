"""Multi-ego wrapper helpers for Argus *_b2d_agent_mr classes.

Argus MR agents are written for a single ego vehicle. CoSafe's leaderboard runs
N egos through ONE agent instance, expecting `setup(path, ego_n)`, sensor data
keyed by `<id>_<ego_id>`, and `run_step` returning a list of N controls.

We bridge the two by spawning N internal Argus instances inside the wrapper,
demuxing per-ego inputs and muxing controls. Heavy model weights are shared
across instances to keep GPU memory bounded.
"""
import math
import os
import sys
import types
from contextlib import contextmanager

import carla  # noqa: E402

if not hasattr(carla.Vector3D, 'length'):
    carla.Vector3D.length = lambda self: math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
if not hasattr(carla.Vector3D, 'distance'):
    carla.Vector3D.distance = lambda self, other: math.sqrt(
        (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
    )
if not hasattr(carla.Vector3D, 'distance_2d'):
    carla.Vector3D.distance_2d = lambda self, other: math.sqrt(
        (self.x - other.x) ** 2 + (self.y - other.y) ** 2
    )
if not hasattr(carla.Vector3D, 'squared_length'):
    carla.Vector3D.squared_length = lambda self: self.x ** 2 + self.y ** 2 + self.z ** 2


def _get_metric_info(self):
    """Compat shim: Argus's autonomous_agent has `get_metric_info` using
    `self.hero_actor`; CoSafe's autonomous_agent does not. Inner Argus agents
    call `self.get_metric_info()` from `run_step`. We bind this method onto
    each inner instance and use `self._vehicle` (set by Argus _init via our
    `patched_hero` context).
    """
    def vec(v, rot=False):
        return [v.roll, v.pitch, v.yaw] if rot else [v.x, v.y, v.z]

    h = getattr(self, '_vehicle', None)
    if h is None:
        return {}
    tf = h.get_transform()
    return {
        'acceleration': vec(h.get_acceleration()),
        'angular_velocity': vec(h.get_angular_velocity()),
        'forward_vector': vec(tf.get_forward_vector()),
        'right_vector': vec(tf.get_right_vector()),
        'location': vec(tf.location),
        'rotation': vec(tf.rotation, rot=True),
    }

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLMDRIVER_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
_B2D_ROOT = os.path.join(_COLMDRIVER_ROOT, 'Bench2DriveZoo')
_B2D_TEAM_CODE = os.path.join(_B2D_ROOT, 'team_code')
for _p in (_B2D_TEAM_CODE, _B2D_ROOT, _COLMDRIVER_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from leaderboard.autoagents.autonomous_agent import AutonomousAgent  # noqa: E402
from leaderboard.utils.route_manipulation import downsample_route  # noqa: E402
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # noqa: E402


@contextmanager
def patched_hero(ego_id):
    """Temporarily redirect `CarlaDataProvider.get_hero_actor()` (default arg) to
    return the hero indexed by `ego_id`. Argus internals call get_hero_actor()
    with no args expecting the single ego; we use this to point them at the
    correct ego per call.
    """
    orig = CarlaDataProvider.get_hero_actor

    def wrapped(hero_id=ego_id):
        return orig(hero_id=hero_id)

    CarlaDataProvider.get_hero_actor = staticmethod(wrapped)
    try:
        yield
    finally:
        CarlaDataProvider.get_hero_actor = orig


class _CachedForward:
    """Wraps a heavy nn.Module (UniAD / BEVFormer) so its forward only runs
    once every `skip_k` calls; intermediate calls return the cached output.

    This mirrors baseline UniAD's frame-skip optimization without breaking MR's
    internal step / collision_window / stuck_window timing — those still update
    every CARLA tick because `inst.run_step` (and therefore `inst.tick`) still
    runs every tick. Only the GPU-heavy `model(...)` and `bev_model(...)` calls
    are skipped.
    """

    def __init__(self, real_module, skip_k):
        self._real = real_module
        self._skip_k = max(1, int(skip_k))
        self._call_count = -1
        self._cache = None

    def __call__(self, *args, **kwargs):
        self._call_count += 1
        if self._cache is None or self._call_count % self._skip_k == 0:
            self._cache = self._real(*args, **kwargs)
        return self._cache

    def __getattr__(self, name):
        return getattr(self._real, name)


def _argus_set_global_plan(inst, plan_gps, plan_world):
    """Replicate Argus's autonomous_agent.set_global_plan for a single-ego inst.
    CoSafe's parent set_global_plan stores nested lists; Argus expects flat.
    """
    inst.org_dense_route_gps = plan_gps
    inst.org_dense_route_world_coord = plan_world
    ds_ids = downsample_route(plan_world, 50)
    inst._global_plan_world_coord = [(plan_world[x][0], plan_world[x][1]) for x in ds_ids]
    inst._global_plan = [plan_gps[x] for x in ds_ids]
    inst._plan_gps_HACK = plan_gps


class MultiEgoMRWrapper(AutonomousAgent):
    """Base wrapper. Subclasses set `INNER_CLS` to an Argus MR class.

    Shared model attributes are listed in `_SHARED_ATTRS`; after instances are
    constructed, attributes from instance 0 are copied to others (avoids N×GPU
    memory for big models).
    """

    INNER_CLS = None
    _SHARED_ATTRS = ('inference_only_pipeline', 'bevformer_pipeline', '_im_transform')
    _CACHED_FORWARD_ATTRS = ('model', 'bev_model')

    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        assert self.INNER_CLS is not None, 'subclass must set INNER_CLS'
        self.ego_vehicles_num = ego_vehicles_num
        self._instances = []
        orig_save_path = os.environ.get('SAVE_PATH')
        try:
            for i in range(ego_vehicles_num):
                if i > 0 and orig_save_path is not None:
                    per_ego = os.path.join(orig_save_path, f'_ego_{i}')
                    os.makedirs(per_ego, exist_ok=True)
                    os.environ['SAVE_PATH'] = per_ego
                inst = object.__new__(self.INNER_CLS)
                inst.agent_name = f'auto_agent_ego{i}'
                inst.rsu = []
                from leaderboard.autoagents.autonomous_agent import Track
                inst.track = Track.SENSORS
                inst._global_plan = None
                inst._global_plan_world_coord = None
                inst.sensor_interface = self.sensor_interface
                inst.wallclock_t0 = None
                inst._mr_ego_id = i
                inst.setup(path_to_conf_file)
                inst.get_metric_info = types.MethodType(_get_metric_info, inst)
                self._instances.append(inst)
        finally:
            if orig_save_path is not None:
                os.environ['SAVE_PATH'] = orig_save_path
            else:
                os.environ.pop('SAVE_PATH', None)

        self.track = self._instances[0].track

        if ego_vehicles_num > 1:
            for attr in self._SHARED_ATTRS:
                if hasattr(self._instances[0], attr):
                    val = getattr(self._instances[0], attr)
                    for inst in self._instances[1:]:
                        setattr(inst, attr, val)

        skip_k = 1 if os.environ.get('REALTIME_MODE', '0') == '1' else 4
        for inst in self._instances:
            for attr in self._CACHED_FORWARD_ATTRS:
                m = getattr(inst, attr, None)
                if m is None or isinstance(m, _CachedForward):
                    continue
                setattr(inst, attr, _CachedForward(m, skip_k))

    def sensors(self):
        return self._instances[0].sensors()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        for i, inst in enumerate(self._instances):
            _argus_set_global_plan(inst, global_plan_gps[i], global_plan_world_coord[i])
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def run_step(self, input_data, timestamp):
        controls = []
        for i, inst in enumerate(self._instances):
            if CarlaDataProvider.get_hero_actor(hero_id=i) is None:
                controls.append(None)
                continue
            suffix = f'_{i}'
            local_data = {
                k[: -len(suffix)]: v
                for k, v in input_data.items()
                if k.endswith(suffix)
            }
            if not local_data:
                controls.append(None)
                continue
            actual_prev_ctrl = None
            if getattr(inst, '_vehicle', None) is not None:
                try:
                    actual_prev_ctrl = inst._vehicle.get_control()
                except Exception:
                    pass

            with patched_hero(i):
                ctrl = inst.run_step(local_data, timestamp)

            if getattr(inst, '_diag_n', 0) < 20 and getattr(inst, 'step', 0) % 20 == 0:
                inst._diag_n = getattr(inst, '_diag_n', 0) + 1
                veh = getattr(inst, '_vehicle', None)
                pos = veh.get_location() if veh is not None else None
                vel = veh.get_velocity() if veh is not None else None
                cmd_near = inst.tick_data.get('command_near') if hasattr(inst, 'tick_data') else None
                wp0 = None
                try:
                    wp0 = inst.pred_wp[0] if getattr(inst, 'pred_wp', None) is not None else None
                except Exception:
                    pass
                if actual_prev_ctrl is not None:
                    actual_str = (
                        f'(t={actual_prev_ctrl.throttle:.2f} '
                        f'b={actual_prev_ctrl.brake:.2f} '
                        f's={actual_prev_ctrl.steer:.2f} '
                        f'hb={actual_prev_ctrl.hand_brake})'
                    )
                else:
                    actual_str = 'None'
                print(
                    f'[MR_DIAG] ego{i} step={inst.step} '
                    f'pos=({pos.x:.1f},{pos.y:.1f}) speed={vel.length() if vel else 0.0:.2f} '
                    f'cmd={cmd_near} pred_wp0={wp0} '
                    f'we_send=(t={ctrl.throttle:.2f} b={ctrl.brake:.2f} s={ctrl.steer:.2f}) '
                    f'carla_actual_prev={actual_str} '
                    f'stuck={getattr(inst, "stuck_detected", None)} coll={getattr(inst, "collision_detected", None)}',
                    flush=True
                )
            controls.append(ctrl)
        return controls

    def destroy(self):
        for inst in self._instances:
            try:
                inst.destroy()
            except Exception:
                pass
