"""Autoware privileged-mode CARLA leaderboard agent.

Multi-ego wrapper that drives N vehicles using N independent Autoware stacks
(one Docker container per ego, DDS-isolated via ROS_DOMAIN_ID).  The agent
process (Python 3.7 / colmdriver env) cannot import rclpy, so it talks HTTP to
``autoware_bridge.py`` running inside each container.

Pattern mirrors ``uniad_b2d_agent.py`` and ``lmdriver_agent.py``:
- ``setup`` stores per-ego state arrays (prev_control, next_action_step).
- ``run_step`` loops over ego ids, dispatches ``run_step_single_vehicle``.
- ``REALTIME_MODE`` env var toggles frame-skip / next-action-step behavior.
- Exposes ``_route_planner`` / ``_global_plan_world_coord_all`` so that
  ``cs_scenario_manager`` can inject route data into the CoSafe RSU.
"""

from __future__ import print_function

import datetime
import math
import os
import pathlib
import time

import carla
import numpy as np
import yaml
from scipy.optimize import fsolve

from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from team_code.autoware_client import AutowareClient
from team_code.planner_b2d import RoutePlanner


SAVE_PATH = os.environ.get("SAVE_PATH", None)

_CARLA_TL_STATE_TO_INT = {
    getattr(carla.TrafficLightState, "Red", 0): 0,
    getattr(carla.TrafficLightState, "Yellow", 1): 1,
    getattr(carla.TrafficLightState, "Green", 2): 2,
    getattr(carla.TrafficLightState, "Off", 3): 3,
    getattr(carla.TrafficLightState, "Unknown", 4): 3,
}

_EXCLUDED_ACTOR_KEYWORDS = (
    "brokentile", "calibrator", "mesh", "friction", "dirtdebris",
    "sensor", "traffic_light", "unknown", "spectator",
)


def get_entry_point():
    return "AutowareB2DAgent"


class AutowareB2DAgent(autonomous_agent.AutonomousAgent):
    """Leaderboard agent that proxies one Autoware stack per ego vehicle."""

    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        self.ego_vehicles_num = ego_vehicles_num
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        with open(path_to_conf_file, "r") as f:
            self.config = yaml.safe_load(f)

        aw_cfg = self.config.get("autoware", {})
        host = aw_cfg.get("server_host", "http://localhost")
        port_base = int(os.environ.get("AUTOWARE_PORT_BASE",
                                        aw_cfg.get("server_port_base", 8100)))
        step_timeout = float(aw_cfg.get("step_timeout", 1.0))
        reset_timeout = float(aw_cfg.get("reset_timeout", 5.0))
        max_retries = int(aw_cfg.get("max_retries", 3))
        self._scene_actor_radius = float(aw_cfg.get("scene_actor_radius", 100.0))
        self._scene_actor_min_z = float(aw_cfg.get("scene_actor_min_z", -10.0))

        self.clients = []
        for ego_id in range(ego_vehicles_num):
            url = "%s:%d" % (host, port_base + ego_id)
            self.clients.append(
                AutowareClient(
                    server_url=url, session_id="ego%d" % ego_id,
                    step_timeout=step_timeout,
                    reset_timeout=reset_timeout,
                    max_retries=max_retries,
                )
            )

        self.prev_control = [self._zero_control() for _ in range(10)]
        self.next_action_step = [0] * 10
        self.lat_ref = 42.0
        self.lon_ref = 2.0

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            tag = "_".join(
                "%02d" % x
                for x in (now.month, now.day, now.hour, now.minute, now.second)
            )
            self.save_path = pathlib.Path(SAVE_PATH) / ("autoware_%s" % tag)
            self.save_path.mkdir(parents=True, exist_ok=True)

    def sensors(self):
        """Privileged mode uses ground truth, so only booking sensors are needed.

        Matches ``uniad_b2d_agent.py`` sensor naming so the leaderboard
        spawns per-ego sensors keyed as ``<id>_<ego_id>``.
        """
        return [
            {
                "type": "sensor.other.imu",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "sensor_tick": 0.05, "id": "IMU",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "sensor_tick": 0.01, "id": "GPS",
            },
            {"type": "sensor.speedometer",
             "reading_frequency": 20, "id": "SPEED"},
        ]

    def run_step(self, input_data, timestamp):
        control_all = []
        self.step += 1
        for ego_id in range(self.ego_vehicles_num):
            if CarlaDataProvider.get_hero_actor(hero_id=ego_id) is not None:
                control_all.append(
                    self.run_step_single_vehicle(ego_id, input_data, timestamp)
                )
            else:
                control_all.append(None)
        return control_all

    def run_step_single_vehicle(self, ego_id, input_data, timestamp):
        if not self.initialized:
            self._init()

        start = time.time()
        payload = self._build_payload(ego_id, timestamp)
        if payload is None:
            return self.prev_control[ego_id]

        response = self.clients[ego_id].step(payload)

        realtime_mode = os.environ.get("REALTIME_MODE", "0")
        use_new_control = True
        if realtime_mode == "1":
            if self.step < self.next_action_step[ego_id] and self.step:
                use_new_control = False
        else:
            if self.step % 4 != 0 and self.step > 0:
                use_new_control = False

        if use_new_control and response is not None and response.get("ok"):
            control = carla.VehicleControl()
            control.steer = float(np.clip(response.get("steer", 0.0), -1.0, 1.0))
            control.throttle = float(np.clip(response.get("throttle", 0.0), 0.0, 0.75))
            control.brake = float(np.clip(response.get("brake", 0.0), 0.0, 1.0))
            self.prev_control[ego_id] = control

        elapsed = time.time() - start
        self.next_action_step[ego_id] = self.step + int(elapsed * 20)
        ctl = self.prev_control[ego_id]
        if self.step % 20 == 0:
            print("[AutowareAgent] step=%d ego=%d steer=%.3f throttle=%.3f brake=%.3f use_new=%s elapsed=%.2f"
                  % (self.step, ego_id, ctl.steer, ctl.throttle, ctl.brake, use_new_control, elapsed),
                  flush=True)
        return ctl

    def destroy(self):
        for client in getattr(self, "clients", []):
            try:
                client.close()
            except Exception:
                pass

    def _init(self):
        try:
            locx = self._global_plan_world_coord[0][0][0].location.x
            locy = self._global_plan_world_coord[0][0][0].location.y
            lon = self._global_plan[0][0][0]["lon"]
            lat = self._global_plan[0][0][0]["lat"]
            earth = 6378137.0

            def equations(vars_):
                x, y = vars_
                eq1 = (lon * math.cos(x * math.pi / 180)
                       - (locx * x * 180) / (math.pi * earth)
                       - math.cos(x * math.pi / 180) * y)
                eq2 = (math.log(math.tan((lat + 90) * math.pi / 360)) * earth
                       * math.cos(x * math.pi / 180)
                       + locy
                       - math.cos(x * math.pi / 180) * earth
                       * math.log(math.tan((90 + x) * math.pi / 360)))
                return [eq1, eq2]

            solution = fsolve(equations, [0, 0])
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print("[AutowareB2DAgent] lat/lon ref fallback:", e, flush=True)
            self.lat_ref, self.lon_ref = 0.0, 0.0

        self._route_planner = RoutePlanner(
            4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref,
        )
        self._route_planner.set_route(self._global_plan, True)

        for ego_id in range(self.ego_vehicles_num):
            plan = self._serialize_route(ego_id)
            ok = self.clients[ego_id].reset(plan)
            if not ok:
                print("[AutowareB2DAgent] bridge reset failed for ego %d" % ego_id,
                      flush=True)

        self.initialized = True

    def _serialize_route(self, ego_id):
        plan = []
        if ego_id >= len(self._global_plan_world_coord):
            return plan
        for wp_tf, _cmd in self._global_plan_world_coord[ego_id]:
            loc = wp_tf.location
            rot = wp_tf.rotation
            plan.append([
                float(loc.x), float(loc.y), float(loc.z),
                float(rot.roll), float(rot.pitch), float(rot.yaw),
            ])
        return plan

    def _build_payload(self, ego_id, timestamp):
        world = CarlaDataProvider.get_world()
        if world is None:
            return None
        snapshot = world.get_snapshot()
        sim_time = snapshot.timestamp.elapsed_seconds

        ego = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
        if ego is None or not ego.is_alive:
            return None

        tf = ego.get_transform()
        vel = ego.get_velocity()
        ang = ego.get_angular_velocity()
        acc = ego.get_acceleration()

        try:
            sl = ego.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
            sr = ego.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
        except Exception:
            sl, sr = 0.0, 0.0

        tl_state = 0
        at_tl = False
        try:
            at_tl = bool(ego.is_at_traffic_light())
            if at_tl:
                tl_state = _CARLA_TL_STATE_TO_INT.get(
                    ego.get_traffic_light_state(), 0
                )
        except Exception:
            pass

        payload = {
            "timestamp": float(sim_time),
            "ego": self._serialize_ego(tf, vel, ang, acc, sl, sr),
            "scene_actors": self._serialize_scene(world, ego),
            "traffic_light": {"at_tl": at_tl, "state": int(tl_state)},
        }
        return payload

    @staticmethod
    def _serialize_ego(tf, vel, ang, acc, sl, sr):
        return {
            "x": float(tf.location.x),
            "y": float(tf.location.y),
            "z": float(tf.location.z),
            "roll": float(tf.rotation.roll),
            "pitch": float(tf.rotation.pitch),
            "yaw": float(tf.rotation.yaw),
            "vx": float(vel.x), "vy": float(vel.y), "vz": float(vel.z),
            "ang_vx": float(ang.x), "ang_vy": float(ang.y), "ang_vz": float(ang.z),
            "ax": float(acc.x), "ay": float(acc.y), "az": float(acc.z),
            "steering_l_deg": float(sl),
            "steering_r_deg": float(sr),
        }

    def _serialize_scene(self, world, ego):
        ego_id = ego.id
        ego_loc = ego.get_location()
        out = []
        try:
            actors = world.get_actors()
        except Exception:
            return out
        radius = self._scene_actor_radius
        min_z = self._scene_actor_min_z
        for actor in actors:
            try:
                if actor.id == ego_id:
                    continue
                if not actor.is_alive:
                    continue
                type_id = str(actor.type_id).lower()
                if any(kw in type_id for kw in _EXCLUDED_ACTOR_KEYWORDS):
                    continue
                loc = actor.get_location()
                if loc.z < min_z:
                    continue
                if loc.distance(ego_loc) > radius:
                    continue
                tf = actor.get_transform()
                vel = actor.get_velocity()
                ang = actor.get_angular_velocity()
                acc = actor.get_acceleration()
                bb = actor.bounding_box
                extent = bb.extent
                wsl, wsr = 0.0, 0.0
                if "vehicle" in type_id:
                    try:
                        wsl = actor.get_wheel_steer_angle(
                            carla.VehicleWheelLocation.FL_Wheel
                        )
                        wsr = actor.get_wheel_steer_angle(
                            carla.VehicleWheelLocation.FR_Wheel
                        )
                    except Exception:
                        pass
                out.append({
                    "id": int(actor.id),
                    "type": type_id,
                    "x": float(tf.location.x),
                    "y": float(tf.location.y),
                    "z": float(tf.location.z),
                    "roll": float(tf.rotation.roll),
                    "pitch": float(tf.rotation.pitch),
                    "yaw": float(tf.rotation.yaw),
                    "vx": float(vel.x), "vy": float(vel.y), "vz": float(vel.z),
                    "ang_vx": float(ang.x),
                    "ang_vy": float(ang.y),
                    "ang_vz": float(ang.z),
                    "ax": float(acc.x), "ay": float(acc.y), "az": float(acc.z),
                    "bbox_extent": [float(extent.x), float(extent.y), float(extent.z)],
                    "wheel_steer_l_deg": float(wsl),
                    "wheel_steer_r_deg": float(wsr),
                })
            except Exception:
                continue
        return out

    @staticmethod
    def _zero_control():
        ctrl = carla.VehicleControl()
        ctrl.steer = 0.0
        ctrl.throttle = 0.0
        ctrl.brake = 0.0
        return ctrl
