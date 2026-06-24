"""Bridge converter utilities (runs inside Autoware Docker, Python 3.10 + ROS2 Humble).

Ported from Autoware/autoware_carla_leaderboard/src/tum_agents/autoware_agent/aw_converter.py
- BridgeHelper.carla2ros_pose, rpy2quat
- AutowareToCarlaControl (ackermann->carla steering + PID throttle/brake)
- PayloadPredictedObjectsBuilder (replaces CarlaObjectsToAutoware; reads serialized scene actors
  from HTTP payload instead of a live CARLA client, because the bridge has no CARLA API access
  when running inside the Autoware container).
"""

from __future__ import annotations

import uuid
from collections import deque

import numpy as np
import transforms3d
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Point32, Pose, Quaternion, Vector3
from unique_identifier_msgs.msg import UUID

from autoware_perception_msgs.msg import (
    ObjectClassification,
    PredictedObject,
    PredictedObjects,
    PredictedPath,
    Shape,
    TrafficLightElement,
)


class BridgeHelper:
    @classmethod
    def carla2ros_pose(cls, x, y, z, roll, pitch, yaw, to_quat=False):
        out_position = {"x": x, "y": -y, "z": z}
        out_orientation = {"roll": -roll, "pitch": pitch, "yaw": -yaw}
        if to_quat:
            out_orientation = cls.rpy2quat(
                out_orientation["roll"], out_orientation["pitch"], out_orientation["yaw"]
            )
        return {"position": out_position, "orientation": out_orientation}

    @staticmethod
    def rpy2quat(roll, pitch, yaw):
        quat = transforms3d.euler.euler2quat(roll, pitch, yaw, axes="sxyz")
        return {"x": quat[1], "y": quat[2], "z": quat[3], "w": quat[0]}


class AutowareToCarlaControl:
    def __init__(self):
        self._ackermann_angle = np.array([
            0.0, 6.7780, 13.1744, 19.2750, 25.1526,
            30.8688, 36.4756, 42.0187, 47.5399, 53.0797, 58.6798,
        ])
        self._carla_angle = np.array([
            0.0, -0.1, -0.2, -0.3, -0.4, -0.5,
            -0.6, -0.7, -0.8, -0.9, -1.0,
        ])
        self._Kp = 0.5
        self._Ki = 0.1
        self._Kd = 0.0
        self._max_throttle = 0.6
        self._max_brake = 0.2
        self._dt = 0.05
        self._integral_error = 0.0
        self._prev_error = 0.0

    def convertLatAwToCarla(self, ackermann_angle_deg):
        ackermann_angle_deg = np.clip(
            ackermann_angle_deg, -self._ackermann_angle[-1], self._ackermann_angle[-1]
        )
        sign = np.sign(ackermann_angle_deg)
        carla_angle = np.interp(
            abs(ackermann_angle_deg), self._ackermann_angle, self._carla_angle
        )
        return float(sign * carla_angle)

    def updateThrottleBrake(self, target_acc, current_acc, velocity):
        error = target_acc - current_acc
        self._integral_error += error * self._dt
        derivative_error = (error - self._prev_error) / self._dt

        if velocity > 2.0:
            output = self._Kp * error + self._Ki * self._integral_error + self._Kd * derivative_error
        else:
            output = self._Kp * error + self._Ki * self._integral_error

        if output > self._max_throttle:
            output = self._max_throttle
            self._integral_error -= error * self._dt
        elif output < -self._max_brake:
            output = -self._max_brake
            self._integral_error -= error * self._dt
        self._prev_error = error

        throttle = max(0.0, min(self._max_throttle, output))
        brake = max(0.0, min(self._max_brake, -output))
        return float(throttle), float(brake)


class PayloadPredictedObjectsBuilder:
    """Build autoware PredictedObjects message from serialized scene actors in payload."""

    def __init__(self, timestep=0.05, prediction_horizon=2.5, wheelbase=2.86,
                 moving_average_window=7):
        self._timestep = timestep
        self._prediction_horizon = prediction_horizon
        self._wheelbase = wheelbase
        self._moving_average_window = moving_average_window
        self._ros_duration = Duration()
        self._ros_duration.sec = int(timestep)
        self._ros_duration.nanosec = int((timestep - int(timestep)) * 1e9)
        self._steering_angle_history = {}

    def build(self, scene_actors):
        """scene_actors: list of dicts matching payload schema in streamed-stirring-turing.md."""
        msg = PredictedObjects()
        msg.header.frame_id = "map"

        for actor in scene_actors:
            obj = PredictedObject()
            obj.object_id = UUID()
            obj.object_id.uuid = list(
                uuid.uuid5(uuid.NAMESPACE_DNS, str(actor["id"])).bytes
            )
            obj.existence_probability = 1.0

            pos_ros = BridgeHelper.carla2ros_pose(
                actor["x"], actor["y"], actor["z"],
                np.deg2rad(actor["roll"]),
                np.deg2rad(actor["pitch"]),
                np.deg2rad(actor["yaw"]),
                to_quat=True,
            )
            vel_ros = BridgeHelper.carla2ros_pose(
                actor["vx"], actor["vy"], actor["vz"],
                np.deg2rad(actor["ang_vx"]),
                np.deg2rad(actor["ang_vy"]),
                np.deg2rad(actor["ang_vz"]),
                to_quat=False,
            )
            acc_ros = BridgeHelper.carla2ros_pose(
                actor["ax"], actor["ay"], actor["az"],
                0.0, 0.0, 0.0,
                to_quat=False,
            )
            vel_norm = float(np.linalg.norm([actor["vx"], actor["vy"]]))
            obj_yaw = -np.deg2rad(actor["yaw"])

            kin = obj.kinematics
            kin.initial_pose_with_covariance.pose.position = Point(**pos_ros["position"])
            kin.initial_pose_with_covariance.pose.orientation = Quaternion(
                **pos_ros["orientation"]
            )
            kin.initial_twist_with_covariance.twist.linear.x = vel_norm
            kin.initial_twist_with_covariance.twist.angular = Vector3(
                x=vel_ros["orientation"]["roll"],
                y=vel_ros["orientation"]["pitch"],
                z=vel_ros["orientation"]["yaw"],
            )
            kin.initial_acceleration_with_covariance.accel.linear = Vector3(
                **acc_ros["position"]
            )

            type_id = actor["type"]
            if "vehicle" in type_id:
                cls = ObjectClassification()
                cls.label = ObjectClassification.CAR
                cls.probability = 1.0
                obj.classification.append(cls)
                kin.predicted_paths.append(
                    self._predict_vehicle_bicycle(actor, pos_ros, vel_norm, obj_yaw)
                )
                obj.shape = self._bbox(actor, Shape.BOUNDING_BOX)
            elif "walker" in type_id:
                cls = ObjectClassification()
                cls.label = ObjectClassification.PEDESTRIAN
                cls.probability = 1.0
                obj.classification.append(cls)
                kin.predicted_paths.append(
                    self._predict_constant_velocity(pos_ros, vel_norm, obj_yaw)
                )
                obj.shape = self._bbox(actor, Shape.CYLINDER)
            else:
                cls = ObjectClassification()
                cls.label = ObjectClassification.UNKNOWN
                cls.probability = 1.0
                obj.classification.append(cls)
                kin.predicted_paths.append(
                    self._predict_constant_velocity(pos_ros, vel_norm, obj_yaw)
                )
                obj.shape = self._bbox(actor, Shape.BOUNDING_BOX)

            msg.objects.append(obj)

        return msg

    def _bbox(self, actor, shape_type):
        shape = Shape()
        shape.type = shape_type
        ex, ey, ez = actor["bbox_extent"]
        shape.dimensions = Vector3(x=ex * 2.0, y=ey * 2.0, z=ez * 2.0)
        yaw = np.deg2rad(actor["yaw"])
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        corners_local = [(ex, ey), (-ex, ey), (-ex, -ey), (ex, -ey)]
        footprint = []
        for lx, ly in corners_local:
            wx = actor["x"] + lx * cos_y - ly * sin_y
            wy = actor["y"] + lx * sin_y + ly * cos_y
            ros = BridgeHelper.carla2ros_pose(wx, wy, actor["z"], 0.0, 0.0, 0.0)
            p = ros["position"]
            footprint.append(Point32(x=p["x"], y=p["y"], z=p["z"]))
        shape.footprint.points = footprint
        return shape

    def _predict_vehicle_bicycle(self, actor, pos_ros, vel, yaw):
        path = PredictedPath()
        path.time_step = self._ros_duration
        path.confidence = 1.0

        steer_l = actor.get("wheel_steer_l_deg", 0.0)
        steer_r = actor.get("wheel_steer_r_deg", 0.0)
        wheel_steering = np.deg2rad(-(steer_l + steer_r) / 2.0)
        actor_id = actor["id"]
        if actor_id not in self._steering_angle_history:
            self._steering_angle_history[actor_id] = deque(maxlen=self._moving_average_window)
        self._steering_angle_history[actor_id].append(wheel_steering)
        wheel_steering = (
            sum(self._steering_angle_history[actor_id])
            / len(self._steering_angle_history[actor_id])
        )

        pos = {k: dict(v) if isinstance(v, dict) else v for k, v in pos_ros.items()}
        t = 0.0
        while t < self._prediction_horizon:
            pos["position"]["x"] += vel * np.cos(yaw) * self._timestep
            pos["position"]["y"] += vel * np.sin(yaw) * self._timestep
            yaw += vel / self._wheelbase * np.tan(wheel_steering) * self._timestep
            pose = Pose()
            pose.position = Point(**pos["position"])
            quat = Rotation.from_euler("z", yaw).as_quat()
            pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            path.path.append(pose)
            t += self._timestep
        return path

    def _predict_constant_velocity(self, pos_ros, vel, yaw):
        path = PredictedPath()
        path.time_step = self._ros_duration
        path.confidence = 1.0
        pos = {k: dict(v) if isinstance(v, dict) else v for k, v in pos_ros.items()}
        t = 0.0
        while t < self._prediction_horizon:
            pos["position"]["x"] += vel * np.cos(yaw) * self._timestep
            pos["position"]["y"] += vel * np.sin(yaw) * self._timestep
            pose = Pose()
            pose.position = Point(**pos["position"])
            pose.orientation = Quaternion(**pos_ros["orientation"])
            path.path.append(pose)
            t += self._timestep
        return path


CARLA_TO_AW_TL = {
    0: TrafficLightElement.RED,
    1: TrafficLightElement.AMBER,
    2: TrafficLightElement.GREEN,
    3: TrafficLightElement.UNKNOWN,
}


class TrafficLightStateTracker:
    """Port of AutowareConverter.get_current_traffic_light_state for payload-driven bridge."""

    def __init__(self):
        self._last_tl = TrafficLightElement.RED
        self._steps_since_last_tl = 0

    def step(self, at_tl, carla_state):
        if at_tl:
            self._steps_since_last_tl = 0
            self._last_tl = CARLA_TO_AW_TL.get(carla_state, TrafficLightElement.RED)
            return self._last_tl
        if self._last_tl != TrafficLightElement.GREEN:
            self._last_tl = TrafficLightElement.RED
            return TrafficLightElement.RED
        self._steps_since_last_tl += 1
        if self._steps_since_last_tl > 30:
            self._steps_since_last_tl = 0
            self._last_tl = TrafficLightElement.RED
            return TrafficLightElement.RED
        return TrafficLightElement.GREEN
