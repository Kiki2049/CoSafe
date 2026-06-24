#!/usr/bin/env python3
"""Autoware <-> CoLMDriver HTTP bridge.

Runs inside the Autoware Docker container (Python 3.10 + ROS2 Humble).
Replaces the in-process ``AutowarePriviligedAgent`` (which required ``rclpy``
alongside CARLA 0.9.10 + Python 3.7 -- impossible to co-install).

Responsibilities
----------------
- Spin a ROS2 node with the same publishers/subscribers as
  ``autoware_carla_leaderboard/src/tum_agents/autoware_agent/aw_priviliged.py``.
- Expose HTTP endpoints (Flask):
    POST /reset  -> receive global_plan_world, reset route bookkeeping.
    POST /step   -> receive one-tick ego state + scene actors + traffic-light,
                    publish ROS topics, wait up to 50 ms for a new control,
                    return it as carla.VehicleControl fields.
    GET  /health -> readiness probe.

CLI
---
    python3 autoware_bridge.py --host 0.0.0.0 --port 8100

One bridge per Autoware stack / per ego vehicle.  DDS isolation is achieved by
setting ``ROS_DOMAIN_ID`` per container (see docker/autoware/docker-compose.yaml).
"""

from __future__ import annotations

import argparse
import math
import queue
import re
import threading
import time

import numpy as np
import rclpy
from flask import Flask, jsonify, request
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock

from autoware_adapi_v1_msgs.msg import (
    LocalizationInitializationState,
    OperationModeState,
)
from autoware_adapi_v1_msgs.srv import SetRoutePoints
from autoware_control_msgs.msg import Control
from autoware_perception_msgs.msg import (
    TrafficLightElement,
    TrafficLightGroup,
    TrafficLightGroupArray,
)
from autoware_vehicle_msgs.msg import Engage, SteeringReport, VelocityReport
from geometry_msgs.msg import (
    AccelWithCovarianceStamped,
    Point,
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray

from bridge_converter import (
    AutowareToCarlaControl,
    BridgeHelper,
    PayloadPredictedObjectsBuilder,
    TrafficLightStateTracker,
)


EPSILON = 0.001
CONTROL_WAIT_SEC = 0.05


class AutowareBridge:
    def __init__(self, ros_node_name="autoware_bridge"):
        rclpy.init(args=None)
        self.node = rclpy.create_node(ros_node_name)

        self._clock_pub = self.node.create_publisher(
            Clock, "/clock",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._localization_pub = self.node.create_publisher(
            Odometry, "/localization/kinematic_state",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._tf_pub = self.node.create_publisher(
            TFMessage, "/tf",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._loc_init_state_pub = self.node.create_publisher(
            LocalizationInitializationState,
            "/localization/initialization_state",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._loc_acc_pub = self.node.create_publisher(
            AccelWithCovarianceStamped, "/localization/acceleration",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._imu_pub = self.node.create_publisher(
            Imu, "/sensing/imu/imu_data",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._ndt_pose_pub = self.node.create_publisher(
            PoseWithCovarianceStamped,
            "/localization/pose_estimator/pose_with_covariance",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._initpose_pub = self.node.create_publisher(
            PoseWithCovarianceStamped, "/initialpose",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._gnss_pose_pub = self.node.create_publisher(
            PoseWithCovarianceStamped,
            "/sensing/gnss/pose_with_covariance",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._steering_pub = self.node.create_publisher(
            SteeringReport, "/vehicle/status/steering_status",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._velocity_pub = self.node.create_publisher(
            VelocityReport, "/vehicle/status/velocity_status",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._goal_pub = self.node.create_publisher(
            PoseStamped, "/rviz/routing/rough_goal",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._engage_pub = self.node.create_publisher(
            Engage, "/autoware/engage",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        from autoware_perception_msgs.msg import PredictedObjects
        self._objects_pub = self.node.create_publisher(
            PredictedObjects, "/perception/object_recognition/objects",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self._tl_pub = self.node.create_publisher(
            TrafficLightGroupArray,
            "/perception/traffic_light_recognition/traffic_signals",
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )

        self.node.create_subscription(
            AccelWithCovarianceStamped, "/localization/acceleration",
            self._acc_cb,
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self.node.create_subscription(
            Control, "/control/command/control_cmd", self._ctrl_cb,
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self.node.create_subscription(
            OperationModeState, "/system/operation_mode/state", self._opmode_cb,
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE),
        )
        self.node.create_subscription(
            MarkerArray, "/map/vector_map_marker", self._lanelet_cb,
            qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self._route_srv = self.node.create_client(
            SetRoutePoints, "/api/routing/change_route_points"
        )

        self._control_converter = AutowareToCarlaControl()
        self._objects_builder = PayloadPredictedObjectsBuilder()
        self._tl_tracker = TrafficLightStateTracker()

        self._control_queue: queue.Queue = queue.Queue(maxsize=1)
        self._last_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
        self._last_acc = 0.0
        self._is_autonomous = False
        self._traffic_light_ids: set[int] = set()

        self._global_plan_world: list[list[float]] = []
        self._route_index = 1
        self._published_latest = False
        self._goal_mod_failure_counter = 0

        self._state_lock = threading.Lock()

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self._spin_thread.start()

    def _acc_cb(self, msg: AccelWithCovarianceStamped):
        self._last_acc = msg.accel.accel.linear.x

    def _opmode_cb(self, msg: OperationModeState):
        self._is_autonomous = msg.mode == OperationModeState.AUTONOMOUS

    def _lanelet_cb(self, msg: MarkerArray):
        pattern = re.compile(r"TLRegElemId:(\d+)")
        for marker in msg.markers:
            if marker.text:
                m = pattern.search(marker.text)
                if m:
                    self._traffic_light_ids.add(int(m.group(1)))

    def _ctrl_cb(self, msg: Control):
        try:
            self._control_queue.get_nowait()
        except queue.Empty:
            pass
        self._control_queue.put_nowait(msg)

    def reset(self, payload):
        with self._state_lock:
            self._global_plan_world = list(payload.get("global_plan_world", []))
            self._route_index = 1
            self._published_latest = False
            self._goal_mod_failure_counter = 0
            self._last_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
            while not self._control_queue.empty():
                try:
                    self._control_queue.get_nowait()
                except queue.Empty:
                    break
        return {"ok": True}

    def step(self, payload):
        timestamp = float(payload["timestamp"])
        time_sec = int(timestamp)
        time_nsec = int((timestamp - time_sec) * 1e9)
        ego = payload["ego"]

        self._publish_clock(time_sec, time_nsec)
        pos_ros = self._publish_pose(ego, time_sec, time_nsec)
        velocity_x = self._publish_velocity(ego, time_sec, time_nsec)
        self._publish_acceleration(ego, time_sec, time_nsec)
        self._publish_steering(ego, time_sec, time_nsec)
        self._publish_imu(ego, time_sec, time_nsec)
        self._publish_objects(payload.get("scene_actors", []), time_sec, time_nsec)
        self._publish_traffic_lights(payload.get("traffic_light", {}), time_sec, time_nsec)
        self._maybe_publish_goal(pos_ros)

        control = self._wait_for_control(velocity_x)

        return {
            "ok": True,
            "steer": control["steer"],
            "throttle": control["throttle"],
            "brake": control["brake"],
            "is_autonomous": self._is_autonomous,
        }

    def _publish_clock(self, sec, nsec):
        msg = Clock()
        msg.clock.sec = sec
        msg.clock.nanosec = nsec
        self._clock_pub.publish(msg)

    def _publish_pose(self, ego, sec, nsec):
        pos_ros = BridgeHelper.carla2ros_pose(
            ego["x"], ego["y"], ego["z"],
            math.radians(ego["roll"]), math.radians(ego["pitch"]),
            math.radians(ego["yaw"]), to_quat=True,
        )

        tf_msg = TFMessage()
        ts = TransformStamped()
        ts.header.stamp.sec = sec
        ts.header.stamp.nanosec = nsec
        ts.header.frame_id = "map"
        ts.child_frame_id = "base_link"
        ts.transform.translation = Vector3(**pos_ros["position"])
        ts.transform.rotation = Quaternion(**pos_ros["orientation"])
        tf_msg.transforms.append(ts)
        self._tf_pub.publish(tf_msg)

        vel_local = _rotate_map_to_vehicle(
            np.array([ego["vx"], ego["vy"], ego["vz"]]),
            ego["roll"], ego["pitch"], ego["yaw"],
        )
        ang_vel_ros = BridgeHelper.carla2ros_pose(
            ego["ang_vx"], ego["ang_vy"], ego["ang_vz"],
            math.radians(ego["ang_vx"]), math.radians(ego["ang_vy"]),
            math.radians(ego["ang_vz"]), to_quat=False,
        )

        odom = Odometry()
        odom.header.stamp.sec = sec
        odom.header.stamp.nanosec = nsec
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position = Point(**pos_ros["position"])
        odom.pose.pose.orientation = Quaternion(**pos_ros["orientation"])
        vel_rotated = BridgeHelper.carla2ros_pose(
            float(vel_local[0]), float(vel_local[1]), float(vel_local[2]),
            0.0, 0.0, 0.0, to_quat=False,
        )
        odom.twist.twist.linear = Vector3(**vel_rotated["position"])
        odom.twist.twist.angular = Vector3(
            x=ang_vel_ros["orientation"]["roll"],
            y=ang_vel_ros["orientation"]["pitch"],
            z=ang_vel_ros["orientation"]["yaw"],
        )
        self._localization_pub.publish(odom)

        init_state = LocalizationInitializationState()
        init_state.stamp.sec = sec
        init_state.stamp.nanosec = nsec
        init_state.state = LocalizationInitializationState.INITIALIZED
        self._loc_init_state_pub.publish(init_state)

        return pos_ros

    def _publish_velocity(self, ego, sec, nsec):
        vel_local = _rotate_map_to_vehicle(
            np.array([ego["vx"], ego["vy"], ego["vz"]]),
            ego["roll"], ego["pitch"], ego["yaw"],
        )
        vel_ros = BridgeHelper.carla2ros_pose(
            float(vel_local[0]), float(vel_local[1]), float(vel_local[2]),
            0.0, 0.0, 0.0, to_quat=False,
        )
        msg = VelocityReport()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = "base_link"
        msg.longitudinal_velocity = float(vel_ros["position"]["x"])
        self._velocity_pub.publish(msg)
        return float(vel_ros["position"]["x"])

    def _publish_acceleration(self, ego, sec, nsec):
        acc_local = _rotate_map_to_vehicle(
            np.array([ego["ax"], ego["ay"], ego["az"]]),
            ego["roll"], ego["pitch"], ego["yaw"],
        )
        acc_ros = BridgeHelper.carla2ros_pose(
            float(acc_local[0]), float(acc_local[1]), float(acc_local[2]),
            0.0, 0.0, 0.0, to_quat=False,
        )
        msg = AccelWithCovarianceStamped()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = "base_link"
        msg.accel.accel.linear = Vector3(**acc_ros["position"])
        self._loc_acc_pub.publish(msg)

    def _publish_steering(self, ego, sec, nsec):
        left = ego.get("steering_l_deg", 0.0)
        right = ego.get("steering_r_deg", 0.0)
        steer_rad = math.radians(-(left + right) / 2.0)
        msg = SteeringReport()
        msg.stamp.sec = sec
        msg.stamp.nanosec = nsec
        msg.steering_tire_angle = float(steer_rad)
        self._steering_pub.publish(msg)

    def _publish_imu(self, ego, sec, nsec):
        msg = Imu()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = "imu_link"
        msg.angular_velocity = Vector3(
            x=math.radians(ego["ang_vx"]),
            y=math.radians(ego["ang_vy"]),
            z=math.radians(ego["ang_vz"]),
        )
        msg.linear_acceleration = Vector3(
            x=float(ego["ax"]), y=float(ego["ay"]), z=float(ego["az"])
        )
        self._imu_pub.publish(msg)

    def _publish_objects(self, scene_actors, sec, nsec):
        msg = self._objects_builder.build(scene_actors)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        self._objects_pub.publish(msg)

    def _publish_traffic_lights(self, tl_payload, sec, nsec):
        at_tl = bool(tl_payload.get("at_tl", False))
        state = int(tl_payload.get("state", 0))
        aw_state = self._tl_tracker.step(at_tl, state)
        msg = TrafficLightGroupArray()
        msg.stamp.sec = sec
        msg.stamp.nanosec = nsec
        for tl_id in self._traffic_light_ids:
            group = TrafficLightGroup()
            group.traffic_light_group_id = tl_id
            element = TrafficLightElement()
            element.color = aw_state
            element.shape = 1
            element.status = 2
            element.confidence = 1.0
            group.elements.append(element)
            msg.traffic_light_groups.append(group)
        self._tl_pub.publish(msg)

    def _maybe_publish_goal(self, current_pos_ros):
        with self._state_lock:
            if not self._global_plan_world:
                return
            plan = self._global_plan_world
            idx = self._route_index
        if idx >= len(plan) - 1:
            if not self._published_latest:
                self._send_route_request(plan[-1])
            return

        goal_wp = plan[idx]
        goal_ros = BridgeHelper.carla2ros_pose(
            goal_wp[0], goal_wp[1], goal_wp[2],
            math.radians(goal_wp[3]), math.radians(goal_wp[4]),
            math.radians(goal_wp[5]), to_quat=True,
        )
        cur = current_pos_ros["position"]
        gp = goal_ros["position"]
        dist = math.sqrt(
            (gp["x"] - cur["x"]) ** 2 + (gp["y"] - cur["y"]) ** 2 + (gp["z"] - cur["z"]) ** 2
        )
        if dist < 50.0 or self._goal_mod_failure_counter > 5:
            with self._state_lock:
                self._route_index += 1
                self._published_latest = False
                self._goal_mod_failure_counter = 0

        if not self._published_latest:
            self._send_route_request(plan[self._route_index])

    def _send_route_request(self, wp_world):
        next_ros = BridgeHelper.carla2ros_pose(
            wp_world[0], wp_world[1], wp_world[2],
            math.radians(wp_world[3]), math.radians(wp_world[4]),
            math.radians(wp_world[5]), to_quat=True,
        )
        req = SetRoutePoints.Request()
        req.header.frame_id = "map"
        req.option.allow_goal_modification = True
        req.option.allow_while_using_route = False
        req.goal = Pose(
            position=Point(**next_ros["position"]),
            orientation=Quaternion(**next_ros["orientation"]),
        )
        if not self._route_srv.service_is_ready():
            return
        future = self._route_srv.call_async(req)
        future.add_done_callback(self._on_route_result)

    def _on_route_result(self, future):
        try:
            result = future.result()
            self._published_latest = bool(result.status.success)
            if not self._published_latest:
                self._goal_mod_failure_counter += 1
            if result.status.message == "The route is not set yet.":
                self._published_latest = True
                with self._state_lock:
                    idx = self._route_index
                    plan = self._global_plan_world
                if idx < len(plan):
                    wp = plan[idx]
                    next_ros = BridgeHelper.carla2ros_pose(
                        wp[0], wp[1], wp[2],
                        math.radians(wp[3]), math.radians(wp[4]),
                        math.radians(wp[5]), to_quat=True,
                    )
                    ps = PoseStamped()
                    ps.header.frame_id = "map"
                    ps.pose = Pose(
                        position=Point(**next_ros["position"]),
                        orientation=Quaternion(**next_ros["orientation"]),
                    )
                    self._goal_pub.publish(ps)
        except Exception as exc:
            print("[autoware_bridge] route service call failed:", exc, flush=True)

    def _wait_for_control(self, velocity_x):
        try:
            msg: Control = self._control_queue.get(True, CONTROL_WAIT_SEC)
        except queue.Empty:
            return self._last_control

        steer = self._control_converter.convertLatAwToCarla(
            math.degrees(msg.lateral.steering_tire_angle)
        )
        throttle, brake = self._control_converter.updateThrottleBrake(
            msg.longitudinal.acceleration, self._last_acc, velocity_x
        )
        self._last_control = {"steer": float(steer), "throttle": throttle, "brake": brake}
        return self._last_control

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()


def _rotate_map_to_vehicle(vec_map, roll_deg, pitch_deg, yaw_deg):
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R.T @ vec_map


def create_app(bridge: AutowareBridge) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def _health():
        return jsonify({"ok": True})

    @app.post("/reset")
    def _reset():
        return jsonify(bridge.reset(request.get_json(force=True)))

    @app.post("/step")
    def _step():
        return jsonify(bridge.step(request.get_json(force=True)))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    bridge = AutowareBridge()
    app = create_app(bridge)
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
