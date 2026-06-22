import os
import pathlib
import sys
import time

import cv2
import numpy as np
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.autoagents.agent_wrapper import AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData


class CsScenarioManager(ScenarioManager):
    """ScenarioManager with CoSafe safety middleware interception.

    Overrides _tick_scenario to insert CoSafe RSU processing between
    ADS agent action production and vehicle control application.
    """

    COSAFE_DEBUG_SAVE_INTERVAL = 5

    def __init__(self, timeout, debug_mode=False):
        super().__init__(timeout, debug_mode)
        self._cosafe_rsu = None
        self._cosafe_sensor_manager = None
        self._cosafe_tick_count = 0
        self._cosafe_save_root = None

    def load_scenario(self, scenario, agent, rep_number, ego_vehicles_num,
                      save_root=None, sensor_tf_list=None, is_crazy=False):
        """Override to capture save_root and create cosafe debug directories."""
        super().load_scenario(scenario, agent, rep_number, ego_vehicles_num,
                              save_root=save_root, sensor_tf_list=sensor_tf_list,
                              is_crazy=is_crazy)

        cosafe_base = save_root or os.environ.get('SAVE_PATH')
        if cosafe_base is not None and self.COSAFE_DEBUG_SAVE_INTERVAL > 0:
            self._cosafe_save_root = pathlib.Path(cosafe_base) / 'cosafe'
            camera_names = [
                'Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight',
                'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight',
            ]
            for cam in camera_names:
                (self._cosafe_save_root / 'rsu' / cam).mkdir(
                    parents=True, exist_ok=True)
            for vid in range(ego_vehicles_num):
                for cam in camera_names:
                    (self._cosafe_save_root / 'veh_{}'.format(vid) / cam).mkdir(
                        parents=True, exist_ok=True)
            print("[CoSafe] Debug image save dir: {}".format(self._cosafe_save_root))

    def set_cosafe_rsu(self, rsu):
        """Set the CoSafe RSU module."""
        self._cosafe_rsu = rsu

    def set_cosafe_sensor_manager(self, sensor_manager):
        """Set the CoSafe sensor manager."""
        self._cosafe_sensor_manager = sensor_manager

    def _upload_vehicle_commands(self):
        """Extract driving commands and target points from ADS agent's
        route planner and upload to RSU.

        Both command and target_point come from the same route entry
        (route[vid][1]) to stay consistent with what ADS agent uses.

        target_point world coordinates are obtained via index correspondence:
        route_planner.route[vid] was initialized from _global_plan[vid],
        entries are popped as the vehicle progresses.  So:
            consumed = len(_global_plan[vid]) - len(route[vid])
            route[vid][1] <=> _global_plan_world_coord[vid][consumed + 1]
        """
        if self._cosafe_rsu is None or self._agent is None:
            return
        try:
            ads_agent = self._agent._agent
            route_planner = getattr(ads_agent, '_route_planner', None)
            if route_planner is None:
                return

            plans_gps = getattr(ads_agent, '_global_plan', None)
            plans_world = getattr(ads_agent, '_global_plan_world_coord', None)

            commands = {}
            targets = {}
            for vid in range(self.ego_vehicles_num):
                vehicle = CarlaDataProvider.get_hero_actor(hero_id=vid)
                if vehicle is None or not vehicle.is_alive:
                    continue
                try:
                    if not hasattr(route_planner, 'route') or vid >= len(route_planner.route):
                        continue
                    route = route_planner.route[vid]
                    if len(route) < 2:
                        continue

                    cmd = route[1][1]
                    cmd_val = cmd.value if hasattr(cmd, 'value') else int(cmd)
                    commands[vehicle.id] = cmd_val

                    if (plans_gps is not None and plans_world is not None
                            and vid < len(plans_gps) and vid < len(plans_world)):
                        original_len = len(plans_gps[vid])
                        current_len = len(route)
                        consumed = original_len - current_len
                        target_idx = consumed + 1
                        if target_idx < len(plans_world[vid]):
                            wp_tf = plans_world[vid][target_idx][0]
                            targets[vehicle.id] = (wp_tf.location.x,
                                                   wp_tf.location.y)
                except Exception:
                    pass

            ref_paths = {}
            plans_world_all = getattr(
                ads_agent, '_global_plan_world_coord_all', None)
            if plans_world_all is not None:
                for vid in range(self.ego_vehicles_num):
                    vehicle = CarlaDataProvider.get_hero_actor(hero_id=vid)
                    if vehicle is None or not vehicle.is_alive:
                        continue
                    try:
                        if vid >= len(plans_world_all):
                            continue
                        full_route = plans_world_all[vid]
                        if not full_route:
                            continue
                        veh_loc = vehicle.get_location()
                        vx, vy = veh_loc.x, veh_loc.y
                        best_idx = 0
                        best_dist = float('inf')
                        for i, (wp_tf, _road_opt) in enumerate(full_route):
                            dx = wp_tf.location.x - vx
                            dy = wp_tf.location.y - vy
                            d = dx * dx + dy * dy
                            if d < best_dist:
                                best_dist = d
                                best_idx = i
                        path_pts = []
                        for i in range(best_idx + 1, len(full_route)):
                            wp_tf = full_route[i][0]
                            path_pts.append(
                                (wp_tf.location.x, wp_tf.location.y))
                        if path_pts:
                            ref_paths[vehicle.id] = path_pts
                    except Exception:
                        pass

            if commands:
                self._cosafe_rsu.set_vehicle_commands(commands)
            if targets:
                self._cosafe_rsu.set_vehicle_targets(targets)
            if ref_paths:
                self._cosafe_rsu.set_vehicle_reference_paths(ref_paths)
        except Exception:
            pass

    def _debug_save_cosafe_images(self, sensor_data, frame):
        """Save CoSafe sensor images to disk for visual inspection.

        sensor_data structure from tick_all():
            {'rsu': {tag: (ts, bgra_array)}, 'vehicles': {vid: {tag: (ts, bgra_array)}}}
        Tags are like 'cs_rsu_0_Camera_Front', 'cs_veh_0_Camera_FrontLeft'.
        """
        if self._cosafe_save_root is None:
            return

        rsu_data = sensor_data.get('rsu', {})
        for tag, (ts, img) in rsu_data.items():
            parts = tag.split('_', 3)
            if len(parts) >= 4:
                camera_name = parts[3]
                bgr = img[:, :, :3]
                save_path = self._cosafe_save_root / 'rsu' / camera_name / (
                    '%04d.jpg' % frame)
                cv2.imwrite(str(save_path), bgr)

        vehicles_data = sensor_data.get('vehicles', {})
        for vid, veh_data in vehicles_data.items():
            for tag, (ts, img) in veh_data.items():
                parts = tag.split('_', 3)
                if len(parts) >= 4:
                    camera_name = parts[3]
                    bgr = img[:, :, :3]
                    save_path = (self._cosafe_save_root
                                 / 'veh_{}'.format(vid) / camera_name
                                 / ('%04d.jpg' % frame))
                    cv2.imwrite(str(save_path), bgr)

    def _tick_scenario(self, timestamp):
        """Run next tick of scenario with CoSafe interception.

        Replicates all original ScenarioManager._tick_scenario logic,
        inserting CoSafe sensor collection and RSU processing between
        ADS agent action and vehicle control application.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            for vehicle_num in range(self.ego_vehicles_num):
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) and \
                        not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    self._agent.del_ego_sensor(vehicle_num)
                    self._agent.cleanup_single(vehicle_num)
                    self._agent.cleanup_rsu(vehicle_num)
                    if self._cosafe_sensor_manager:
                        self._cosafe_sensor_manager.cleanup_vehicle(vehicle_num)
                    print("destroy ego type 0 : {}".format(vehicle_num))
                    CarlaDataProvider.remove_actor_by_id(
                        CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)

            try:
                ego_action = self._agent()

            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._cosafe_tick_count += 1
            if self._cosafe_rsu and self._cosafe_sensor_manager:
                try:
                    cosafe_sensor_data = self._cosafe_sensor_manager.tick_all()
                except Exception:
                    cosafe_sensor_data = {}
                if (self.COSAFE_DEBUG_SAVE_INTERVAL > 0
                        and self._cosafe_tick_count
                        % self.COSAFE_DEBUG_SAVE_INTERVAL == 0):
                    try:
                        self._debug_save_cosafe_images(
                            cosafe_sensor_data, self._cosafe_tick_count)
                    except Exception as e:
                        print("[CoSafe] Debug save error: {}".format(e))
                self._upload_vehicle_commands()
                final_actions = self._cosafe_rsu.step(
                    cosafe_sensor_data, ego_action, None)
            else:
                final_actions = ego_action

            for vehicle_num in range(self.ego_vehicles_num):
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) and \
                        not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    self._agent.del_ego_sensor(vehicle_num)
                    self._agent.cleanup_single(vehicle_num)
                    self._agent.cleanup_rsu(vehicle_num)
                    if self._cosafe_sensor_manager:
                        self._cosafe_sensor_manager.cleanup_vehicle(vehicle_num)
                    print("destroy ego type 1 : {}".format(vehicle_num))
                    CarlaDataProvider.remove_actor_by_id(
                        CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)

            for vehicle_num in range(self.ego_vehicles_num):
                try:
                    ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if ego:
                        if ego.is_alive:
                            self.ego_vehicles[vehicle_num].apply_control(
                                final_actions[vehicle_num])
                except:
                    pass

            for vehicle_num in range(self.ego_vehicles_num):
                try:
                    ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                    if ego and ego.is_alive:
                        self.scenario_tree[vehicle_num].tick_once()
                except:
                    pass

            if self._debug_mode:
                print("\n")
                for vehicle_num in range(self.ego_vehicles_num):
                    try:
                        ego = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
                        if ego and ego.is_alive:
                            py_trees.display.print_ascii_tree(
                                self.scenario_tree[vehicle_num], show_status=True)
                            sys.stdout.flush()
                    except:
                        pass

            stop_flag = 0
            for vehicle_num in range(self.ego_vehicles_num):
                if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num) is None:
                    stop_flag += 1
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        self._agent.del_ego_sensor(vehicle_num)
                        self._agent.cleanup_single(vehicle_num)
                        self._agent.cleanup_rsu(vehicle_num)
                        if self._cosafe_sensor_manager:
                            self._cosafe_sensor_manager.cleanup_vehicle(vehicle_num)
                        print("destroy ego type 2 {}".format(vehicle_num))
                        CarlaDataProvider.remove_actor_by_id(
                            CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)
                    if stop_flag == self.ego_vehicles_num:
                        self._running = False

                elif self.scenario_tree[vehicle_num].status != py_trees.common.Status.RUNNING \
                        or not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive:
                    stop_flag += 1
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        self._agent.del_ego_sensor(vehicle_num)
                        self._agent.cleanup_single(vehicle_num)
                        self._agent.cleanup_rsu(vehicle_num)
                        if self._cosafe_sensor_manager:
                            self._cosafe_sensor_manager.cleanup_vehicle(vehicle_num)
                        print("destroy ego type 3 {}".format(vehicle_num))
                        print('flag1:', self.scenario_tree[vehicle_num].status != py_trees.common.Status.RUNNING)
                        print('flag2:', not CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).is_alive)
                        CarlaDataProvider.remove_actor_by_id(
                            CarlaDataProvider.get_hero_actor(hero_id=vehicle_num).id)
                    if stop_flag == self.ego_vehicles_num:
                        self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            if CarlaDataProvider.get_hero_actor(hero_id=0):
                ego_trans = CarlaDataProvider.get_hero_actor(hero_id=0).get_transform()
                self.prev_ego_trans = ego_trans
            else:
                for vehicle_num in range(1, self.ego_vehicles_num):
                    if CarlaDataProvider.get_hero_actor(hero_id=vehicle_num):
                        ego_trans = self.ego_vehicles[vehicle_num].get_transform()
                        self.prev_ego_trans = ego_trans
                        break
                ego_trans = self.prev_ego_trans
            spectator.set_transform(carla.Transform(
                ego_trans.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)
