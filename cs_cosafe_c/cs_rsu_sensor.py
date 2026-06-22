"""CoSafe sensor management: RSU and vehicle camera units.

Spawns independent 6-camera rigs for RSU (world-fixed) and per-vehicle sensing.
Each unit has its own SensorInterface to avoid collisions with ADS agent sensors.
"""

import carla
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.envs.sensor_interface import SensorInterface, CallBack

from cs_cosafe_c.cs_config import CsCoSafeConfig, CsCameraConfig


class CsRsuSensorUnit:
    """Virtual RSU: 6 RGB cameras at a fixed world location (not attached to any actor)."""

    def __init__(self, config: CsCameraConfig, rsu_id: int = 0):
        self._config = config
        self._id = rsu_id
        self.sensor_interface = SensorInterface()
        self._sensors_list = []
        self._spawn_location = None
        self.deleted = False

    def _get_camera_specs(self):
        """Return list of camera spec dicts for all 6 cameras."""
        specs = []
        for i, (name, yaw) in enumerate(zip(self._config.camera_names, self._config.rsu_yaws)):
            specs.append({
                'type': 'sensor.camera.rgb',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': self._config.rsu_pitch,
                'yaw': yaw,
                'width': self._config.rsu_width,
                'height': self._config.rsu_height,
                'fov': self._config.rsu_fovs[i],
                'id': name,
            })
        return specs

    def spawn(self, location: carla.Location):
        """Spawn 6 cameras at a fixed world location (no parent actor)."""
        self._spawn_location = location
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()

        for spec in self._get_camera_specs():
            bp = bp_library.find(spec['type'])
            bp.set_attribute('image_size_x', str(spec['width']))
            bp.set_attribute('image_size_y', str(spec['height']))
            bp.set_attribute('fov', str(spec['fov']))
            if self._config.sensor_tick > 0:
                bp.set_attribute('sensor_tick', str(self._config.sensor_tick))

            transform = carla.Transform(
                carla.Location(
                    x=location.x + spec['x'],
                    y=location.y + spec['y'],
                    z=location.z + spec['z'],
                ),
                carla.Rotation(
                    roll=spec['roll'],
                    pitch=spec['pitch'],
                    yaw=spec['yaw'],
                ),
            )

            sensor = world.spawn_actor(bp, transform)
            tag = 'cs_rsu_{}_{}'.format(self._id, spec['id'])
            sensor.listen(CallBack(tag, spec['type'], sensor, self.sensor_interface))
            self._sensors_list.append(sensor)

    def tick(self):
        """Collect data from all RSU cameras. Returns dict: tag -> (timestamp, data)."""
        if self.deleted:
            return {}
        return self.sensor_interface.get_data()

    def cleanup(self):
        """Stop and destroy all RSU sensors."""
        self.deleted = True
        for sensor in self._sensors_list:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensors_list = []


class CsVehicleSensorUnit:
    """6 RGB cameras attached to an ego vehicle."""

    def __init__(self, config: CsCameraConfig, vehicle_id: int = 0):
        self._config = config
        self._vehicle_id = vehicle_id
        self.sensor_interface = SensorInterface()
        self._sensors_list = []
        self.deleted = False

    def _get_camera_specs(self):
        """Return list of camera spec dicts for all 6 vehicle cameras."""
        specs = []
        for i, (name, yaw) in enumerate(zip(self._config.camera_names, self._config.veh_yaws)):
            px, py, pz = self._config.veh_positions[i]
            specs.append({
                'type': 'sensor.camera.rgb',
                'x': px,
                'y': py,
                'z': pz,
                'roll': 0.0,
                'pitch': self._config.veh_pitch,
                'yaw': yaw,
                'width': self._config.veh_width,
                'height': self._config.veh_height,
                'fov': self._config.veh_fovs[i],
                'id': name,
            })
        return specs

    def spawn(self, vehicle: carla.Actor):
        """Attach 6 cameras to the given vehicle actor."""
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()

        for spec in self._get_camera_specs():
            bp = bp_library.find(spec['type'])
            bp.set_attribute('image_size_x', str(spec['width']))
            bp.set_attribute('image_size_y', str(spec['height']))
            bp.set_attribute('fov', str(spec['fov']))
            if self._config.sensor_tick > 0:
                bp.set_attribute('sensor_tick', str(self._config.sensor_tick))

            transform = carla.Transform(
                carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                carla.Rotation(roll=spec['roll'], pitch=spec['pitch'], yaw=spec['yaw']),
            )

            sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
            tag = 'cs_veh_{}_{}'.format(self._vehicle_id, spec['id'])
            sensor.listen(CallBack(tag, spec['type'], sensor, self.sensor_interface))
            self._sensors_list.append(sensor)

    def tick(self):
        """Collect data from all vehicle cameras. Returns dict: tag -> (timestamp, data)."""
        if self.deleted:
            return {}
        return self.sensor_interface.get_data()

    def cleanup(self):
        """Stop and destroy all vehicle sensors."""
        self.deleted = True
        for sensor in self._sensors_list:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensors_list = []


class CsSensorManager:
    """Top-level manager for all CoSafe sensors (RSU + per-vehicle cameras)."""

    def __init__(self, config: CsCoSafeConfig):
        self._config = config
        self._rsu_unit = None
        self._vehicle_units = {}

    def _find_rsu_location(self, ego_vehicles):
        """Find RSU placement location.

        Strategy: try junction center first (via CARLA junction API),
        fallback to road-side location ahead of ego 0.
        """
        carla_map = CarlaDataProvider.get_map()

        if self._config.rsu.use_junction_center and ego_vehicles:
            vehicle = ego_vehicles[0]
            if vehicle and vehicle.is_alive:
                wp = carla_map.get_waypoint(vehicle.get_location())
                search_dist = 0.0
                current_wp = wp
                while search_dist < 100.0:
                    if current_wp.is_junction:
                        junction = current_wp.get_junction()
                        bb = junction.bounding_box
                        return carla.Location(
                            x=bb.location.x,
                            y=bb.location.y,
                            z=bb.location.z + self._config.rsu.height,
                        )
                    next_wps = current_wp.next(2.0)
                    if not next_wps:
                        break
                    current_wp = next_wps[0]
                    search_dist += 2.0

        if ego_vehicles and ego_vehicles[0] and ego_vehicles[0].is_alive:
            vehicle = ego_vehicles[0]
            loc = vehicle.get_location()
            wp = carla_map.get_waypoint(loc)
            while wp.get_right_lane():
                wp = wp.get_right_lane()
            next_wps = wp.next(self._config.rsu.distance)
            if next_wps:
                spawn_loc = next_wps[0].transform.location
            else:
                spawn_loc = wp.transform.location
            return carla.Location(
                x=spawn_loc.x,
                y=spawn_loc.y,
                z=spawn_loc.z + self._config.rsu.height,
            )

        return carla.Location(x=0.0, y=0.0, z=self._config.rsu.height)

    def spawn_rsu(self, ego_vehicles):
        """Spawn RSU cameras at an intersection or road-side location."""
        location = self._find_rsu_location(ego_vehicles)
        self._rsu_unit = CsRsuSensorUnit(self._config.camera, rsu_id=0)
        self._rsu_unit.spawn(location)
        print("[CoSafe] RSU spawned at ({:.1f}, {:.1f}, {:.1f})".format(
            location.x, location.y, location.z))

    def spawn_vehicle_sensors(self, ego_vehicles):
        """Spawn 6 cameras for the closest vehicles to RSU.

        Limited to max_vehicle_cameras (default 4) vehicles, matching
        the BEVerse model's 5-agent limit (4 vehicles + 1 RSU).
        Vehicles are sorted by distance to RSU; only the closest are equipped.
        """
        max_veh = self._config.camera.max_vehicle_cameras
        rsu_loc = self.rsu_location

        candidates = []
        for vid, vehicle in enumerate(ego_vehicles):
            if vehicle is not None and vehicle.is_alive:
                if rsu_loc is not None:
                    dist = vehicle.get_location().distance(rsu_loc)
                else:
                    dist = 0.0
                candidates.append((vid, vehicle, dist))
        candidates.sort(key=lambda x: x[2])
        selected = candidates[:max_veh]

        for vid, vehicle, dist in selected:
            unit = CsVehicleSensorUnit(self._config.camera, vehicle_id=vid)
            unit.spawn(vehicle)
            self._vehicle_units[vid] = unit
            print("[CoSafe] Vehicle {} sensors spawned (dist={:.1f}m)".format(
                vid, dist))
        if len(candidates) > max_veh:
            skipped = [c[0] for c in candidates[max_veh:]]
            print("[CoSafe] Skipped vehicles {} (over max_vehicle_cameras={})".format(
                skipped, max_veh))

    def tick_all(self):
        """Collect data from all CoSafe sensors.

        Returns:
            dict: {'rsu': {tag: (ts, data)}, 'vehicles': {vid: {tag: (ts, data)}}}
        """
        result = {
            'rsu': {},
            'vehicles': {},
        }
        if self._rsu_unit and not self._rsu_unit.deleted:
            try:
                result['rsu'] = self._rsu_unit.tick()
            except Exception:
                pass

        for vid, unit in self._vehicle_units.items():
            if not unit.deleted:
                try:
                    result['vehicles'][vid] = unit.tick()
                except Exception:
                    pass

        return result

    def cleanup_vehicle(self, vehicle_id):
        """Cleanup sensors for a specific vehicle."""
        if vehicle_id in self._vehicle_units:
            self._vehicle_units[vehicle_id].cleanup()
            del self._vehicle_units[vehicle_id]

    @property
    def rsu_location(self):
        """Return the RSU spawn location, or None if not spawned."""
        if self._rsu_unit and self._rsu_unit._spawn_location:
            return self._rsu_unit._spawn_location
        return None

    def cleanup(self):
        """Cleanup all CoSafe sensors."""
        if self._rsu_unit:
            self._rsu_unit.cleanup()
            self._rsu_unit = None
        for vid in list(self._vehicle_units.keys()):
            self._vehicle_units[vid].cleanup()
        self._vehicle_units = {}
