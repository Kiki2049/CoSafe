"""BEV instance ↔ CARLA vehicle matching.

Projects CARLA vehicle world positions into BEV pixel space and uses
Hungarian matching to associate predicted BEV instance centers with
actual vehicles.

References:
- BEV transforms: cs_cosafe/perception/cs_bev_transform.py
- Instance extraction: cs_cosafe/perception/cs_instance_utils.py
  (predict_instance_segmentation_and_trajectories)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from cs_cosafe_c.cs_config import CsBevConfig
from cs_cosafe_c.perception.cs_bev_transform import world_to_bev, get_grid_conf


class CsInstanceTracker:
    """Match BEV predicted instances to CARLA vehicle actors.

    Uses the motion grid (200x200, 0.5m) for BEV projection, then
    Hungarian assignment between projected vehicle positions and
    predicted instance centers.
    """

    def __init__(self, config: CsBevConfig, match_threshold: float = 10.0):
        """
        Args:
            config: BEV grid configuration
            match_threshold: max BEV pixel distance for a valid match
                             (10 px * 0.5m = 5m world distance)
        """
        self._config = config
        self._grid_conf = config.motion_grid
        self._match_threshold = match_threshold
        self._prev_matching = {}
        self._last_vehicle_bev_pos = {}
        self._last_bev_centers = {}

    def _get_vehicle_bev_positions(
        self, vehicles, rsu_transform_tuple
    ) -> Dict[int, Tuple[float, float]]:
        """Project vehicle world positions into BEV pixel coordinates.

        Uses the RSU position as the BEV ego pose (center of BEV grid).

        Args:
            vehicles: list of carla.Actor (all vehicles in BEV range,
                      including ADS ego vehicles and NPC vehicles)
            rsu_transform_tuple: (rsu_x, rsu_y, rsu_yaw_rad) in world frame

        Returns:
            dict: {carla_actor_id: (bev_px, bev_py)}
        """
        positions = {}
        for vehicle in vehicles:
            if vehicle is None or not vehicle.is_alive:
                continue
            loc = vehicle.get_location()
            px, py = world_to_bev(
                loc.x, loc.y, self._grid_conf, rsu_transform_tuple)
            nx = int((self._grid_conf['xbound'][1] - self._grid_conf['xbound'][0])
                     / self._grid_conf['xbound'][2])
            ny = int((self._grid_conf['ybound'][1] - self._grid_conf['ybound'][0])
                     / self._grid_conf['ybound'][2])
            if 0 <= px < nx and 0 <= py < ny:
                positions[vehicle.id] = (px, py)
        return positions

    def _match_instances(
        self,
        bev_centers: Dict[int, Tuple[float, float]],
        vehicle_bev_pos: Dict[int, Tuple[float, float]],
    ) -> Dict[int, int]:
        """Hungarian matching between BEV instance centers and vehicle positions.

        Args:
            bev_centers: {instance_id: (center_x, center_y)} in BEV pixels
            vehicle_bev_pos: {carla_actor_id: (bev_px, bev_py)}

        Returns:
            dict: {instance_id: carla_actor_id} for matched pairs
        """
        if not bev_centers or not vehicle_bev_pos:
            return {}

        inst_ids = list(bev_centers.keys())
        veh_ids = list(vehicle_bev_pos.keys())

        inst_positions = np.array([bev_centers[iid] for iid in inst_ids])
        veh_positions = np.array([vehicle_bev_pos[vid] for vid in veh_ids])

        diff = inst_positions[:, None, :] - veh_positions[None, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=-1))

        row_idx, col_idx = linear_sum_assignment(distances)

        matching = {}
        for r, c in zip(row_idx, col_idx):
            if distances[r, c] < self._match_threshold:
                matching[inst_ids[r]] = veh_ids[c]

        return matching

    def update_gt_positions(self, vehicles, rsu_location):
        """Compute GT vehicle BEV positions from CARLA. Always call this.

        This is independent of model predictions — it only uses CARLA actor
        positions to project vehicles into BEV pixel space.

        Args:
            vehicles: list of carla.Actor (all vehicles in BEV range)
            rsu_location: carla.Location
        """
        rsu_pose = (rsu_location.x, rsu_location.y, 0.0)
        self._last_vehicle_bev_pos = self._get_vehicle_bev_positions(
            vehicles, rsu_pose)
        if self._last_vehicle_bev_pos:
            print("[CoSafe Tracker] GT vehicles in BEV: {}".format(
                {vid: "({:.1f},{:.1f})".format(px, py)
                 for vid, (px, py) in self._last_vehicle_bev_pos.items()}))

    def update(
        self,
        instance_seg,
        matched_centers: Optional[Dict],
        vehicles,
        rsu_location,
    ) -> Dict[int, int]:
        """Update instance-to-vehicle matching.

        Args:
            instance_seg: (1, seq_len, H, W) tensor from model
            matched_centers: dict {instance_id: [[x,y],...]} from model
                (centers in BEV pixel space, x=col, y=row)
            vehicles: list of carla.Actor (all vehicles in BEV range)
            rsu_location: carla.Location

        Returns:
            dict: {instance_id: carla_actor_id} mapping
        """
        self.update_gt_positions(vehicles, rsu_location)

        if matched_centers is None or not matched_centers:
            return self._prev_matching

        bev_centers = {}
        for inst_id, trajectory in matched_centers.items():
            if trajectory and len(trajectory) > 0:
                bev_centers[inst_id] = (trajectory[0][0], trajectory[0][1])

        self._last_bev_centers = bev_centers

        matching = self._match_instances(bev_centers, self._last_vehicle_bev_pos)
        self._prev_matching = matching

        return matching

    def get_last_matching(self) -> Dict[int, int]:
        """Return the most recent instance-to-vehicle matching."""
        return self._prev_matching

    def get_last_vehicle_bev_pos(self) -> Dict[int, Tuple[float, float]]:
        """Return CARLA GT vehicle positions in BEV pixel space from last update."""
        return self._last_vehicle_bev_pos

    def get_last_bev_centers(self) -> Dict[int, Tuple[float, float]]:
        """Return predicted instance centers in BEV pixel space from last update."""
        return self._last_bev_centers
