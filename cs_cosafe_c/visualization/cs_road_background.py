"""Generate BEV road background from live CARLA map.

Equivalent to DeepAccident's plot_map_navsim() but using live CARLA
waypoints instead of pre-saved .npz map files.

Reference: projects/mmdet3d_plugin/visualize/visualizer_navsim.py:73
"""

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROAD_COLOR = (211, 211, 211)
LANE_COLOR = (40, 40, 40)
BACKGROUND_COLOR = (255, 255, 255)


def generate_road_background(
    rsu_location,
    grid_conf: Dict[str, List[float]],
) -> np.ndarray:
    """Generate a 200x200 road background from CARLA map waypoints.

    Mirrors plot_map_navsim(): road surfaces in light gray, lane center
    markings in dark gray, white background. Final image is rotated
    90 degrees clockwise to match NavSim output convention.

    Args:
        rsu_location: carla.Location of the RSU (BEV center).
        grid_conf: motion grid config dict with 'xbound', 'ybound'.

    Returns:
        (H, W, 3) uint8 RGB image in NavSim-rotated BEV orientation.
    """
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    carla_map = CarlaDataProvider.get_map()

    xbound = grid_conf['xbound']
    ybound = grid_conf['ybound']
    res = xbound[2]
    W = int((xbound[1] - xbound[0]) / res)
    H = int((ybound[1] - ybound[0]) / res)
    bev_range = max(abs(xbound[0]), abs(xbound[1]))

    rsu_x = rsu_location.x
    rsu_y = rsu_location.y

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    waypoints = carla_map.generate_waypoints(0.5)

    for wp in waypoints:
        loc = wp.transform.location
        dx = loc.x - rsu_x
        dy = loc.y - rsu_y
        if abs(dx) > bev_range + 5 or abs(dy) > bev_range + 5:
            continue

        px = (dx - xbound[0]) / res
        py = (dy - ybound[0]) / res
        ipx = int(round(px))
        ipy = int(round(py))

        if 0 <= ipx < W and 0 <= ipy < H:
            lane_width = wp.lane_width
            radius = max(1, int(lane_width / (2 * res)))
            cv2.circle(canvas, (ipx, ipy), radius, ROAD_COLOR, -1)

    for wp in waypoints:
        loc = wp.transform.location
        dx = loc.x - rsu_x
        dy = loc.y - rsu_y
        if abs(dx) > bev_range + 5 or abs(dy) > bev_range + 5:
            continue

        px = (dx - xbound[0]) / res
        py = (dy - ybound[0]) / res
        ipx = int(round(px))
        ipy = int(round(py))

        if 0 <= ipx < W and 0 <= ipy < H:
            cv2.circle(canvas, (ipx, ipy), 0, LANE_COLOR, -1)


    return canvas
