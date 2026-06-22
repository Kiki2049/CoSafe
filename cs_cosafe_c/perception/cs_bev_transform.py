"""BEV (Bird's Eye View) coordinate transforms.

Pure math functions with no CARLA dependency.
Based on DeepAccident motion_grid_conf: xbound=[-50,50,0.5], ybound=[-50,50,0.5].
"""

import math
from typing import Dict, List, Tuple


def get_grid_conf(mode: str = 'motion') -> Dict[str, List[float]]:
    """Get grid configuration by mode.

    Args:
        mode: 'motion' for motion prediction grid (200x200, 0.5m)
              'det' for detection grid (128x128, 0.8m)

    Returns:
        Dict with 'xbound', 'ybound', 'zbound', 'dbound' keys.
    """
    if mode == 'motion':
        return {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-30.0, 30.0, 60.0],
            'dbound': [1.0, 60.0, 1.0],
        }
    elif mode == 'det':
        return {
            'xbound': [-51.2, 51.2, 0.8],
            'ybound': [-51.2, 51.2, 0.8],
            'zbound': [-30.0, 30.0, 60.0],
            'dbound': [1.0, 60.0, 1.0],
        }
    else:
        raise ValueError("Unknown grid mode: '{}'. Use 'motion' or 'det'.".format(mode))


def bev_to_world(
    px: float,
    py: float,
    grid_conf: Dict[str, List[float]],
    ego_pose: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Convert BEV pixel coordinates to world coordinates.

    Args:
        px: BEV pixel x index
        py: BEV pixel y index
        grid_conf: grid config dict with 'xbound' and 'ybound'
        ego_pose: (ego_x, ego_y, ego_yaw_rad) in world frame

    Returns:
        (world_x, world_y)
    """
    ego_x, ego_y, ego_yaw = ego_pose
    xbound = grid_conf['xbound']
    ybound = grid_conf['ybound']
    resolution_x = xbound[2]
    resolution_y = ybound[2]

    local_x = px * resolution_x + xbound[0]
    local_y = py * resolution_y + ybound[0]

    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    world_x = local_x * cos_yaw - local_y * sin_yaw + ego_x
    world_y = local_x * sin_yaw + local_y * cos_yaw + ego_y

    return (world_x, world_y)


def world_to_bev(
    wx: float,
    wy: float,
    grid_conf: Dict[str, List[float]],
    ego_pose: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Convert world coordinates to BEV pixel coordinates.

    Args:
        wx: world x coordinate
        wy: world y coordinate
        grid_conf: grid config dict with 'xbound' and 'ybound'
        ego_pose: (ego_x, ego_y, ego_yaw_rad) in world frame

    Returns:
        (bev_px, bev_py) as float pixel indices
    """
    ego_x, ego_y, ego_yaw = ego_pose
    xbound = grid_conf['xbound']
    ybound = grid_conf['ybound']
    resolution_x = xbound[2]
    resolution_y = ybound[2]

    dx = wx - ego_x
    dy = wy - ego_y

    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw

    px = (local_x - xbound[0]) / resolution_x
    py = (local_y - ybound[0]) / resolution_y

    return (px, py)
