"""WoTE candidate trajectory generator for CoSafe Phase 3.

Wraps CsWoTENet for inference: BEV feature + ego status -> refined
candidate trajectories + scores.  Vehicles without cameras fall back
to raw K-means anchors with uniform scores.

ego_status (8-dim) layout (from cs_wote_dataset.py):
  [0:4] cmd one-hot  (FOLLOW=0, LEFT=1, RIGHT=2, STRAIGHT=3)
  [4]   vx_ego       ego-local forward velocity  (m/s)
  [5]   vy_ego       ego-local lateral velocity   (m/s)
  [6]   ax_ego       ego-local forward accel      (m/s^2)
  [7]   ay_ego       ego-local lateral accel       (m/s^2)

Reference: cs_cosafe/training/cs_wote_net.py (model architecture)
"""

import math

import numpy as np
import torch

from cs_cosafe_c.cs_config import CsWoTEConfig

_CARLA_CMD_TO_WOTE = {1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2}


def build_ego_status(vehicle_state, carla_vehicle=None, command=None):
    """Build the 8-dim ego_status vector for WoTE.

    Args:
        vehicle_state: CsVehicleState (heading_rad, speed).
        carla_vehicle: carla.Vehicle or None.  When available, used to
            compute accurate ego-local vx/vy/ax/ay.
        command: int (CARLA RoadOption value 1-6) or None.
            None defaults to FOLLOW (4 -> WoTE idx 0).

    Returns:
        np.ndarray (8,) float32.
    """
    status = np.zeros(8, dtype=np.float32)

    cmd_idx = _CARLA_CMD_TO_WOTE.get(command, 0) if command else 0
    status[cmd_idx] = 1.0

    yaw = vehicle_state.heading_rad
    cos_h = math.cos(yaw)
    sin_h = math.sin(yaw)

    if carla_vehicle is not None and carla_vehicle.is_alive:
        vel = carla_vehicle.get_velocity()
        acc = carla_vehicle.get_acceleration()
        status[4] = vel.x * cos_h + vel.y * sin_h
        status[5] = -vel.x * sin_h + vel.y * cos_h
        status[6] = acc.x * cos_h + acc.y * sin_h
        status[7] = -acc.x * sin_h + acc.y * cos_h
    else:
        status[4] = vehicle_state.speed

    return status


class CsWoTECandidateGenerator:
    """Generate candidate trajectories using the trained WoTE offset network."""

    def __init__(self, config: CsWoTEConfig, device):
        from cs_cosafe_c.training.cs_wote_net import CsWoTENet

        self._config = config
        self._device = torch.device(device) if isinstance(device, str) else device

        self._anchors = np.load(config.anchors_path)
        self._anchors_tensor = torch.tensor(
            self._anchors, dtype=torch.float32, device=self._device)

        self._model = CsWoTENet(
            anchors_path=config.anchors_path,
            bev_channels=config.bev_channels,
            d_model=config.d_model,
        ).to(self._device)

        state = torch.load(config.checkpoint_path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()

        M = self._anchors.shape[0]
        self._uniform_scores = np.full(M, 1.0 / M, dtype=np.float32)

        print("[CoSafe WoTE] Loaded checkpoint: {} ({} anchors)".format(
            config.checkpoint_path, M))

    def generate(self, bev_feature, ego_status):
        """Full WoTE: BEV feature + ego status -> refined candidates + scores.

        Args:
            bev_feature: (C, H, W) per-agent pre-fusion BEV tensor (on device).
            ego_status: np.ndarray (8,) from build_ego_status().

        Returns:
            candidates: (M, T, 3) numpy, ego-local coords.
            scores: (M,) numpy, softmax scores.
        """
        bev_batch = bev_feature.unsqueeze(0)
        status_batch = torch.tensor(
            ego_status, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        with torch.no_grad():
            offset, scores = self._model(bev_batch, status_batch)

        offset_np = offset[0].cpu().numpy()
        scores_np = scores[0].cpu().numpy()

        candidates = self._anchors + offset_np
        return candidates, scores_np

    def generate_fallback(self):
        """No camera: return raw anchors with uniform scores.

        Returns:
            candidates: (M, T, 3) numpy, ego-local coords (raw anchors).
            scores: (M,) numpy, uniform 1/M.
        """
        return self._anchors.copy(), self._uniform_scores.copy()
