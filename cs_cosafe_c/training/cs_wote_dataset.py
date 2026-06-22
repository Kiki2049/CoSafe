"""
CoSafe WoTE training dataset.

Loads pre-extracted BEV features (.pt) + computes GT trajectories from Bench2Drive pkl.
GT trajectory interpolation logic matches cs_kmeans_trajectory.py.

Usage:
    dataset = CsWoTEDataset(
        pkl_path='b2d_e2e_ego_infos_train.pkl',
        bev_dir='extracted_bev_features/',   # directory with .pt files
    )
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class CsWoTEDataset(Dataset):
    """Dataset for WoTE offset network training.

    Each sample: (bev_feature, ego_status, gt_trajectory).
    GT trajectories are computed on-the-fly from pkl ego poses using interpolation.
    BEV features are loaded from pre-extracted .pt files.
    """

    def __init__(
        self,
        pkl_path: str,
        bev_dir: str = None,
        t_plan: int = 40,
        dt_target: float = 0.05,
        dt_source: float = 0.1,
        stride_frames: int = 5,
        bev_channels: int = 64,
        bev_size: int = 128,
    ):
        super().__init__()
        self.bev_dir = bev_dir
        self.bev_channels = bev_channels
        self.bev_size = bev_size
        self.t_plan = t_plan
        self.dt_target = dt_target
        self.dt_source = dt_source

        print(f"Loading {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            infos = pickle.load(f)

        clips = defaultdict(list)
        for info in infos:
            clips[info["folder"]].append(info)
        for folder in clips:
            clips[folder].sort(key=lambda x: x["frame_idx"])

        total_horizon_time = t_plan * dt_target
        min_frames = int(np.ceil(total_horizon_time / dt_source)) + 2

        self.samples = []
        for folder, frames in clips.items():
            if len(frames) < min_frames:
                continue
            for start_idx in range(0, len(frames) - min_frames, stride_frames):
                self.samples.append((folder, start_idx, frames))

        print(f"Dataset: {len(self.samples)} samples from {len(clips)} clips")

    def __len__(self):
        return len(self.samples)

    def _interpolate_pose(self, frames, target_time):
        """Linear interpolation of ego pose at continuous time."""
        frame_float = target_time / self.dt_source
        frame_lo = int(np.floor(frame_float))
        frame_hi = frame_lo + 1
        if frame_lo < 0 or frame_hi >= len(frames):
            return None

        alpha = frame_float - frame_lo
        lo, hi = frames[frame_lo], frames[frame_hi]

        x = (1 - alpha) * lo["ego_translation"][0] + alpha * hi["ego_translation"][0]
        y = (1 - alpha) * lo["ego_translation"][1] + alpha * hi["ego_translation"][1]

        yaw_lo = float(lo["ego_yaw"])
        yaw_hi = float(hi["ego_yaw"])
        dyaw = (yaw_hi - yaw_lo + np.pi) % (2 * np.pi) - np.pi
        yaw = yaw_lo + alpha * dyaw

        return x, y, yaw

    def __getitem__(self, idx):
        folder, start_idx, frames = self.samples[idx]
        info_t0 = frames[start_idx]

        x0 = info_t0["ego_translation"][0]
        y0 = info_t0["ego_translation"][1]
        yaw0 = float(info_t0["ego_yaw"])
        cos_neg = np.cos(-yaw0)
        sin_neg = np.sin(-yaw0)

        t0_time = start_idx * self.dt_source
        gt_traj = []
        for step in range(1, self.t_plan + 1):
            t_future = t0_time + step * self.dt_target
            pose = self._interpolate_pose(frames, t_future)
            if pose is None:
                gt_traj.append([0.0, 0.0, 0.0])
                continue
            xw, yw, yaw_w = pose
            dx, dy = xw - x0, yw - y0
            local_x = cos_neg * dx - sin_neg * dy
            local_y = sin_neg * dx + cos_neg * dy
            d_yaw = (yaw_w - yaw0 + np.pi) % (2 * np.pi) - np.pi
            gt_traj.append([local_x, local_y, d_yaw])

        gt_trajectory = torch.tensor(gt_traj, dtype=torch.float32)

        ego_vel = info_t0["ego_vel"][:2]
        vx_ego = cos_neg * ego_vel[0] - sin_neg * ego_vel[1]
        vy_ego = sin_neg * ego_vel[0] + cos_neg * ego_vel[1]

        if "ego_accel" in info_t0:
            ego_acc = info_t0["ego_accel"][:2]
            ax_ego = cos_neg * ego_acc[0] - sin_neg * ego_acc[1]
            ay_ego = sin_neg * ego_acc[0] + cos_neg * ego_acc[1]
        else:
            ax_ego, ay_ego = 0.0, 0.0

        CARLA_TO_WOTE = {1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2}
        cmd_raw = int(info_t0.get("command_near", 4))
        cmd_idx = CARLA_TO_WOTE.get(cmd_raw, 0)
        cmd_onehot = [0.0] * 4
        cmd_onehot[cmd_idx] = 1.0

        ego_status = torch.tensor(
            cmd_onehot + [float(vx_ego), float(vy_ego), float(ax_ego), float(ay_ego)],
            dtype=torch.float32,
        )

        if self.bev_dir is not None:
            bev_name = f"{folder.replace('/', '_')}_{info_t0['frame_idx']:05d}.pt"
            bev_path = os.path.join(self.bev_dir, bev_name)
            if os.path.exists(bev_path):
                bev_feature = torch.load(bev_path, map_location="cpu")
                while bev_feature.dim() > 3:
                    bev_feature = bev_feature.squeeze(0)
            else:
                bev_feature = torch.zeros(
                    self.bev_channels, self.bev_size, self.bev_size
                )
        else:
            bev_feature = torch.zeros(
                self.bev_channels, self.bev_size, self.bev_size
            )

        return {
            "bev_feature": bev_feature,
            "ego_status": ego_status,
            "gt_trajectory": gt_trajectory,
        }
