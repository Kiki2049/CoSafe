"""BEVerse V2X inference wrapper for CoSafe Phase 1.

Loads the DeepAccident BEVerse_V2X model and runs cooperative perception
on live CARLA camera data. Produces BEV segmentation + K=6 future
trajectory predictions.

Key references:
- Model: DeepAccident/projects/mmdet3d_plugin/models/detectors/beverse.py
  (BEVerse_V2X.simple_test, line 815; extract_img_feat, line 588)
- Preprocessing: DeepAccident/projects/mmdet3d_plugin/datasets/pipelines/loading.py
  (sample_augmentation line 100, img_transform line 25, normalize_img line 52)
- Instance extraction: DeepAccident/projects/mmdet3d_plugin/datasets/utils/instance.py
  (predict_instance_segmentation_and_trajectories, line 382)
"""

import math
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np
import torch
import torchvision

from cs_cosafe_c.cs_config import CsPerceptionConfig, CsCameraConfig
from cs_cosafe_c.perception.cs_geometry import mat2pose_vec, pose_vec2mat
from cs_cosafe_c.perception.cs_instance_utils import (
    predict_instance_segmentation_and_trajectories,
)


_normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
))


def _get_rot(h):
    """2D rotation matrix for angle h (radians)."""
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def _img_transform(img, post_rot, post_tran, resize, resize_dims, crop,
                   flip, rotate):
    """Apply image augmentation and track the 2D post-transformation.

    Copied from DeepAccident loading.py:25-49.
    """
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        from PIL import Image
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = _get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


def _sample_augmentation_test(H, W, fH, fW):
    """Compute test-time augmentation parameters.

    From DeepAccident loading.py:100-128, test branch (is_train=False).
    """
    resize = max(fH / H, fW / W) + 0.04
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = max(0, newH - fH)
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
    return resize, resize_dims, crop, flip, rotate


def _carla_transform_to_matrix(transform):
    """Convert carla.Transform to 4x4 numpy matrix.

    Uses CARLA's convention: yaw=rotation about Z, pitch=Y, roll=X.
    Based on fixed_sensors.py:109-137.
    """
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.identity(4)
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    return matrix


def _compute_intrinsic(width, height, fov):
    """Compute 3x3 camera intrinsic matrix from FOV and resolution.

    Uses DeepAccident's non-standard format which bakes in the CARLA camera
    coordinate convention (X=forward, Y=right, Z=up):
        [[cx, fx,   0],
         [cy,  0, -fy],
         [ 1,  0,   0]]

    This ensures inv(K) maps pixel coords [u*d, v*d, d] directly to CARLA
    camera-frame 3D coords, which pairs correctly with our CARLA-derived
    extrinsic rotation matrices (cam_to_agent in CARLA coords).

    Reference: DeepAccident calibration .pkl camera_intrinsic_matrix format.
    """
    fx = fy = width / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [cx, fx,   0.0],
        [cy, 0.0, -fy],
        [1.0, 0.0, 0.0],
    ])
    return K


class CsBeverseInference:
    """BEVerse V2X model wrapper for live CARLA inference.

    Handles:
    - Model loading via mmcv/mmdet3d registry
    - CARLA sensor data -> model tensor conversion
    - Agent selection (4 vehicles + 1 RSU)
    - Temporal frame buffering
    - K=6 scene prediction
    """

    def __init__(self, perception_config: CsPerceptionConfig,
                 camera_config: CsCameraConfig,
                 bev_config=None):
        self._config = perception_config
        self._cam_config = camera_config
        self._bev_config = bev_config
        self._device = torch.device(perception_config.device)
        self._model = None
        self._tick_count = 0
        self._inference_count = 0

        self._frame_buffers = {}

        H, W = perception_config.orig_height, perception_config.orig_width
        fH, fW = perception_config.final_dim
        self._aug_params = _sample_augmentation_test(H, W, fH, fW)

        self._last_predictions = None

        self._prev_agent_transforms = {}

        self._last_ego_vehicles = None

    @property
    def is_warmed_up(self):
        """True when all temporal buffers have enough real (non-padded) frames."""
        rf = self._config.receptive_field
        if not self._frame_buffers:
            return False
        return all(
            sum(buf.get('valid', [])) >= rf
            for buf in self._frame_buffers.values()
        )

    def load_model(self):
        """Load BEVerse_V2X from mmcv config + checkpoint.

        Uses mmcv Config.fromfile() + mmdet3d build_model() + load_checkpoint().
        Config path: DeepAccident/projects/configs/DeepAccident_tiny_v2x_4agents_infra.py
        """
        cfg_path = self._config.model_config_path
        ckpt_path = self._config.checkpoint_path

        if not cfg_path or not os.path.exists(cfg_path):
            print("[CoSafe] WARNING: perception model config not found: {}".format(cfg_path))
            print("[CoSafe] Perception will run in dummy mode.")
            return

        if not ckpt_path or not os.path.exists(ckpt_path):
            print("[CoSafe] WARNING: perception checkpoint not found: {}".format(ckpt_path))
            print("[CoSafe] Perception will run in dummy mode.")
            return

        configs_dir = os.path.dirname(os.path.abspath(cfg_path))
        projects_dir = os.path.dirname(configs_dir)
        deepaccident_root = os.path.dirname(projects_dir)
        for p in (deepaccident_root, projects_dir):
            if p not in sys.path:
                sys.path.insert(0, p)

        import mmdet3d_plugin  # noqa: F401

        from mmcv import Config
        from mmdet3d.models import build_model
        from mmcv.runner import load_checkpoint

        cfg = Config.fromfile(cfg_path)
        cfg.model.pretrained = None

        self._model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(self._model, ckpt_path, map_location='cpu')
        self._model = self._model.to(self._device)
        self._model.eval()

        self._per_agent_bev_raw = None

        def _hook_fn(module, args):
            self._per_agent_bev_raw = args[0].detach()

        self._model.v2x_fusion.register_forward_pre_hook(_hook_fn)

        if not self._config.cooperative_perception:
            def _no_coop_hook(module, args, output):
                return args[0][:, 0].contiguous()
            self._model.v2x_fusion.register_forward_hook(_no_coop_hook)
            print("[CoSafe] Ablation: cross-agent fusion disabled (RSU-only)")

        print("[CoSafe] BEVerse_V2X model loaded on {}".format(self._device))

    def _extract_images_from_sensor_data(self, agent_sensor_data, camera_names):
        """Extract RGB images from CARLA sensor data dict.

        CARLA sensor data format: {tag: (timestamp, bgra_array)}
        The BGRA array shape is (H, W, 4). We take [:, :, :3] and convert BGR->RGB.

        Args:
            agent_sensor_data: dict {tag: (ts, data)} from SensorInterface
            camera_names: list of camera name strings

        Returns:
            dict: {camera_name: PIL.Image} or None if missing cameras
        """
        from PIL import Image

        images = {}
        for tag, (ts, data) in agent_sensor_data.items():
            for cam_name in camera_names:
                if tag.endswith(cam_name):
                    if data is not None and len(data.shape) == 3:
                        rgb = data[:, :, :3][:, :, ::-1]
                        images[cam_name] = Image.fromarray(rgb.copy())
                    break
        return images if len(images) == len(camera_names) else None

    def _preprocess_single_image(self, pil_img):
        """Preprocess a single PIL image for the model.

        Applies: resize -> crop -> normalize (no flip/rotate at test time).
        From DeepAccident loading.py:143-241 (test path).

        Returns:
            img_tensor: (C, H, W) normalized tensor
            post_rot: (3, 3) post-augmentation rotation
            post_tran: (3,) post-augmentation translation
        """
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        resize, resize_dims, crop, flip, rotate = self._aug_params

        img, post_rot2, post_tran2 = _img_transform(
            pil_img, post_rot, post_tran,
            resize=resize, resize_dims=resize_dims,
            crop=crop, flip=flip, rotate=rotate,
        )

        post_tran_3 = torch.zeros(3)
        post_rot_3 = torch.eye(3)
        post_tran_3[:2] = post_tran2
        post_rot_3[:2, :2] = post_rot2

        img_tensor = _normalize_img(img)
        return img_tensor, post_rot_3, post_tran_3

    def _compute_camera_extrinsics(self, cam_yaws, cam_pitch, cam_positions,
                                   agent_type='rsu'):
        """Compute camera-to-agent rotation and translation for all 6 cameras.

        Each camera is positioned relative to the agent center. We compute the
        transform from camera frame to agent (ego/RSU) frame.

        Based on fixed_sensors.py:98-107 (get_camera_extrinsic).

        Args:
            cam_yaws: tuple of 6 yaw angles (degrees)
            cam_pitch: pitch angle (degrees)
            cam_positions: tuple of 6 (x, y, z) camera mount positions,
                or single float z for RSU (all at origin with given height)
            agent_type: 'rsu' or 'vehicle'

        Returns:
            rots: list of 6 (3, 3) rotation tensors
            trans: list of 6 (3,) translation tensors
        """
        rots = []
        trans_list = []
        for i, yaw in enumerate(cam_yaws):
            if agent_type == 'vehicle':
                cx, cy, cz = cam_positions[i]
            else:
                cx, cy, cz = 0.0, 0.0, 0.0
            cam_transform = carla.Transform(
                carla.Location(x=cx, y=cy, z=cz),
                carla.Rotation(roll=0.0, pitch=cam_pitch, yaw=yaw),
            )
            agent_transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.0),
                carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
            )
            cam_mat = np.array(cam_transform.get_matrix())
            agent_mat = np.array(agent_transform.get_matrix())
            agent_inv = np.linalg.inv(agent_mat)
            extrinsic = agent_inv @ cam_mat

            rots.append(torch.Tensor(extrinsic[:3, :3]))
            trans_list.append(torch.Tensor(extrinsic[:3, 3]))
        return rots, trans_list

    def _prepare_agent_frame(self, images, intrinsics, cam_rots, cam_trans):
        """Convert one agent's 6-camera images to model input tensors for one frame.

        Args:
            images: dict of camera_name -> PIL image
            intrinsics: list of 6 (3,3) intrinsic matrices (per-camera)
            cam_rots: list of 6 (3,3) rotation tensors
            cam_trans: list of 6 (3,) translation tensors

        Returns:
            imgs: (N=6, C, H, W)
            rots: (N=6, 3, 3)
            trans: (N=6, 3)
            intrins: (N=6, 3, 3)
            post_rots: (N=6, 3, 3)
            post_trans: (N=6, 3)
        """
        camera_names = self._cam_config.camera_names
        imgs_list = []
        rots_list = []
        trans_list = []
        intrins_list = []
        post_rots_list = []
        post_trans_list = []

        for i, cam_name in enumerate(camera_names):
            pil_img = images[cam_name]
            img_tensor, post_rot, post_tran = self._preprocess_single_image(pil_img)
            imgs_list.append(img_tensor)
            rots_list.append(cam_rots[i])
            trans_list.append(cam_trans[i])
            intrins_list.append(torch.Tensor(intrinsics[i]))
            post_rots_list.append(post_rot)
            post_trans_list.append(post_tran)

        return (
            torch.stack(imgs_list),
            torch.stack(rots_list),
            torch.stack(trans_list),
            torch.stack(intrins_list),
            torch.stack(post_rots_list),
            torch.stack(post_trans_list),
        )

    def select_agents(self, sensor_data, ego_vehicles, rsu_location):
        """Select 4 closest vehicles + RSU as the 5 agents.

        The model expects exactly 5 agents (4 vehicle + 1 infrastructure).
        When fewer than 4 vehicles are available, we duplicate one vehicle's
        data to fill the remaining slots.

        Args:
            sensor_data: dict from CsSensorManager.tick_all()
            ego_vehicles: list of carla.Actor
            rsu_location: carla.Location

        Returns:
            selected: list of (agent_key, agent_type) length 5
                agent_key: vehicle index (int) or 'rsu'
                agent_type: 'vehicle' or 'rsu'
            agent_mapping: dict {model_slot: actual_vehicle_id or 'rsu'}
        """
        bev_range = abs(self._bev_config.motion_grid['xbound'][0])
        max_dist = bev_range + 5
        vehicles_with_cameras = set(sensor_data.get('vehicles', {}).keys())
        vehicle_dists = []
        for vid, vehicle in enumerate(ego_vehicles):
            if vehicle is not None and vehicle.is_alive and vid in vehicles_with_cameras:
                vloc = vehicle.get_location()
                dist = vloc.distance(rsu_location)
                if dist <= max_dist:
                    vehicle_dists.append((vid, dist))

        vehicle_dists.sort(key=lambda x: x[1])
        selected_vids = [vd[0] for vd in vehicle_dists[:4]]

        if not selected_vids:
            return [], {}

        while len(selected_vids) < 4:
            selected_vids.append(selected_vids[0])

        selected = [('rsu', 'rsu')]
        agent_mapping = {0: 'rsu'}
        for slot_idx, vid in enumerate(selected_vids):
            selected.append((vid, 'vehicle'))
            agent_mapping[slot_idx + 1] = vid

        return selected, agent_mapping

    def _compute_egomotion(self, agent_key, current_transform):
        """Compute 6-DoF egomotion: transform from prev frame TO current frame.

        Stored in buffer at the current frame's position.  After the 1-slot
        shift in ``_get_agent_tensors`` this becomes ego[t] = from-t-to-(t+1),
        matching the training convention (DeepAccident_dataset.py:283).

        Returns:
            (6,) tensor [tx, ty, tz, rx, ry, rz] or zeros if first frame
        """
        if agent_key in self._prev_agent_transforms:
            prev_mat = _carla_transform_to_matrix(self._prev_agent_transforms[agent_key])
            curr_mat = _carla_transform_to_matrix(current_transform)
            curr_inv = np.linalg.inv(curr_mat)
            relative = curr_inv @ prev_mat
            egomotion = mat2pose_vec(
                torch.Tensor(relative).unsqueeze(0)
            ).squeeze(0)
        else:
            egomotion = torch.zeros(6)

        self._prev_agent_transforms[agent_key] = current_transform
        return egomotion

    def prepare_relative_poses(self, agent_transforms, ref_index=0):
        """Compute relative 6-DoF poses between agents.

        Each agent's pose is expressed relative to the reference agent (index 0 or RSU).
        From DeepAccident geometry.py:21-45 (mat2pose_vec).

        Args:
            agent_transforms: list of 5 carla.Transform
            ref_index: index of reference agent

        Returns:
            (5, 6) tensor of relative poses [tx,ty,tz,rx,ry,rz]
        """
        ref_mat = _carla_transform_to_matrix(agent_transforms[ref_index])
        ref_inv = np.linalg.inv(ref_mat)

        poses = []
        for t in agent_transforms:
            agent_mat = _carla_transform_to_matrix(t)
            relative = ref_inv @ agent_mat
            pose_vec = mat2pose_vec(
                torch.Tensor(relative).unsqueeze(0)
            ).squeeze(0)
            poses.append(pose_vec)

        return torch.stack(poses)

    def _update_frame_buffer(self, agent_key, frame_data, egomotion):
        """Append a new frame to the temporal buffer for an agent.

        The buffer stores the last `receptive_field` frames.
        First ticks: duplicate current frame to fill buffer.

        Args:
            agent_key: identifier for this agent
            frame_data: tuple of (imgs, rots, trans, intrins, post_rots, post_trans)
            egomotion: (6,) tensor
        """
        rf = self._config.receptive_field
        if agent_key not in self._frame_buffers:
            self._frame_buffers[agent_key] = {
                'frames': deque(maxlen=rf),
                'egomotions': deque(maxlen=rf),
                'valid': deque(maxlen=rf),
            }

        buf = self._frame_buffers[agent_key]
        buf['frames'].append(frame_data)
        buf['egomotions'].append(egomotion)
        buf['valid'].append(True)

        while len(buf['frames']) < rf:
            buf['frames'].appendleft(frame_data)
            buf['egomotions'].appendleft(torch.zeros(6))
            buf['valid'].appendleft(False)

    def _get_agent_tensors(self, agent_key):
        """Retrieve stacked temporal tensors for one agent from the frame buffer.

        Returns:
            imgs: (S, N, C, H, W)
            rots: (S, N, 3, 3)
            trans: (S, N, 3)
            intrins: (S, N, 3, 3)
            post_rots: (S, N, 3, 3)
            post_trans: (S, N, 3)
            egomotions: (S, 6)
            valid: (S,) boolean
        """
        buf = self._frame_buffers[agent_key]
        frames = list(buf['frames'])

        imgs = torch.stack([f[0] for f in frames])
        rots = torch.stack([f[1] for f in frames])
        trans = torch.stack([f[2] for f in frames])
        intrins = torch.stack([f[3] for f in frames])
        post_rots = torch.stack([f[4] for f in frames])
        post_trans = torch.stack([f[5] for f in frames])

        egos = list(buf['egomotions'])
        egomotions = torch.stack(egos[1:] + [torch.zeros(6)])
        valid = torch.tensor(list(buf['valid']))

        return imgs, rots, trans, intrins, post_rots, post_trans, egomotions, valid

    def run_inference(self, sensor_data, ego_vehicles, rsu_location):
        """Run one step of perception inference.

        Only actually runs the model every `inference_interval` ticks.
        Between runs, returns cached predictions.

        Args:
            sensor_data: dict from CsSensorManager.tick_all()
                {'rsu': {tag: (ts, data)}, 'vehicles': {vid: {tag: (ts, data)}}}
            ego_vehicles: list of carla.Actor
            rsu_location: carla.Location

        Returns:
            dict with keys:
                'instance_seg': (1, seq_len, H, W) tensor
                'matched_centers': dict {inst_id: [[x,y],...]}
                'segmentpixelscnt': dict {inst_id: [count,...]}
                'agent_mapping': dict {slot: vehicle_id or 'rsu'}
                'raw_output': dict of raw model outputs
            or None if model not loaded
        """
        self._tick_count += 1
        self._last_ego_vehicles = ego_vehicles

        if self._model is None:
            return self._last_predictions

        if self._tick_count % self._config.inference_interval != 0:
            return self._last_predictions

        selected, agent_mapping = self.select_agents(
            sensor_data, ego_vehicles, rsu_location)

        if not selected:
            return self._last_predictions

        camera_names = self._cam_config.camera_names

        rsu_rots, rsu_trans = self._compute_camera_extrinsics(
            self._cam_config.rsu_yaws, self._cam_config.rsu_pitch, 0.0, 'rsu')
        rsu_intrinsics = [_compute_intrinsic(
            self._cam_config.rsu_width, self._cam_config.rsu_height, fov)
            for fov in self._cam_config.rsu_fovs]

        veh_rots, veh_trans = self._compute_camera_extrinsics(
            self._cam_config.veh_yaws, self._cam_config.veh_pitch,
            self._cam_config.veh_positions, 'vehicle')
        veh_intrinsics = [_compute_intrinsic(
            self._cam_config.veh_width, self._cam_config.veh_height, fov)
            for fov in self._cam_config.veh_fovs]

        agent_transforms = []

        for agent_key, agent_type in selected:
            if agent_type == 'vehicle':
                vid = agent_key
                veh_data = sensor_data.get('vehicles', {}).get(vid, {})
                images = self._extract_images_from_sensor_data(
                    veh_data, camera_names)

                vehicle = ego_vehicles[vid]
                current_transform = vehicle.get_transform()
                agent_transforms.append(current_transform)

                if images is None:
                    egomotion = self._compute_egomotion(
                        'veh_{}'.format(vid), current_transform)
                    self._update_frame_buffer(
                        'veh_{}'.format(vid),
                        self._make_zero_frame(),
                        egomotion,
                    )
                    continue

                egomotion = self._compute_egomotion(
                    'veh_{}'.format(vid), current_transform)
                frame = self._prepare_agent_frame(
                    images, veh_intrinsics, veh_rots, veh_trans)
                self._update_frame_buffer('veh_{}'.format(vid), frame, egomotion)

            else:
                rsu_data = sensor_data.get('rsu', {})
                images = self._extract_images_from_sensor_data(
                    rsu_data, camera_names)

                rsu_transform = carla.Transform(
                    rsu_location,
                    carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
                )
                agent_transforms.append(rsu_transform)

                if images is None:
                    egomotion = torch.zeros(6)
                    self._update_frame_buffer(
                        'rsu', self._make_zero_frame(), egomotion)
                    continue

                egomotion = torch.zeros(6)
                frame = self._prepare_agent_frame(
                    images, rsu_intrinsics, rsu_rots, rsu_trans)
                self._update_frame_buffer('rsu', frame, egomotion)

        img_list = []
        future_egomotions_list = []
        img_is_valid_list = []

        agent_keys = []
        for agent_key, agent_type in selected:
            if agent_type == 'vehicle':
                agent_keys.append('veh_{}'.format(agent_key))
            else:
                agent_keys.append('rsu')

        for ak in agent_keys:
            imgs, rots, trans, intrins, post_rots, post_trans, egos, valid = \
                self._get_agent_tensors(ak)

            img_tuple = (
                imgs.unsqueeze(0).to(self._device),
                rots.unsqueeze(0).to(self._device),
                trans.unsqueeze(0).to(self._device),
                intrins.unsqueeze(0).to(self._device),
                post_rots.unsqueeze(0).to(self._device),
                post_trans.unsqueeze(0).to(self._device),
            )
            img_list.append(img_tuple)
            future_egomotions_list.append(egos.unsqueeze(0).to(self._device))
            img_is_valid_list.append(valid.unsqueeze(0).to(self._device))

        relative_poses = self.prepare_relative_poses(agent_transforms, ref_index=0)
        relative_poses = relative_poses.unsqueeze(0).to(self._device)

        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        dummy_lidar2ego_rots = torch.tensor([
            [-5.4280e-04,  9.9893e-01,  4.6229e-02],
            [-1.0000e+00, -4.0569e-04, -2.9750e-03],
            [-2.9531e-03, -4.6231e-02,  9.9893e-01]])
        dummy_lidar2ego_trans = torch.tensor([0.9858, 0.0000, 1.8402])
        img_metas = [
            [{'box_type_3d': LiDARInstance3DBoxes,
              'lidar2ego_rots': dummy_lidar2ego_rots,
              'lidar2ego_trans': dummy_lidar2ego_trans}]
            for _ in range(len(selected))
        ]

        self._inference_count += 1

        if self._inference_count <= 3:
            print("[CoSafe DBG] === Inference #{} (agent order: RSU@0, vehicles@1-4) ===".format(
                self._inference_count))
            for i, agent_data in enumerate(img_list):
                imgs = agent_data[0]
                slot_label = 'RSU' if i == 0 else 'veh'
                print("[CoSafe DBG] Slot {} ({}): shape={}, min={:.3f}, max={:.3f}".format(
                    i, slot_label, list(imgs.shape), imgs.min().item(), imgs.max().item()))
            print("[CoSafe DBG] Relative poses:\n{}".format(relative_poses))
            for ak, buf in self._frame_buffers.items():
                n_valid = sum(buf.get('valid', []))
                print("[CoSafe DBG] Buffer '{}': {}/{} valid frames".format(
                    ak, n_valid, len(buf.get('valid', []))))

        try:
            with torch.no_grad():
                predictions = self._model.simple_test(
                    img_metas=img_metas,
                    img=img_list,
                    future_egomotions=future_egomotions_list,
                    img_is_valid=img_is_valid_list,
                    relative_pose_to_ego=[relative_poses],
                )
        except Exception as e:
            print("[CoSafe] Inference error: {}".format(e))
            return self._last_predictions

        rsu_only_predictions = None
        if self._config.enable_rsu_only_inference:
            try:
                with torch.no_grad():
                    rsu_only_img_list = []
                    rsu_only_ego_list = []
                    rsu_only_valid_list = []
                    for i, (ak, img_tuple) in enumerate(zip(agent_keys, img_list)):
                        if i == 0:
                            rsu_only_img_list.append(img_tuple)
                            rsu_only_ego_list.append(future_egomotions_list[i])
                            rsu_only_valid_list.append(img_is_valid_list[i])
                        else:
                            zero_imgs = torch.zeros_like(img_tuple[0])
                            rsu_only_img_list.append((
                                zero_imgs,
                                img_tuple[1], img_tuple[2],
                                img_tuple[3], img_tuple[4], img_tuple[5],
                            ))
                            rsu_only_ego_list.append(torch.zeros_like(
                                future_egomotions_list[i]))
                            rsu_only_valid_list.append(torch.zeros_like(
                                img_is_valid_list[i]))

                    rsu_only_predictions = self._model.simple_test(
                        img_metas=img_metas,
                        img=rsu_only_img_list,
                        future_egomotions=rsu_only_ego_list,
                        img_is_valid=rsu_only_valid_list,
                        relative_pose_to_ego=[relative_poses],
                    )
            except Exception as e:
                print("[CoSafe] RSU-only inference error: {}".format(e))
                import traceback
                traceback.print_exc()

        try:
            motion_preds = predictions.get('motion_predictions')
            if motion_preds is not None:
                if isinstance(motion_preds, list):
                    mean_pred = motion_preds[-1]
                else:
                    mean_pred = motion_preds

                if self._inference_count <= 3:
                    for key in ['segmentation', 'instance_center',
                                'instance_offset', 'instance_flow']:
                        t = mean_pred.get(key)
                        if t is not None:
                            print("[CoSafe DBG] {}: shape={}, min={:.6f}, "
                                  "max={:.6f}, mean={:.6f}".format(
                                      key, list(t.shape), t.min().item(),
                                      t.max().item(), t.mean().item()))

                seg = mean_pred.get('segmentation')
                if seg is not None:
                    seg_vals = seg.detach()
                    sig_ch1 = seg_vals[:, :, 1, :, :].sigmoid()
                    print("[CoSafe] seg ch1 sigmoid: min={:.4f}, max={:.4f}, "
                          ">0.4 pixels={}".format(
                              sig_ch1.min().item(), sig_ch1.max().item(),
                              (sig_ch1 > 0.4).sum().item()))

                instance_seg, matched_centers, seg_counts, inst_flows = \
                    predict_instance_segmentation_and_trajectories(
                        mean_pred,
                        compute_matched_centers=True,
                        make_consistent=True,
                    )
                all_samples = []
                preds_list = motion_preds if isinstance(motion_preds, list) else [motion_preds]
                for k, pred_k in enumerate(preds_list):
                    try:
                        iseg_k, mc_k, sc_k, if_k = \
                            predict_instance_segmentation_and_trajectories(
                                pred_k,
                                compute_matched_centers=True,
                                make_consistent=True,
                            )
                        all_samples.append({
                            'instance_seg': iseg_k,
                            'matched_centers': mc_k,
                            'segmentpixelscnt': sc_k,
                            'instance_flows': if_k,
                        })
                    except Exception as ek:
                        print("[CoSafe] Instance extraction for sample {} error: {}".format(k, ek))
                        all_samples.append({
                            'instance_seg': None,
                            'matched_centers': {},
                            'segmentpixelscnt': {},
                            'instance_flows': {},
                        })

                self._last_predictions = {
                    'instance_seg': instance_seg,
                    'matched_centers': matched_centers,
                    'segmentpixelscnt': seg_counts,
                    'instance_flows': inst_flows,
                    'all_samples': all_samples,
                    'agent_mapping': agent_mapping,
                    'motion_predictions': motion_preds,
                    'raw_output': predictions,
                }
                if rsu_only_predictions is not None:
                    rsu_mp = rsu_only_predictions.get('motion_predictions')
                    self._last_predictions['rsu_only_motion_predictions'] = rsu_mp
                    if rsu_mp:
                        rsu_first = rsu_mp[0] if isinstance(rsu_mp, list) else rsu_mp
                        rsu_seg = rsu_first.get('segmentation')
                        if rsu_seg is not None:
                            sig = rsu_seg[:, :, 1, :, :].sigmoid()
                            print("[CoSafe] RSU-only seg ch1: min={:.4f}, max={:.4f}, "
                                  ">0.4 px={}".format(
                                      sig.min().item(), sig.max().item(),
                                      (sig > 0.4).sum().item()))
                print("[CoSafe] Inference #{} OK: {} instances detected".format(
                    self._inference_count,
                    len(matched_centers) if matched_centers else 0))
            else:
                print("[CoSafe] No motion predictions in output. Keys: {}".format(
                    list(predictions.keys())))
                self._last_predictions = {
                    'raw_output': predictions,
                    'agent_mapping': agent_mapping,
                }
        except Exception as e:
            print("[CoSafe] Instance extraction error: {}".format(e))
            import traceback
            traceback.print_exc()
            self._last_predictions = {
                'raw_output': predictions,
                'agent_mapping': agent_mapping,
            }

        return self._last_predictions

    def _make_zero_frame(self):
        """Create a zero-filled frame tuple for missing data."""
        fH, fW = self._config.final_dim
        N = 6
        return (
            torch.zeros(N, 3, fH, fW),
            torch.eye(3).unsqueeze(0).repeat(N, 1, 1),
            torch.zeros(N, 3),
            torch.eye(3).unsqueeze(0).repeat(N, 1, 1),
            torch.eye(3).unsqueeze(0).repeat(N, 1, 1),
            torch.zeros(N, 3),
        )

    def get_per_agent_bev(self):
        """Return per-agent pre-fusion BEV features from last inference.

        Uses the forward_pre_hook on v2x_fusion to capture the (B, 5, C, H, W)
        tensor before fusion.  Slot 0 = RSU (skipped), slots 1-4 = vehicles.
        Maps model slots to CARLA actor IDs via agent_mapping from last inference.

        Returns:
            dict: {carla_actor_id: tensor(C, H, W)} for vehicles with cameras,
            or empty dict if hook data is unavailable.
        """
        if self._per_agent_bev_raw is None or self._last_predictions is None:
            return {}

        agent_mapping = self._last_predictions.get('agent_mapping', {})
        bev_all = self._per_agent_bev_raw[0]

        result = {}
        for slot_idx in range(1, bev_all.shape[0]):
            vid = agent_mapping.get(slot_idx)
            if vid is None or vid == 'rsu':
                continue
            if hasattr(self, '_last_ego_vehicles') and self._last_ego_vehicles is not None:
                vehicle = self._last_ego_vehicles[vid]
                if vehicle is not None and vehicle.is_alive:
                    result[vehicle.id] = bev_all[slot_idx]
        return result

    def destroy(self):
        """Release model and buffers."""
        if self._model is not None:
            del self._model
            self._model = None
        self._frame_buffers.clear()
        self._prev_agent_transforms.clear()
        self._last_predictions = None
        self._per_agent_bev_raw = None
        torch.cuda.empty_cache()
        print("[CoSafe] BEVerse inference destroyed")
