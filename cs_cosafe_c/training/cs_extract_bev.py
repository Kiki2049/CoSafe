"""
Extract pre-fusion ego-centric BEV features from Bench2Drive images using BEVerse backbone.

Runs BEVerse (SwinTransformer + LSS) on single-agent 6-camera images from Bench2Drive,
extracts per-frame BEV features (64, 128, 128) BEFORE V2X fusion.

Requirements:
    - Run from an environment with mmdet3d, mmcv, and the DeepAccident repo installed
    - BEVerse checkpoint (e.g., work_dirs/DeepAccident_tiny_v2x_4agents_infra_pyramid/epoch_X.pth)

Usage:
    python cs_extract_bev.py \
        --pkl_path /path/to/b2d_e2e_ego_infos_train.pkl \
        --data_root /path/to/bench2drive/ \
        --config_path /path/to/DeepAccident/projects/configs/DeepAccident_singleframe_tiny_v2x.py \
        --checkpoint /path/to/beverse_checkpoint.pth \
        --output_dir /path/to/extracted_bev/
"""

import argparse
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


B2D_CAMERAS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
BEVERSE_CAMERAS = [
    "Camera_FrontLeft", "Camera_Front", "Camera_FrontRight",
    "Camera_BackLeft", "Camera_Back", "Camera_BackRight",
]

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_beverse_model(config_path, checkpoint_path, device):
    """Load BEVerse model from config and checkpoint."""
    import sys
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(config_path)

    if hasattr(cfg, 'plugin') and cfg.plugin:
        plugin_dir = getattr(cfg, 'plugin_dir', 'projects/mmdet3d_plugin').rstrip('/')
        config_abs = os.path.abspath(config_path)
        search_dir = os.path.dirname(config_abs)
        resolved = None
        for _ in range(10):
            candidate = os.path.join(search_dir, plugin_dir)
            if os.path.isdir(candidate):
                resolved = candidate
                break
            search_dir = os.path.dirname(search_dir)
        if resolved:
            parent = os.path.dirname(resolved)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            print(f"Plugin found: {resolved}")
        import mmdet3d_plugin  # noqa: F401 — triggers @register_module()

    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model, cfg


def prepare_img_inputs(info, data_root, final_dim=(256, 704), device="cuda"):
    """Prepare BEVerse image inputs from a Bench2Drive frame info.

    Returns:
        img_inputs: tuple of (imgs, rots, trans, intrins, post_rots, post_trans)
        Each has shape (1, 1, 6, ...) for batch=1, seq=1, 6 cameras.
    """
    imgs = []
    rots = []
    trans = []
    intrins = []
    post_rots_list = []
    post_trans_list = []

    H_orig, W_orig = 900, 1600
    fH, fW = final_dim

    for cam_name in B2D_CAMERAS:
        cam_info = info["sensors"][cam_name]

        img_path = os.path.join(data_root, cam_info["data_path"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resize = fW / W_orig
        newW = int(W_orig * resize)
        newH = int(H_orig * resize)
        img = cv2.resize(img, (newW, newH))

        crop_h = max(0, newH - fH)
        crop_w = max(0, (newW - fW) // 2)
        img = img[crop_h : crop_h + fH, crop_w : crop_w + fW]

        img = img.astype(np.float32) / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        imgs.append(img)

        K = cam_info["intrinsic"].copy()
        K[0, :] *= resize
        K[1, :] *= resize
        K[0, 2] -= crop_w
        K[1, 2] -= crop_h
        intrins.append(torch.tensor(K, dtype=torch.float32))

        cam2ego = cam_info["cam2ego"]
        rot = cam2ego[:3, :3]
        tran = cam2ego[:3, 3]
        rots.append(torch.tensor(rot, dtype=torch.float32))
        trans.append(torch.tensor(tran, dtype=torch.float32))

        post_rot = torch.eye(3, dtype=torch.float32)
        post_tran = torch.zeros(3, dtype=torch.float32)
        post_rots_list.append(post_rot)
        post_trans_list.append(post_tran)

    imgs = torch.stack(imgs)[None, None]
    rots = torch.stack(rots)[None, None]
    trans_t = torch.stack(trans)[None, None]
    intrins_t = torch.stack(intrins)[None, None]
    post_rots_t = torch.stack(post_rots_list)[None, None]
    post_trans_t = torch.stack(post_trans_list)[None, None]

    img_inputs = (
        imgs.to(device),
        rots.to(device),
        trans_t.to(device),
        intrins_t.to(device),
        post_rots_t.to(device),
        post_trans_t.to(device),
    )
    return img_inputs


@torch.no_grad()
def extract_bev_single(model, img_inputs):
    """Extract pre-fusion BEV features for a single frame.

    Runs BEVerse backbone + LSS + temporal model, stops before V2X fusion.
    Returns: (64, 128, 128) BEV feature tensor.
    """
    imgs, rots, trans, intrins, post_rots, post_trans = img_inputs
    B, S, N, C, imH, imW = imgs.shape

    x = imgs.view(B * S * N, C, imH, imW)
    x = model.img_backbone(x)
    if hasattr(model, "with_img_neck") and model.with_img_neck:
        x = model.img_neck(x)
        if isinstance(x, list):
            x = x[0]

    _, output_dim, outH, outW = x.shape
    x = x.view(B, S, N, output_dim, outH, outW)

    x = model.transformer(
        [x, rots, trans, intrins, post_rots, post_trans]
    )

    if hasattr(model, "temporal_model"):
        x = x.view(B, S, *x.shape[1:])
        future_egomotion = torch.zeros(B, S, 6, device=x.device)
        img_is_valid = torch.ones(B, S, device=x.device, dtype=torch.bool)
        x = model.temporal_model(
            x, future_egomotion=future_egomotion,
            aug_transform=None, img_is_valid=img_is_valid,
        )
    while x.dim() > 4:
        x = x.squeeze(1)

    return x[0].cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Extract BEVerse pre-fusion BEV features from Bench2Drive"
    )
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path to b2d_e2e_ego_infos_train.pkl")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of bench2drive images")
    parser.add_argument("--config_path", type=str, required=True,
                        help="BEVerse config file path")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="BEVerse checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted BEV .pt files")
    parser.add_argument("--stride_frames", type=int, default=5,
                        help="Extract every N frames (match training dataset stride)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading BEVerse model...")
    model, cfg = load_beverse_model(args.config_path, args.checkpoint, args.device)
    print("Model loaded.")

    print(f"Loading {args.pkl_path} ...")
    with open(args.pkl_path, "rb") as f:
        infos = pickle.load(f)
    print(f"Loaded {len(infos)} frames")

    clips = defaultdict(list)
    for info in infos:
        clips[info["folder"]].append(info)
    for folder in clips:
        clips[folder].sort(key=lambda x: x["frame_idx"])
    print(f"Found {len(clips)} clips")

    dt_target, dt_source = 0.05, 0.1
    t_plan = 40
    total_horizon = t_plan * dt_target
    min_frames = int(np.ceil(total_horizon / dt_source)) + 2

    frames_to_extract = []
    for folder, frames in clips.items():
        if len(frames) < min_frames:
            continue
        for start_idx in range(0, len(frames) - min_frames, args.stride_frames):
            info = frames[start_idx]
            frames_to_extract.append(info)

    print(f"Extracting BEV for {len(frames_to_extract)} frames...")

    n_extracted = 0
    n_skipped = 0
    for info in tqdm(frames_to_extract, desc="Extracting BEV"):
        folder = info["folder"]
        frame_idx = info["frame_idx"]
        out_name = f"{folder.replace('/', '_')}_{frame_idx:05d}.pt"
        out_path = os.path.join(args.output_dir, out_name)

        if os.path.exists(out_path):
            n_skipped += 1
            continue

        try:
            img_inputs = prepare_img_inputs(info, args.data_root, device=args.device)
            bev_feature = extract_bev_single(model, img_inputs)
            torch.save(bev_feature, out_path)
            n_extracted += 1
        except Exception as e:
            print(f"Error extracting {folder}/{frame_idx}: {e}")
            continue

    print(f"Done. Extracted: {n_extracted}, Skipped (existing): {n_skipped}")


if __name__ == "__main__":
    main()
