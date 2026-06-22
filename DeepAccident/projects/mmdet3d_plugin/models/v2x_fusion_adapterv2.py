"""
Adapter utilities for converting DeepAccident's V2X data format to CoBEVT format.
Handles coordinate transformation and data reshaping.
"""
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def warp_affine_simple(src, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    Warp features using affine transformation matrix.
    
    Parameters
    ----------
    src : torch.Tensor
        Input feature map with shape (B, C, H, W)
    M : torch.Tensor
        Transformation matrix with shape (B, 2, 3)
    dsize : tuple
        Output size (H_out, W_out)
    mode : str
        Interpolation mode for grid_sample
    padding_mode : str
        Padding mode for grid_sample
    align_corners : bool
        Whether to align corners in grid generation
        
    Returns
    -------
    torch.Tensor
        Warped features with shape (B, C, H_out, W_out)
    """
    B, C, H, W = src.size()
    grid = F.affine_grid(M, [B, C, dsize[0], dsize[1]], 
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, mode=mode, padding_mode=padding_mode,
                        align_corners=align_corners)


def Regroup(dense_feature, record_len, max_len):
    """
    Regroup features from flat format to batched format with padding and mask.
    
    Converts [sum(N_i), C, H, W] to [B, L, C, H, W] where:
    - B: batch size
    - L: max number of agents (padded)
    - N_i: actual number of agents in batch i
    
    Parameters
    ----------
    dense_feature : torch.Tensor
        Concatenated features from all agents, shape [sum(N), C, H, W]
    record_len : torch.Tensor or list
        Number of agents per batch, shape [B]
    max_len : int
        Maximum number of agents (for padding)
        
    Returns
    -------
    regroup_features : torch.Tensor
        Regrouped features with shape [B, L, C, H, W]
    mask : torch.Tensor
        Binary mask indicating valid agents, shape [B, L]
    """
    if isinstance(record_len, torch.Tensor):
        record_len = record_len.cpu().numpy()
    
    cum_sum_len = list(np.cumsum(record_len))
    split_features = torch.tensor_split(dense_feature,
                                        cum_sum_len[:-1])
    
    regroup_features = []
    mask = []

    for split_feature in split_features:
        feature_shape = split_feature.shape

        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        if padding_len > 0:
            padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                         feature_shape[2], feature_shape[3])
            padding_tensor = padding_tensor.to(split_feature.device)
            split_feature = torch.cat([split_feature, padding_tensor], dim=0)

        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    regroup_features = torch.cat(regroup_features, dim=0)
    
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w',
                                 l=max_len)
    
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask


def pose_vec_to_affine(relative_pose, batch_size, num_agents, H=None, W=None, 
                        discrete_ratio=0.8, downsample_rate=1):
    """
    Convert 6DoF pose vectors to 2D affine transformation matrices for BEV.
    
    DeepAccident uses 6DoF pose vectors [tx, ty, tz, rx, ry, rz] where:
    - tx, ty, tz: translation in meters
    - rx, ry, rz: rotation in radians (roll, pitch, yaw)
    
    For BEV (Bird's Eye View), we only need tx, ty, and rz (yaw).
    The affine matrix is normalized for torch.nn.functional.affine_grid.
    
    Parameters
    ----------
    relative_pose : torch.Tensor
        Pose vectors from ego to other agents, shape [B, L, 6]
        Format: [tx, ty, tz, rx, ry, rz] in meters and radians
    batch_size : int
        Batch size
    num_agents : int
        Number of agents per batch
    H : int, optional
        Feature map height in pixels. If None, normalization is skipped.
    W : int, optional
        Feature map width in pixels. If None, normalization is skipped.
    discrete_ratio : float
        Physical distance per pixel (meters/pixel), default 0.8
    downsample_rate : int
        Downsampling rate, default 1
        
    Returns
    -------
    affine_matrix : torch.Tensor
        Normalized affine transformation matrices, shape [B, L, L, 2, 3]
        affine_matrix[b, i, j] transforms from agent j to agent i's frame
        Coordinates are normalized to [-1, 1] for affine_grid
    """
    device = relative_pose.device
    dtype = relative_pose.dtype
    
    affine_matrix = torch.zeros(batch_size, num_agents, num_agents, 2, 3,
                                device=device, dtype=dtype)
    
    tx = relative_pose[..., 0]
    ty = relative_pose[..., 1]
    rz = relative_pose[..., 5]
    
    for b in range(batch_size):
        affine_matrix[b, 0, 0] = torch.tensor([[1, 0, 0], [0, 1, 0]], 
                                              device=device, dtype=dtype)
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    affine_matrix[b, i, j] = torch.tensor([[1, 0, 0], [0, 1, 0]], 
                                                          device=device, dtype=dtype)
                else:
                    
                    if i == 0:
                        cos_yaw = torch.cos(-rz[b, j])
                        sin_yaw = torch.sin(-rz[b, j])
                        tx_inv = -(cos_yaw * tx[b, j] + sin_yaw * ty[b, j])
                        ty_inv = -(-sin_yaw * tx[b, j] + cos_yaw * ty[b, j])
                        
                        affine_matrix[b, i, j, 0, 0] = cos_yaw
                        affine_matrix[b, i, j, 0, 1] = -sin_yaw
                        affine_matrix[b, i, j, 0, 2] = tx_inv
                        affine_matrix[b, i, j, 1, 0] = sin_yaw
                        affine_matrix[b, i, j, 1, 1] = cos_yaw
                        affine_matrix[b, i, j, 1, 2] = ty_inv
                    else:
                        cos_yaw = torch.cos(rz[b, i] - rz[b, j])
                        sin_yaw = torch.sin(rz[b, i] - rz[b, j])
                        
                        dx = tx[b, j] - tx[b, i]
                        dy = ty[b, j] - ty[b, i]
                        tx_rel = cos_yaw * dx + sin_yaw * dy
                        ty_rel = -sin_yaw * dx + cos_yaw * dy
                        
                        affine_matrix[b, i, j, 0, 0] = cos_yaw
                        affine_matrix[b, i, j, 0, 1] = -sin_yaw
                        affine_matrix[b, i, j, 0, 2] = tx_rel
                        affine_matrix[b, i, j, 1, 0] = sin_yaw
                        affine_matrix[b, i, j, 1, 1] = cos_yaw
                        affine_matrix[b, i, j, 1, 2] = ty_rel
    
    if H is not None and W is not None:
        affine_matrix[..., 0, 1] = affine_matrix[..., 0, 1] * H / W
        affine_matrix[..., 1, 0] = affine_matrix[..., 1, 0] * W / H
        
        affine_matrix[..., 0, 2] = affine_matrix[..., 0, 2] / (downsample_rate * discrete_ratio * W) * 2
        affine_matrix[..., 1, 2] = affine_matrix[..., 1, 2] / (downsample_rate * discrete_ratio * H) * 2
    
    return affine_matrix


def normalize_affine_for_grid(affine_matrix, src_size, dst_size):
    """
    Normalize affine matrix for torch.nn.functional.affine_grid.
    
    affine_grid expects normalized coordinates [-1, 1], so we need to
    convert pixel coordinates to normalized coordinates.
    
    Parameters
    ----------
    affine_matrix : torch.Tensor
        Affine matrix in pixel coordinates, shape (..., 2, 3)
    src_size : tuple
        Source image size (H, W)
    dst_size : tuple
        Destination image size (H, W)
        
    Returns
    -------
    torch.Tensor
        Normalized affine matrix for affine_grid
    """
    H_src, W_src = src_size
    H_dst, W_dst = dst_size
    
    device = affine_matrix.device
    dtype = affine_matrix.dtype
    
    norm_src = torch.tensor([
        [2.0 / W_src, 0, -1],
        [0, 2.0 / H_src, -1],
        [0, 0, 1]
    ], device=device, dtype=dtype)
    
    norm_dst_inv = torch.tensor([
        [W_dst / 2.0, 0, W_dst / 2.0],
        [0, H_dst / 2.0, H_dst / 2.0],
        [0, 0, 1]
    ], device=device, dtype=dtype)
    
    affine_3x3 = torch.cat([
        affine_matrix,
        torch.tensor([[[0, 0, 1]]], device=device, dtype=dtype).expand(
            *affine_matrix.shape[:-2], 1, 3)
    ], dim=-2)
    
    normalized = norm_dst_inv @ affine_3x3 @ norm_src
    
    return normalized[..., :2, :]

