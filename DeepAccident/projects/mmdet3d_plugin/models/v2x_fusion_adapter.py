"""
Adapter utilities for converting DeepAccident's V2X data format to CoBEVT format.
Handles coordinate transformation and data reshaping.

Coordinate System (CARLA/DeepAccident):
- X_local = Right (positive is to the right of the vehicle)
- Y_local = Forward (positive is in front of the vehicle)  
- Z_local = Up

In relative_pose (tx, ty, rz):
- tx = rightward offset (positive = right)
- ty = forward offset (positive = front)
- rz = yaw angle
"""
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from ..datasets.utils.geometry import pose_vec2mat


def warp_affine_simple(src, M, dsize, mode='bilinear', padding_mode='zeros', 
                       align_corners=True, spatial_extent=51.2):
    """
    Warp features using transformation matrix.
    
    This follows warper.py's approach:
    1. Generate BEV grid in physical coordinates
    2. Use M.inverse() to transform coordinates
    3. Normalize by spatial_extent
    4. Use grid_sample
    
    Parameters
    ----------
    src : torch.Tensor
        Input feature map with shape (B, C, H, W)
    M : torch.Tensor
        4x4 transformation matrix with shape (B, 4, 4)
        Represents the transform from source frame to target frame
    dsize : tuple
        Output size (H_out, W_out)
    mode : str
        Interpolation mode for grid_sample
    padding_mode : str
        Padding mode for grid_sample
    align_corners : bool
        Whether to align corners in grid generation
    spatial_extent : float
        Physical extent of BEV in meters (half-width), default 51.2
        
    Returns
    -------
    torch.Tensor
        Warped features with shape (B, C, H_out, W_out)
    """
    B = src.size(0)
    H_out, W_out = dsize
    
    dx = 2 * spatial_extent / W_out
    dy = 2 * spatial_extent / H_out
    xs = torch.linspace(-spatial_extent + dx/2, spatial_extent - dx/2, W_out, 
                        device=src.device, dtype=src.dtype).view(1, -1)
    ys = torch.linspace(-spatial_extent + dy/2, spatial_extent - dy/2, H_out,
                        device=src.device, dtype=src.dtype).view(-1, 1)
    bev_grid = torch.stack((xs.expand(H_out, W_out), 
                           ys.expand(H_out, W_out)), dim=2)
    
    points = torch.cat((bev_grid, torch.ones(H_out, W_out, 1, device=src.device, dtype=src.dtype)), 
                       dim=-1).unsqueeze(-1)
    
    M_inv = torch.inverse(M.float())
    
    xy_flow = M_inv[:, :2, [0, 1, 3]]
    
    trans_points = xy_flow.view(B, 1, 1, 2, 3) @ points.unsqueeze(0)
    trans_points = trans_points.squeeze(-1)
    
    trans_points[..., 0] /= spatial_extent
    trans_points[..., 1] /= spatial_extent
    
    warped = F.grid_sample(src, trans_points, mode=mode, 
                          padding_mode=padding_mode, align_corners=align_corners)
    return warped


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
    Convert 6DoF pose vectors to 4x4 transformation matrices for BEV warping.
    
    DeepAccident/CARLA coordinate system:
    - X_local = Right (positive is to the right)
    - Y_local = Forward (positive is in front)
    - rz = yaw angle
    
    relative_pose[b, j] = (tx, ty, tz, rx, ry, rz) represents agent j's pose 
    in ego's coordinate frame.
    
    pose_vec2mat(relative_pose[b, j]) returns T_agent_j_to_ego:
    - P_ego = T_agent_j_to_ego @ P_j
    - This is the matrix form where the translation column is (tx, ty, tz)
    
    warp_affine_simple expects T_agent_to_ego and takes inverse internally
    to get T_ego_to_agent for sampling positions.
    
    Parameters
    ----------
    relative_pose : torch.Tensor
        Pose vectors from ego to other agents, shape [B, L, 6]
        Format: [tx, ty, tz, rx, ry, rz] in meters and radians
    batch_size : int
        Batch size
    num_agents : int
        Number of agents per batch
    H, W, discrete_ratio, downsample_rate : 
        Kept for API compatibility, but not used (normalization done in warp_affine_simple)
        
    Returns
    -------
    transform_matrix : torch.Tensor
        4x4 transformation matrices, shape [B, L, L, 4, 4]
        transform_matrix[b, i, j] is the matrix to pass to warp_affine_simple
        when warping from agent i's perspective with agent j's features.
        
        For i=ego (index 0), transform_matrix[b, 0, j] = T_agent_j_to_ego
        which is what warp_affine_simple expects.
    """
    device = relative_pose.device
    dtype = relative_pose.dtype
    
    transform_matrix = torch.zeros(batch_size, num_agents, num_agents, 4, 4,
                                   device=device, dtype=dtype)
    
    for b in range(batch_size):
        T_agents_to_ego = pose_vec2mat(relative_pose[b])
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    transform_matrix[b, i, j] = torch.eye(4, device=device, dtype=dtype)
                elif i == 0:
                    transform_matrix[b, i, j] = T_agents_to_ego[j]
                elif j == 0:
                    transform_matrix[b, i, j] = torch.inverse(T_agents_to_ego[i])
                else:
                    T_ego_to_i = torch.inverse(T_agents_to_ego[i])
                    T_j_to_ego = T_agents_to_ego[j]
                    transform_matrix[b, i, j] = T_ego_to_i @ T_j_to_ego
    
    return transform_matrix


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

