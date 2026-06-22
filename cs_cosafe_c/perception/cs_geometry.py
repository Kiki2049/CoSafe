"""Geometric transformation utilities for BEV coordinate systems.

Migrated from DeepAccident/projects/mmdet3d_plugin/datasets/utils/geometry.py
with minimal modifications for standalone use in CoSafe.

Functions:
    mat2pose_vec: 4x4 matrix -> 6-DoF vector [tx,ty,tz,rx,ry,rz]
    euler2mat: Euler angles -> 3x3 rotation matrix
    pose_vec2mat: 6-DoF vector -> 4x4 matrix
    invert_matrix_egopose_numpy: Invert 4x4 numpy pose matrix
"""

import numpy as np
import torch


def invert_matrix_egopose_numpy(egopose):
    """Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def mat2pose_vec(matrix: torch.Tensor):
    """Convert a 4x4 pose matrix into a 6-DoF pose vector.

    Args:
        matrix: (..., 4, 4) pose matrix
    Returns:
        (..., 6) vector [tx, ty, tz, rx, ry, rz]
    """
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.

    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [..., 3]
    Returns:
        Rotation matrix [..., 3, 3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros,
                       zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack([cosy, zeros, siny, zeros, ones,
                       zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -
                       sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def pose_vec2mat(vec: torch.Tensor):
    """Convert 6-DoF parameters to 4x4 transformation matrix.

    Args:
        vec: [..., 6] in order [tx, ty, tz, rx, ry, rz]
    Returns:
        [..., 4, 4] transformation matrix
    """
    translation = vec[..., :3].unsqueeze(-1)
    rot = vec[..., 3:].contiguous()
    rot_mat = euler2mat(rot)
    transform_mat = torch.cat([rot_mat, translation], dim=-1)
    transform_mat = torch.nn.functional.pad(
        transform_mat, [0, 0, 0, 1], value=0)
    transform_mat[..., 3, 3] = 1.0
    return transform_mat
