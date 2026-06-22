"""
CoSafe WoTE loss functions.

Direct port of WoTE_loss.py's offset and imitation reward losses.
Only offset L1 + score CE (no sim reward / agent / map losses).

Reference: WoTE/navsim/agents/WoTE/WoTE_loss.py
"""

import torch
import torch.nn.functional as F


def compute_traj_offset_loss(
    trajectory_offset: torch.Tensor,
    gt_trajectory: torch.Tensor,
    trajectory_anchors: torch.Tensor,
) -> torch.Tensor:
    """Winner-Take-All trajectory offset L1 loss.

    Identical to WoTE's compute_traj_offset_loss:
      1. Find closest anchor to GT (L2 on flattened trajectory)
      2. L1 loss between predicted offset and GT offset for winner only

    Args:
        trajectory_offset: (B, M, T, 3) predicted offsets
        gt_trajectory:     (B, T, 3)    ground truth trajectory
        trajectory_anchors: (M, T, 3)   anchor trajectories
    """
    B = gt_trajectory.shape[0]
    M = trajectory_anchors.shape[0]

    gt_flat = gt_trajectory.reshape(B, -1)
    anchors_flat = trajectory_anchors.reshape(M, -1).unsqueeze(0).expand(B, -1, -1)
    pred_flat = trajectory_offset.reshape(B, M, -1)

    distances = torch.norm(anchors_flat - gt_flat.unsqueeze(1), dim=2)
    winner_idx = distances.argmin(dim=1)
    batch_idx = torch.arange(B, device=gt_trajectory.device)

    selected_anchors = anchors_flat[batch_idx, winner_idx]
    gt_offset = gt_flat - selected_anchors

    selected_pred = pred_flat[batch_idx, winner_idx]

    return F.l1_loss(selected_pred, gt_offset)


def compute_score_loss(
    scores: torch.Tensor,
    gt_trajectory: torch.Tensor,
    trajectory_anchors: torch.Tensor,
) -> torch.Tensor:
    """Imitation reward cross-entropy loss.

    Identical to WoTE's compute_im_reward_loss:
      target = softmax(-L2_distance)
      loss   = -sum(target * log(pred)) / B

    Args:
        scores:             (B, M) predicted softmax scores
        gt_trajectory:      (B, T, 3)
        trajectory_anchors: (M, T, 3)
    """
    B = gt_trajectory.shape[0]
    M = trajectory_anchors.shape[0]

    gt_flat = gt_trajectory.reshape(B, -1).unsqueeze(1)
    anchors_flat = trajectory_anchors.reshape(M, -1).unsqueeze(0).expand(B, -1, -1)

    l2_distances = torch.cdist(anchors_flat, gt_flat, p=2).squeeze(-1)
    reward_targets = torch.softmax(-l2_distances, dim=-1)

    scores_clamped = scores.clamp(1e-6, 1 - 1e-6)
    loss = -torch.sum(reward_targets * scores_clamped.log()) / B

    return loss


def compute_cosafe_wote_loss(
    trajectory_offset: torch.Tensor,
    scores: torch.Tensor,
    gt_trajectory: torch.Tensor,
    trajectory_anchors: torch.Tensor,
    offset_weight: float = 1.0,
    score_weight: float = 0.1,
) -> dict:
    """Combined loss. Weights match WoTE: traj_offset=1.0, offset_im_reward=0.1."""
    offset_loss = compute_traj_offset_loss(
        trajectory_offset, gt_trajectory, trajectory_anchors
    )
    score_loss = compute_score_loss(scores, gt_trajectory, trajectory_anchors)

    total = offset_loss * offset_weight + score_loss * score_weight

    return {
        "offset_loss": offset_loss,
        "score_loss": score_loss,
        "total": total,
    }
