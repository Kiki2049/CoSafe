"""Instance segmentation and trajectory extraction utilities.

Migrated from DeepAccident/projects/mmdet3d_plugin/datasets/utils/instance.py
Only the functions needed for inference are included here.

Functions:
    predict_instance_segmentation_and_trajectories: Main entry point
    find_instance_centers: NMS-based center detection from heatmap
    group_pixels: Assign pixels to nearest center via offsets
    get_instance_segmentation_and_centers: Per-frame instance segmentation
    make_instance_id_temporally_consistent: Hungarian matching across time
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def find_instance_centers(
    center_prediction: torch.Tensor,
    conf_threshold: float = 0.1,
    nms_kernel_size: float = 3,
):
    """Find instance centers via NMS on the center heatmap."""
    assert len(center_prediction.shape) == 3
    center_prediction = F.threshold(
        center_prediction, threshold=conf_threshold, value=-1)

    nms_padding = (nms_kernel_size - 1) // 2
    maxpooled_center_prediction = F.max_pool2d(
        center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
    )

    center_prediction[center_prediction != maxpooled_center_prediction] = -1
    return torch.nonzero(center_prediction > 0)[:, 1:]


def group_pixels(centers: torch.Tensor, offset_predictions: torch.Tensor) -> torch.Tensor:
    """Assign each pixel to the nearest center using offset predictions."""
    width, height = offset_predictions.shape[-2:]
    x_grid = (
        torch.arange(width, dtype=offset_predictions.dtype,
                     device=offset_predictions.device)
        .view(1, width, 1)
        .repeat(1, 1, height)
    )
    y_grid = (
        torch.arange(height, dtype=offset_predictions.dtype,
                     device=offset_predictions.device)
        .view(1, 1, height)
        .repeat(1, width, 1)
    )
    pixel_grid = torch.cat((x_grid, y_grid), dim=0)
    center_locations = (pixel_grid + offset_predictions).view(2,
                                                              width * height, 1).permute(2, 1, 0)
    centers = centers.view(-1, 1, 2)

    distances = torch.norm(centers - center_locations, dim=-1)

    instance_id = torch.argmin(distances, dim=0).reshape(1, width, height) + 1
    return instance_id


def update_instance_ids(instance_seg, old_ids, new_ids):
    """Remap instance IDs: old_ids -> new_ids."""
    indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
    for old_id, new_id in zip(old_ids, new_ids):
        indices[old_id] = new_id
    return indices[instance_seg].long()


def make_instance_seg_consecutive(instance_seg):
    """Make instance segment IDs consecutive starting from 0."""
    unique_ids = torch.unique(instance_seg)
    new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg


def get_instance_segmentation_and_centers(
    center_predictions: torch.Tensor,
    offset_predictions: torch.Tensor,
    foreground_mask: torch.Tensor,
    conf_threshold: float = 0.1,
    nms_kernel_size: float = 3,
    max_n_instance_centers: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get instance segmentation and centers for a single frame."""
    width, height = center_predictions.shape[-2:]
    center_predictions = center_predictions.view(1, width, height)
    offset_predictions = offset_predictions.view(2, width, height)
    foreground_mask = foreground_mask.view(1, width, height)

    centers = find_instance_centers(
        center_predictions, conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size)
    if not len(centers):
        return torch.zeros(center_predictions.shape, dtype=torch.int64, device=center_predictions.device), \
            torch.zeros((0, 2), device=centers.device)

    if len(centers) > max_n_instance_centers:
        centers = centers[:max_n_instance_centers].clone()

    instance_ids = group_pixels(centers, offset_predictions)
    instance_seg = (instance_ids * foreground_mask.float()).long()
    instance_seg = make_instance_seg_consecutive(instance_seg)

    return instance_seg.long(), centers


def make_instance_id_temporally_consistent(pred_inst, future_flow, matching_threshold=3.0):
    """Make instance IDs consistent across time using flow-based Hungarian matching.

    Args:
        pred_inst: (1, seq_len, h, w)
        future_flow: (1, seq_len, 2, h, w)
        matching_threshold: distance threshold for a valid match
    Returns:
        consistent_instance_seg: (1, seq_len, h, w)
    """
    assert pred_inst.shape[0] == 1, 'Assumes batch size = 1'

    consistent_instance_seg = [pred_inst[0, 0]]
    largest_instance_id = consistent_instance_seg[0].max().item()

    _, seq_len, h, w = pred_inst.shape
    device = pred_inst.device
    for t in range(seq_len - 1):
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=device),
            torch.arange(w, dtype=torch.float, device=device),
            indexing='ij',
        ))

        grid = grid + future_flow[0, t]
        warped_centers = []
        t_instance_ids = torch.unique(
            consistent_instance_seg[-1])[1:].cpu().numpy()

        if len(t_instance_ids) == 0:
            consistent_instance_seg.append(pred_inst[0, t + 1])
            continue

        for instance_id in t_instance_ids:
            instance_mask = (consistent_instance_seg[-1] == instance_id)
            warped_centers.append(grid[:, instance_mask].mean(dim=1))
        warped_centers = torch.stack(warped_centers)

        centers = []
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=device),
            torch.arange(w, dtype=torch.float, device=device),
            indexing='ij',
        ))
        n_instances = int(pred_inst[0, t + 1].max().item())

        if n_instances == 0:
            consistent_instance_seg.append(pred_inst[0, t + 1])
            continue

        for instance_id in range(1, n_instances + 1):
            instance_mask = (pred_inst[0, t + 1] == instance_id)
            centers.append(grid[:, instance_mask].mean(dim=1))
        centers = torch.stack(centers)

        distances = torch.norm(centers.unsqueeze(
            0) - warped_centers.unsqueeze(1), dim=-1).cpu().numpy()
        ids_t, ids_t_one = linear_sum_assignment(distances)
        matching_distances = distances[ids_t, ids_t_one]
        ids_t += 1
        ids_t_one += 1

        id_mapping = dict(
            zip(np.arange(1, len(t_instance_ids) + 1), t_instance_ids))
        ids_t = np.vectorize(id_mapping.__getitem__, otypes=[np.int64])(ids_t)

        ids_t = ids_t[matching_distances < matching_threshold]
        ids_t_one = ids_t_one[matching_distances < matching_threshold]

        remaining_ids = set(torch.unique(
            pred_inst[0, t + 1]).cpu().numpy()).difference(set(ids_t_one))
        remaining_ids.discard(0)
        for remaining_id in list(remaining_ids):
            largest_instance_id += 1
            ids_t = np.append(ids_t, largest_instance_id)
            ids_t_one = np.append(ids_t_one, remaining_id)

        consistent_instance_seg.append(update_instance_ids(
            pred_inst[0, t + 1], old_ids=ids_t_one, new_ids=ids_t))

    consistent_instance_seg = torch.stack(consistent_instance_seg).unsqueeze(0)
    return consistent_instance_seg


def predict_instance_segmentation_and_trajectories(
        output, compute_matched_centers=False, make_consistent=True, vehicles_id=1,
):
    """Extract instance segmentation and trajectories from model output.

    Args:
        output: dict with keys 'segmentation', 'instance_center',
                'instance_offset', 'instance_flow'
        compute_matched_centers: if True, also return matched center trajectories
        make_consistent: if True, apply temporal consistency via flow
        vehicles_id: foreground class index (1 = vehicles)
    Returns:
        consistent_instance_seg: (B, seq_len, H, W)
        matched_centers: dict {instance_id: [[x,y], ...]} (if compute_matched_centers)
        segmentpixelscnt: dict {instance_id: [count, ...]} (if compute_matched_centers)
    """
    preds = output['segmentation'].detach()
    preds = torch.argmax(preds, dim=2, keepdims=True)
    foreground_masks = preds.squeeze(2) == vehicles_id

    batch_size, seq_len = preds.shape[:2]
    pred_inst = []
    for b in range(batch_size):
        pred_inst_batch = []
        for t in range(seq_len):
            pred_instance_t, centers = get_instance_segmentation_and_centers(
                output['instance_center'][b, t].detach(),
                output['instance_offset'][b, t].detach(),
                foreground_masks[b, t].detach(),
                conf_threshold=0.1,
            )
            pred_inst_batch.append(pred_instance_t)
        pred_inst.append(torch.stack(pred_inst_batch, dim=0))

    pred_inst = torch.stack(pred_inst).squeeze(2)

    if make_consistent:
        if 'instance_flow' not in output or output['instance_flow'] is None:
            output['instance_flow'] = torch.zeros_like(
                output['instance_offset'])
        consistent_instance_seg = []
        for b in range(batch_size):
            consistent_instance_seg.append(
                make_instance_id_temporally_consistent(
                    pred_inst[b:b + 1],
                    output['instance_flow'][b:b + 1].detach(),
                    matching_threshold=10.0,
                )
            )
        consistent_instance_seg = torch.cat(consistent_instance_seg, dim=0)
    else:
        consistent_instance_seg = pred_inst

    if compute_matched_centers:
        assert batch_size == 1
        matched_centers = {}
        _, seq_len, h, w = consistent_instance_seg.shape
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=preds.device),
            torch.arange(w, dtype=torch.float, device=preds.device),
            indexing='ij',
        ))

        segmentpixelscnt = dict()
        for instance_id in torch.unique(consistent_instance_seg[0, 0])[1:].cpu().numpy():
            for t in range(seq_len):
                instance_mask = consistent_instance_seg[0, t] == instance_id
                segmentpixelscnt[instance_id] = segmentpixelscnt.get(instance_id, []) + [
                    instance_mask.sum().detach().cpu().item()]

                if instance_mask.sum() > 0:
                    matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                        grid[:, instance_mask].mean(dim=-1)]

        for key, value in matched_centers.items():
            matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1].tolist()

        instance_flows = {}
        if 'instance_flow' in output and output['instance_flow'] is not None:
            flow_t0 = output['instance_flow'][0, 0].detach()
            for instance_id in torch.unique(consistent_instance_seg[0, 0])[1:].cpu().numpy():
                instance_mask = consistent_instance_seg[0, 0] == instance_id
                if instance_mask.sum() > 0:
                    mean_flow = flow_t0[:, instance_mask].mean(dim=-1)
                    instance_flows[instance_id] = mean_flow.cpu().numpy().tolist()

        return consistent_instance_seg, matched_centers, segmentpixelscnt, instance_flows

    return consistent_instance_seg
