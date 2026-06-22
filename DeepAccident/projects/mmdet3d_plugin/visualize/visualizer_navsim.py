"""
NavSim-style Visualizer for DeepAccident
Improvements:
1. Map colors: white background + light gray roads
2. Instance rendering: with black outlines
3. Trajectory: with circle markers
"""
import numpy as np
import os
import torch
import imageio
import cv2
from PIL import Image

from ..datasets.utils.instance import (
    predict_instance_segmentation_and_trajectories_accident,
    predict_instance_segmentation_and_trajectories_accident_gt
)


MAP_COLORS = {
    'background': (255, 255, 255),
    'road': (211, 211, 211),
    'lane_marking': (40, 40, 40),
    'sidewalk': (212, 209, 158),
}

NAVSIM_INSTANCE_COLORS = np.array([
    [105, 156, 219],
    [222, 112, 97],
    [89, 161, 79],
    [242, 142, 43],
    [176, 122, 161],
    [118, 183, 178],
    [237, 201, 72],
    [225, 87, 89],
    [78, 121, 167],
    [156, 117, 95],
    [255, 157, 167],
    [186, 176, 172],
    [74, 196, 189],
    [227, 140, 71],
    [176, 230, 133],
], dtype=np.uint8)


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_instance_colours_navsim(instance_map):
    """
    Generate NavSim-style instance colors
    """
    return {
        instance_id: NAVSIM_INSTANCE_COLORS[global_instance_id % len(NAVSIM_INSTANCE_COLORS)]
        for instance_id, global_instance_id in instance_map.items()
    }


def plot_map_navsim(map_file):
    """
    Plot map with NavSim-style colors (white background + light gray roads)
    """
    file = np.load(map_file)
    data_raw = file['data']

    width = data_raw.shape[0]
    cropped_width = width / 60 * 50

    start_idx = int(width // 2 - cropped_width // 2)
    end_idx = int(width // 2 + cropped_width // 2)

    data_raw = data_raw[start_idx:end_idx, start_idx:end_idx]

    img_show = np.ones([data_raw.shape[0], data_raw.shape[1], 3]) * 255

    semantic_layer = data_raw[:, :, 2]
    
    road_mask = (semantic_layer == 7) | (semantic_layer == 10)
    img_show[road_mask] = MAP_COLORS['road']
    
    lane_center_mask = semantic_layer == 6
    img_show[lane_center_mask] = MAP_COLORS['lane_marking']
    
    lane_mask = semantic_layer == 8
    img_show[lane_mask] = MAP_COLORS['lane_marking']

    img_show = cv2.resize(img_show.astype(np.uint8), (200, 200), interpolation=cv2.INTER_AREA)
    img_show = cv2.rotate(img_show, cv2.ROTATE_90_CLOCKWISE)

    return img_show


def plot_instance_map_navsim(instance_seg, instance_map):
    """
    Plot instance segmentation map with NavSim-style colors and black outlines
    """
    h, w = instance_seg.shape
    vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    instance_colours = generate_instance_colours_navsim(instance_map)
    
    for instance_id in instance_map.keys():
        mask = instance_seg == instance_id
        if not mask.any():
            continue
            
        color = instance_colours[instance_id]
        
        vis_image[mask] = color
        
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 0, 0), 1)
    
    return vis_image


def draw_trajectory_line(img, path, color, line_width=3):
    """
    Draw trajectory lines only (without markers)
    """
    color_tuple = tuple(int(c) for c in color)
    
    for t in range(len(path) - 1):
        pt1 = (round(path[t][0]), round(path[t][1]))
        pt2 = (round(path[t + 1][0]), round(path[t + 1][1]))
        cv2.line(img, pt1, pt2, color_tuple, line_width)
    
    return img


def draw_trajectory_markers(img, path, color, marker_radius=2):
    """
    Draw circle markers at each waypoint (NavSim style: marker_size=5 ≈ radius 2-3px)
    Should be called AFTER alpha blending to ensure visibility
    """
    color_tuple = tuple(int(c) for c in color)
    
    for t in range(len(path)):
        pt = (round(path[t][0]), round(path[t][1]))
        cv2.circle(img, pt, marker_radius, color_tuple, -1)
        cv2.circle(img, pt, marker_radius, (0, 0, 0), 1)
    
    return img


def plot_motion_prediction_with_map_navsim(motion_preds, map_img):
    """
    Plot motion prediction with NavSim-style rendering
    - Instance segmentation with black outlines
    - Trajectory with circle markers (drawn after alpha blending for visibility)
    """
    consistent_instance_seg, matched_centers, _ = predict_instance_segmentation_and_trajectories_accident(
        motion_preds,
        compute_matched_centers=True,
        make_consistent=True
    )

    unique_ids = torch.unique(consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours_navsim(instance_map)

    vis_image = plot_instance_map_navsim(
        consistent_instance_seg[0, 0].cpu().numpy(), instance_map
    )

    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    last_seg = consistent_instance_seg[0, -1].cpu().numpy()

    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        color = instance_colours[instance_id]
        
        draw_trajectory_line(trajectory_img, path, color, line_width=2)
        
        mask = last_seg == instance_id
        trajectory_img[mask] = color

    temp_img = cv2.addWeighted(vis_image, 0.6, trajectory_img, 0.4, 1.0)
    mask = ~np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]

    mask = ~np.all(vis_image == 255, axis=2)
    result_img = map_img.copy()
    result_img[mask] = vis_image[mask]

    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        color = instance_colours[instance_id]
        draw_trajectory_markers(result_img, path, color, marker_radius=2)

    return result_img


def plot_motion_label_with_map_navsim(motion_labels, map_img):
    """
    Plot motion ground truth with NavSim-style rendering
    - Instance segmentation with black outlines
    - Trajectory with circle markers (drawn after alpha blending for visibility)
    """
    consistent_instance_seg, matched_centers, _ = predict_instance_segmentation_and_trajectories_accident_gt(
        motion_labels,
        compute_matched_centers=True,
        make_consistent=False
    )

    unique_ids = torch.unique(consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours_navsim(instance_map)

    vis_image = plot_instance_map_navsim(
        consistent_instance_seg[0, 0].cpu().numpy(), instance_map
    )

    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    last_seg = consistent_instance_seg[0, -1].cpu().numpy()

    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        color = instance_colours[instance_id]
        
        draw_trajectory_line(trajectory_img, path, color, line_width=2)
        
        mask = last_seg == instance_id
        trajectory_img[mask] = color

    temp_img = cv2.addWeighted(vis_image, 0.6, trajectory_img, 0.4, 1.0)
    mask = ~np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]

    mask = ~np.all(vis_image == 255, axis=2)
    result_img = map_img.copy()
    result_img[mask] = vis_image[mask]

    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        color = instance_colours[instance_id]
        draw_trajectory_markers(result_img, path, color, marker_radius=2)

    return result_img


def flip_rotate_image(image):
    """Flip and rotate image for correct BEV orientation"""
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.ROTATE_90)
    return np.array(pil_img)

