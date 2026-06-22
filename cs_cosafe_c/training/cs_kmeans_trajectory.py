"""
K-means trajectory clustering for CoSafe WoTE candidate generation.

Extracts ego expert trajectories from Bench2Drive dataset at 0.1s interval
(matching source 10Hz), and clusters into M anchor trajectories using
MiniBatchKMeans.

Usage:
    python cs_kmeans_trajectory.py \
        --pkl_path Bench2DriveZoo/data/infos/b2d_e2e_ego_infos_train.pkl \
        --output_dir cs_cosafe_c/training/kmeans_output_base/k128 \
        --num_anchors 128 \
        --t_plan 20 \
        --dt_target 0.1
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def build_clip_index(infos: list) -> dict:
    """Group frames by clip folder, sorted by frame_idx."""
    clips = defaultdict(list)
    for info in infos:
        clips[info['folder']].append(info)
    for folder in clips:
        clips[folder].sort(key=lambda x: x['frame_idx'])
    return clips


def interpolate_pose(frames, target_time, dt_source=0.1):
    """Linearly interpolate ego pose at a continuous frame index.

    Args:
        frames: sorted list of frame dicts for one clip.
        target_time: desired time in seconds from clip start.
        dt_source: time between consecutive frames (0.1s for 10Hz).

    Returns:
        (x, y, yaw) interpolated world pose, or None if out of range.
    """
    frame_float = target_time / dt_source
    frame_lo = int(np.floor(frame_float))
    frame_hi = frame_lo + 1

    if frame_lo < 0 or frame_hi >= len(frames):
        return None

    alpha = frame_float - frame_lo
    info_lo = frames[frame_lo]
    info_hi = frames[frame_hi]

    x = (1 - alpha) * info_lo['ego_translation'][0] + alpha * info_hi['ego_translation'][0]
    y = (1 - alpha) * info_lo['ego_translation'][1] + alpha * info_hi['ego_translation'][1]

    yaw_lo = float(info_lo['ego_yaw'])
    yaw_hi = float(info_hi['ego_yaw'])
    dyaw = (yaw_hi - yaw_lo + np.pi) % (2 * np.pi) - np.pi
    yaw = yaw_lo + alpha * dyaw

    return x, y, yaw


def extract_ego_trajectories(clips, t_plan=20, dt_target=0.1, dt_source=0.1,
                             stride_frames=5):
    """Extract ego future trajectories from all clips.

    For each valid starting frame t0, extract T_plan future poses at dt_target
    intervals, transformed to t0's ego-local coordinate frame.

    Args:
        clips: dict of {folder: [sorted frame dicts]}.
        t_plan: number of future trajectory steps.
        dt_target: target time interval per step (seconds).
        dt_source: source frame interval (seconds).
        stride_frames: step between consecutive starting frames.

    Returns:
        np.ndarray of shape (N, t_plan, 3) — trajectories in local coords.
    """
    trajectories = []
    total_horizon_time = t_plan * dt_target
    min_frames_needed = int(np.ceil(total_horizon_time / dt_source)) + 2

    for folder, frames in tqdm(clips.items(), desc="Extracting trajectories"):
        if len(frames) < min_frames_needed:
            continue

        for start_idx in range(0, len(frames) - min_frames_needed, stride_frames):
            info_t0 = frames[start_idx]
            t0_time = start_idx * dt_source

            x0 = info_t0['ego_translation'][0]
            y0 = info_t0['ego_translation'][1]
            yaw0 = float(info_t0['ego_yaw'])
            cos_neg_yaw = np.cos(-yaw0)
            sin_neg_yaw = np.sin(-yaw0)

            future_poses = []
            valid = True
            for step in range(1, t_plan + 1):
                t_future = t0_time + step * dt_target
                pose = interpolate_pose(frames, t_future, dt_source)
                if pose is None:
                    valid = False
                    break

                xw, yw, yaw_w = pose
                dx = xw - x0
                dy = yw - y0

                local_x = cos_neg_yaw * dx - sin_neg_yaw * dy
                local_y = sin_neg_yaw * dx + cos_neg_yaw * dy
                d_yaw = (yaw_w - yaw0 + np.pi) % (2 * np.pi) - np.pi

                future_poses.append([local_x, local_y, d_yaw])

            if valid and len(future_poses) == t_plan:
                trajectories.append(future_poses)

    return np.array(trajectories, dtype=np.float32)


def cluster_trajectories(trajectories, num_anchors=64, batch_size=1024):
    """Run MiniBatchKMeans on flattened trajectories.

    Args:
        trajectories: (N, T, 3) array.
        num_anchors: number of cluster centers M.
        batch_size: MiniBatchKMeans batch size.

    Returns:
        anchors: (M, T, 3) cluster centers.
        labels: (N,) cluster assignments.
    """
    N, T, D = trajectories.shape
    flat = trajectories.reshape(N, -1).astype(np.float32)

    print(f"Running MiniBatchKMeans: {N} samples, {num_anchors} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_anchors,
        random_state=42,
        batch_size=batch_size,
        max_iter=300,
        n_init=3,
    )
    labels = kmeans.fit_predict(flat)
    anchors = kmeans.cluster_centers_.reshape(num_anchors, T, D)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.0f}, std={counts.std():.0f}")

    return anchors, labels


def visualize_anchors(anchors, save_path, title=None):
    """Plot all anchor trajectories on a single figure."""
    M = anchors.shape[0]
    cmap = get_cmap('hsv', M)

    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(M):
        traj = anchors[i]
        ax.plot(traj[:, 0], traj[:, 1],
                marker='o', color=cmap(i),
                alpha=0.6, linewidth=1.5)

    ax.set_title(title or f'All Cluster Centers')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization: {save_path}")


def visualize_cluster_distribution(labels, save_path, num_anchors):
    """Plot histogram of cluster sizes."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(labels, bins=num_anchors, range=(-0.5, num_anchors - 0.5),
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Count')
    ax.set_title('Cluster Size Distribution')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="K-means trajectory clustering for CoSafe WoTE")
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to b2d_infos_{split}.pkl')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save outputs')
    parser.add_argument('--num_anchors', type=int, default=64,
                        help='Number of K-means clusters (M)')
    parser.add_argument('--t_plan', type=int, default=20,
                        help='Number of trajectory steps')
    parser.add_argument('--dt_target', type=float, default=0.1,
                        help='Target time interval per step (seconds)')
    parser.add_argument('--dt_source', type=float, default=0.1,
                        help='Source dataset frame interval (seconds)')
    parser.add_argument('--stride_frames', type=int, default=5,
                        help='Stride between starting frames within each clip')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.pkl_path} ...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} frames")

    clips = build_clip_index(data)
    print(f"Found {len(clips)} clips")
    for folder, frames in clips.items():
        print(f"  {folder}: {len(frames)} frames")

    trajectories = extract_ego_trajectories(
        clips,
        t_plan=args.t_plan,
        dt_target=args.dt_target,
        dt_source=args.dt_source,
        stride_frames=args.stride_frames,
    )
    print(f"Extracted {len(trajectories)} raw ego trajectories, "
          f"shape: {trajectories.shape}")

    if len(trajectories) == 0:
        print("ERROR: No trajectories extracted. Check pkl data.")
        return

    if len(trajectories) < args.num_anchors:
        print(f"WARNING: Only {len(trajectories)} trajectories after filtering, "
              f"less than num_anchors={args.num_anchors}. Reducing num_anchors.")
        args.num_anchors = max(2, len(trajectories) // 2)

    anchors, labels = cluster_trajectories(
        trajectories,
        num_anchors=args.num_anchors,
    )

    M = args.num_anchors
    anchors_path = output_dir / f"trajectory_anchors_{M}.npy"
    np.save(anchors_path, anchors)
    print(f"Saved anchors: {anchors_path}, shape: {anchors.shape}")

    visualize_anchors(
        anchors,
        output_dir / f"trajectory_anchors_{M}.png",
        title=f"CoSafe Trajectory Anchors (M={M}, T={args.t_plan}, "
              f"dt={args.dt_target}s, horizon={args.t_plan * args.dt_target}s)",
    )
    visualize_cluster_distribution(
        labels,
        output_dir / f"cluster_distribution_{M}.png",
        num_anchors=M,
    )

    print("\n=== Anchor Statistics ===")
    endpoint_disp = np.linalg.norm(anchors[:, -1, :2], axis=1)
    endpoint_yaw = np.abs(anchors[:, -1, 2])
    print(f"Endpoint displacement: "
          f"min={endpoint_disp.min():.2f}, max={endpoint_disp.max():.2f}, "
          f"mean={endpoint_disp.mean():.2f} m")
    print(f"Endpoint yaw change:   "
          f"min={endpoint_yaw.min():.3f}, max={endpoint_yaw.max():.3f}, "
          f"mean={endpoint_yaw.mean():.3f} rad")

    n_static = np.sum(endpoint_disp < 1.0)
    n_straight = np.sum((endpoint_disp >= 1.0) & (endpoint_yaw < 0.2))
    n_turning = np.sum((endpoint_disp >= 1.0) & (endpoint_yaw >= 0.2))
    print(f"Categories: {n_static} static, {n_straight} straight, "
          f"{n_turning} turning (out of {M})")


if __name__ == '__main__':
    main()
