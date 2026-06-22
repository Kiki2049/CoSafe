"""
CoSafe WoTE training visualization.

Visualization style matches WoTE/navsim/visualization (color palettes,
trajectory rendering, score-based coloring with plasma colormap).

Sub-commands:
    anchors     Plot all trajectory anchors (HSV colormap)
    predict     Visualize model predictions: candidates (plasma) + GT (green)
    curves      Parse training log, plot loss / min-ADE / min-FDE curves

Usage:
    python cs_wote_vis.py anchors \
        --anchors_path trajectory_anchors_128.npy \
        --output_dir vis_output/

    python cs_wote_vis.py predict \
        --checkpoint /data/jhw/Bench2Drive/Bench2DriveZoo/cosafe/training/output/final.pth \
        --anchors_path /data/jhw/Bench2Drive/Bench2DriveZoo/cosafe/training/kmeans_output_base/k128/trajectory_anchors_128.npy \
        --data_root /path/to/bench2drive/ \
        --pkl_path /data/jhw/Bench2Drive/Bench2DriveZoo/data/infos/b2d_e2e_ego_infos_val.pkl \
        --bev_dir /data/jhw/Bench2Drive/Bench2DriveZoo/cosafe/training/extracted_bev_val \
        --output_dir /data/jhw/Bench2Drive/Bench2DriveZoo/cosafe/training/vis_output \
        --num_samples 8

    python cs_wote_vis.py curves \
        --log_path train.log \
        --output_dir vis_output/
"""

import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.cm import get_cmap


NEW_TAB_10 = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#e15759",
    3: "#76b7b2",
    4: "#59a14f",
    5: "#edc948",
    6: "#b07aa1",
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ac",
}

ELLIS_5 = {
    0: "#DE7061",
    1: "#B0E685",
    2: "#4AC4BD",
    3: "#E38C47",
    4: "#699CDB",
}

GT_STYLE = dict(
    color=NEW_TAB_10[4],
    alpha=1.0,
    linewidth=2.0,
    linestyle="-",
    marker="o",
    markersize=5,
    markeredgecolor="black",
    zorder=3,
    label="GT",
)

SELECTED_STYLE = dict(
    color=ELLIS_5[0],
    alpha=1.0,
    linewidth=2.0,
    linestyle="-",
    marker="o",
    markersize=5,
    markeredgecolor="black",
    zorder=3,
    label="Selected",
)

CANDIDATES_CONFIG = dict(
    colormap="plasma",
    line_width=2.0,
    line_style="--",
    line_alpha=1.0,
    low_score_color="#d3d3d3",
    low_score_alpha=0.4,
    score_threshold=0.5,
    zorder=2,
)


def plot_anchors(anchors, save_path, title=None, highlight_idx=None):
    """Plot all anchor trajectories on a single figure.

    Matches WoTE/scripts/miscs/k_means_trajs.py style:
      - HSV colormap, marker='o', alpha=0.6, linewidth=1.5
      - Optional single-trajectory highlight (red foreground, light-blue bg)
    """
    M = anchors.shape[0]
    fig, ax = plt.subplots(figsize=(15, 15))

    if highlight_idx is not None:
        bg_color = (115 / 255, 137 / 255, 177 / 255)
        for i in range(M):
            traj = anchors[i]
            if i == highlight_idx:
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    marker="o", color="red",
                    alpha=0.9, linewidth=5,
                    label=f"Highlighted #{i}",
                )
            else:
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    color=bg_color, alpha=0.9, linewidth=5,
                )
        ax.legend(loc="upper right", fontsize=12)
    else:
        cmap = get_cmap("hsv", M)
        for i in range(M):
            traj = anchors[i]
            ax.plot(
                traj[:, 0], traj[:, 1],
                marker="o", color=cmap(i),
                alpha=0.6, linewidth=1.5,
            )

    ax.set_title(title or f"Trajectory Anchors (M={M})", fontsize=14)
    ax.set_xlabel("X Position (forward, m)", fontsize=12)
    ax.set_ylabel("Y Position (lateral, m)", fontsize=12)
    ax.grid(False)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_anchor_statistics(anchors, save_dir):
    """Plot endpoint displacement histogram + yaw histogram."""
    M = anchors.shape[0]
    endpoint_disp = np.linalg.norm(anchors[:, -1, :2], axis=1)
    endpoint_yaw = np.abs(anchors[:, -1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(endpoint_disp, bins=30, color=ELLIS_5[4], edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Endpoint Displacement (m)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Anchor Endpoint Displacement", fontsize=12)

    axes[1].hist(endpoint_yaw, bins=30, color=ELLIS_5[3], edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Endpoint |Yaw Change| (rad)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Anchor Endpoint Yaw Change", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "anchor_statistics.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def _draw_candidates_on_ax(ax, candidates, scores, gt_trajectory, config=None,
                           show_legend=True, title=None):
    """Draw candidate trajectories + GT + selected on a given ax."""
    if config is None:
        config = CANDIDATES_CONFIG
    M = candidates.shape[0]
    threshold = config["score_threshold"]

    valid_mask = scores > threshold
    if np.any(valid_mask):
        valid_scores = scores[valid_mask]
        cmap = cm.get_cmap(config["colormap"])
        norm = plt.Normalize(vmin=valid_scores.min(), vmax=valid_scores.max())
    else:
        cmap = cm.get_cmap(config["colormap"])
        norm = plt.Normalize(vmin=0, vmax=1)

    for i in range(M):
        traj = candidates[i]
        poses = np.concatenate([np.array([[0, 0]]), traj[:, :2]])
        if scores[i] > threshold and np.any(valid_mask):
            color = cmap(norm(scores[i]))
            ns = (scores[i] - valid_scores.min()) / (
                valid_scores.max() - valid_scores.min() + 1e-8
            )
            alpha = np.clip(ns * config["line_alpha"], 0.15, 1.0)
        else:
            color = config["low_score_color"]
            alpha = config["low_score_alpha"]
        ax.plot(
            poses[:, 1], poses[:, 0],
            color=color, alpha=alpha,
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )

    best_idx = int(np.argmax(scores))
    best_poses = np.concatenate([np.array([[0, 0]]), candidates[best_idx, :, :2]])
    ax.plot(
        best_poses[:, 1], best_poses[:, 0],
        color=SELECTED_STYLE["color"], alpha=SELECTED_STYLE["alpha"],
        linewidth=SELECTED_STYLE["linewidth"], linestyle=SELECTED_STYLE["linestyle"],
        marker=SELECTED_STYLE["marker"], markersize=SELECTED_STYLE["markersize"],
        markeredgecolor=SELECTED_STYLE["markeredgecolor"],
        zorder=SELECTED_STYLE["zorder"],
        label=f"Selected (#{best_idx}, s={scores[best_idx]:.3f})",
    )

    gt_poses = np.concatenate([np.array([[0, 0]]), gt_trajectory[:, :2]])
    ax.plot(
        gt_poses[:, 1], gt_poses[:, 0],
        color=GT_STYLE["color"], alpha=GT_STYLE["alpha"],
        linewidth=GT_STYLE["linewidth"], linestyle=GT_STYLE["linestyle"],
        marker=GT_STYLE["marker"], markersize=GT_STYLE["markersize"],
        markeredgecolor=GT_STYLE["markeredgecolor"],
        zorder=GT_STYLE["zorder"], label="GT",
    )

    ax.plot(0, 0, "s", color=ELLIS_5[0], markersize=10,
            markeredgecolor="black", zorder=4, label="Ego")

    ax.set_aspect("equal")
    ax.invert_xaxis()
    if show_legend:
        ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("Lateral (m)", fontsize=11)
    ax.set_ylabel("Forward (m)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(False)

    if np.any(valid_mask):
        return cmap, norm
    return None, None


def plot_candidates_with_gt(
    candidates,
    scores,
    gt_trajectory,
    save_path,
    title=None,
    instance_front_path=None,
):
    """Visualize trajectory candidates with score-based coloring + GT.

    If instance_front_path is given, shows the instance_front camera image
    side-by-side (left: camera, right: trajectory BEV).

    Args:
        candidates: (M, T, 3) refined trajectories (anchor + offset).
        scores:     (M,) softmax scores.
        gt_trajectory: (T, 3) ground truth.
        save_path:  output image path.
        instance_front_path: optional path to instance_front camera image.
    """
    import cv2

    has_img = instance_front_path and os.path.exists(instance_front_path)

    if has_img:
        fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(16, 8))
        img = cv2.imread(instance_front_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)
        ax_img.set_title("Instance Front", fontsize=13)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
    else:
        fig, ax_traj = plt.subplots(figsize=(8, 8))

    cmap_obj, norm_obj = _draw_candidates_on_ax(
        ax_traj, candidates, scores, gt_trajectory, title=title or "WoTE Prediction"
    )

    if cmap_obj is not None:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_traj, fraction=0.046, pad=0.04).set_label(
            "Trajectory Score", fontsize=11
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_multi_samples(results, save_path, ncols=4):
    """Grid of prediction visualizations for multiple samples.

    If results contain 'instance_front_path', each sample occupies two rows:
    top = instance_front image, bottom = trajectory BEV.
    Otherwise single row per sample (trajectory only).

    Args:
        results: list of dicts with 'candidates', 'scores', 'gt_trajectory', 'idx',
                 and optionally 'instance_front_path'.
        save_path: output image path.
        ncols: columns in grid.
    """
    import cv2

    n = len(results)
    has_imgs = any(
        r.get("instance_front_path") and os.path.exists(r["instance_front_path"])
        for r in results
    )
    rows_per_sample = 2 if has_imgs else 1
    nrows = math.ceil(n / ncols) * rows_per_sample

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx_plot, res in enumerate(results):
        grid_row, c = divmod(idx_plot, ncols)

        if has_imgs:
            ax_img = axes[grid_row * 2, c]
            img_path = res.get("instance_front_path", "")
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax_img.imshow(img)
            else:
                ax_img.text(0.5, 0.5, "No image", transform=ax_img.transAxes,
                            ha="center", va="center", fontsize=10, color="gray")
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(f"Sample {res.get('idx', idx_plot)} - Instance Front",
                             fontsize=10)
            ax_traj = axes[grid_row * 2 + 1, c]
        else:
            ax_traj = axes[grid_row, c]

        _draw_candidates_on_ax(
            ax_traj, res["candidates"], res["scores"], res["gt_trajectory"],
            show_legend=False,
            title=f"Sample {res.get('idx', idx_plot)}",
        )

    total_cells = nrows * ncols
    used = n * rows_per_sample
    for idx_plot in range(n, math.ceil(n / ncols) * ncols):
        grid_row, c = divmod(idx_plot, ncols)
        if has_imgs:
            axes[grid_row * 2, c].axis("off")
            axes[grid_row * 2 + 1, c].axis("off")
        else:
            axes[grid_row, c].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*loss=([\d.]+)\s*"
    r"\(offset=([\d.]+),\s*score=([\d.]+)\)"
    r".*?lr=([\d.eE+-]+)"
    r"(?:.*?val_loss=([\d.]+)\s+min_ADE=([\d.]+)\s+min_FDE=([\d.]+))?"
)


def parse_training_log(log_path):
    """Parse training log lines into structured arrays."""
    epochs, losses, offsets, score_losses, lrs = [], [], [], [], []
    val_epochs, val_losses, val_ades, val_fdes = [], [], [], []

    with open(log_path, "r") as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            epochs.append(ep)
            losses.append(float(m.group(2)))
            offsets.append(float(m.group(3)))
            score_losses.append(float(m.group(4)))
            lrs.append(float(m.group(5)))

            if m.group(6) is not None:
                val_epochs.append(ep)
                val_losses.append(float(m.group(6)))
                val_ades.append(float(m.group(7)))
                val_fdes.append(float(m.group(8)))

    return {
        "epochs": np.array(epochs),
        "loss": np.array(losses),
        "offset_loss": np.array(offsets),
        "score_loss": np.array(score_losses),
        "lr": np.array(lrs),
        "val_epochs": np.array(val_epochs),
        "val_loss": np.array(val_losses),
        "val_min_ade": np.array(val_ades),
        "val_min_fde": np.array(val_fdes),
    }


def plot_training_curves(log_data, save_dir):
    """Plot training loss + validation metrics in WoTE style."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(log_data["epochs"], log_data["loss"],
            color=ELLIS_5[4], linewidth=1.5, label="Train")
    if len(log_data["val_epochs"]) > 0:
        ax.plot(log_data["val_epochs"], log_data["val_loss"],
                color=ELLIS_5[0], linewidth=1.5, marker="o",
                markersize=4, markeredgecolor="black", label="Val")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.set_title("Total Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(log_data["epochs"], log_data["offset_loss"],
            color=NEW_TAB_10[0], linewidth=1.5, label="Offset L1")
    ax.plot(log_data["epochs"], log_data["score_loss"],
            color=NEW_TAB_10[1], linewidth=1.5, label="Score CE")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Component Losses", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(log_data["val_epochs"]) > 0:
        ax.plot(log_data["val_epochs"], log_data["val_min_ade"],
                color=NEW_TAB_10[4], linewidth=1.5, marker="o",
                markersize=4, markeredgecolor="black", label="min-ADE")
        ax.plot(log_data["val_epochs"], log_data["val_min_fde"],
                color=NEW_TAB_10[2], linewidth=1.5, marker="s",
                markersize=4, markeredgecolor="black", label="min-FDE")
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "No validation data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Distance (m)", fontsize=11)
    ax.set_title("Validation Metrics", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(log_data["epochs"], log_data["lr"],
            color=NEW_TAB_10[6], linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("Learning Rate Schedule", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.suptitle("CoSafe WoTE Training", fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def cmd_anchors(args):
    anchors = np.load(args.anchors_path)
    M = anchors.shape[0]
    os.makedirs(args.output_dir, exist_ok=True)

    plot_anchors(
        anchors,
        os.path.join(args.output_dir, f"anchors_{M}.png"),
        title=f"CoSafe Trajectory Anchors (M={M})",
    )
    if args.highlight is not None:
        plot_anchors(
            anchors,
            os.path.join(
                args.output_dir, f"anchors_{M}_highlight_{args.highlight}.png"
            ),
            title=f"Highlighted Anchor #{args.highlight}",
            highlight_idx=args.highlight,
        )
    plot_anchor_statistics(anchors, args.output_dir)


def _get_instance_front_path(data_root, folder, frame_idx):
    """Construct path to instance_front image.

    Pattern: {data_root}/{folder}/camera/instance_front/{frame_idx:05d}.jpg
    Falls back to .png if .jpg doesn't exist.
    """
    if not data_root:
        return None
    base = os.path.join(data_root, folder, "camera", "instance_front")
    for ext in (".jpg", ".png"):
        p = os.path.join(base, f"{frame_idx:05d}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(base, f"{frame_idx:05d}.jpg")


def cmd_predict(args):
    import torch
    from cs_wote_net import CsWoTENet
    from cs_wote_dataset import CsWoTEDataset

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CsWoTENet(
        anchors_path=args.anchors_path,
        bev_channels=args.bev_channels,
        d_model=args.d_model,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    anchors = model.trajectory_anchors.detach().cpu().numpy()

    dataset = CsWoTEDataset(
        pkl_path=args.pkl_path,
        bev_dir=args.bev_dir,
        bev_channels=args.bev_channels,
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)),
                         replace=False)
    indices.sort()

    results = []
    with torch.no_grad():
        for idx in indices:
            sample = dataset[int(idx)]
            bev = sample["bev_feature"].unsqueeze(0).to(device)
            status = sample["ego_status"].unsqueeze(0).to(device)
            gt_traj = sample["gt_trajectory"].numpy()

            offset, scores = model(bev, status)
            offset = offset[0].cpu().numpy()
            scores = scores[0].cpu().numpy()

            candidates = anchors + offset

            folder, start_idx, frames = dataset.samples[int(idx)]
            frame_idx = frames[start_idx]["frame_idx"]
            img_path = _get_instance_front_path(args.data_root, folder, frame_idx)

            plot_candidates_with_gt(
                candidates, scores, gt_traj,
                os.path.join(args.output_dir, f"pred_{idx}.png"),
                title=f"Sample {idx}",
                instance_front_path=img_path,
            )
            results.append(dict(
                candidates=candidates, scores=scores,
                gt_trajectory=gt_traj, idx=int(idx),
                instance_front_path=img_path,
            ))

    if len(results) > 1:
        plot_multi_samples(
            results,
            os.path.join(args.output_dir, "pred_grid.png"),
            ncols=min(4, len(results)),
        )


def cmd_curves(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_data = parse_training_log(args.log_path)
    if len(log_data["epochs"]) == 0:
        print("No training epochs parsed from log.")
        return
    print(f"Parsed {len(log_data['epochs'])} training epochs, "
          f"{len(log_data['val_epochs'])} validation epochs")
    plot_training_curves(log_data, args.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="CoSafe WoTE training visualization"
    )
    sub = parser.add_subparsers(dest="command")

    p_anc = sub.add_parser("anchors", help="Visualize trajectory anchors")
    p_anc.add_argument("--anchors_path", type=str, required=True)
    p_anc.add_argument("--output_dir", type=str, default="vis_output")
    p_anc.add_argument("--highlight", type=int, default=None,
                       help="Index of anchor to highlight")

    p_pred = sub.add_parser("predict",
                            help="Visualize model predictions with candidates")
    p_pred.add_argument("--checkpoint", type=str, required=True)
    p_pred.add_argument("--anchors_path", type=str, required=True)
    p_pred.add_argument("--pkl_path", type=str, required=True)
    p_pred.add_argument("--bev_dir", type=str, default=None)
    p_pred.add_argument("--data_root", type=str, default=None,
                        help="Bench2Drive data root for instance_front images "
                             "(e.g. .../bench2drive/v1/)")
    p_pred.add_argument("--output_dir", type=str, default="vis_output")
    p_pred.add_argument("--num_samples", type=int, default=8)
    p_pred.add_argument("--seed", type=int, default=42)
    p_pred.add_argument("--bev_channels", type=int, default=64)
    p_pred.add_argument("--d_model", type=int, default=256)

    p_cur = sub.add_parser("curves", help="Plot training curves from log")
    p_cur.add_argument("--log_path", type=str, required=True)
    p_cur.add_argument("--output_dir", type=str, default="vis_output")

    args = parser.parse_args()
    if args.command == "anchors":
        cmd_anchors(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "curves":
        cmd_curves(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
