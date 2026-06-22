"""
CoSafe WoTE offset network training script.

Usage:
    python cs_wote_train.py \
        --pkl_path /path/to/b2d_e2e_ego_infos_train.pkl \
        --anchors_path /path/to/trajectory_anchors_128.npy \
        --bev_dir /path/to/extracted_bev_features/ \
        --output_dir /path/to/checkpoints/

Training matches WoTE's setup:
  - AdamW optimizer, weight_decay=1e-4
  - Warmup cosine LR scheduler (3 epochs warmup)
  - WTA offset L1 + score CE loss
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cs_wote_net import CsWoTENet
from cs_wote_loss import compute_cosafe_wote_loss
from cs_wote_dataset import CsWoTEDataset


class WarmupCosLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + cosine annealing. Identical to WoTE's WarmupCosLR."""

    def __init__(self, optimizer, lr, min_lr, epochs, warmup_epochs, last_epoch=-1):
        self.lr = lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1 + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        return [lr for _ in self.optimizer.param_groups]


def train_one_epoch(model, dataloader, optimizer, anchors, device, config):
    model.train()
    total_loss = 0.0
    total_offset = 0.0
    total_score = 0.0
    n_batches = 0

    for batch in dataloader:
        bev = batch["bev_feature"].to(device)
        status = batch["ego_status"].to(device)
        gt_traj = batch["gt_trajectory"].to(device)

        offset, scores = model(bev, status)

        loss_dict = compute_cosafe_wote_loss(
            offset, scores, gt_traj, anchors,
            offset_weight=config.offset_weight,
            score_weight=config.score_weight,
        )

        optimizer.zero_grad()
        loss_dict["total"].backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        total_loss += loss_dict["total"].item()
        total_offset += loss_dict["offset_loss"].item()
        total_score += loss_dict["score_loss"].item()
        n_batches += 1

    return {
        "total": total_loss / n_batches,
        "offset": total_offset / n_batches,
        "score": total_score / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, anchors, device, config):
    model.eval()
    total_loss = 0.0
    total_min_ade = 0.0
    total_min_fde = 0.0
    n_samples = 0

    for batch in dataloader:
        bev = batch["bev_feature"].to(device)
        status = batch["ego_status"].to(device)
        gt_traj = batch["gt_trajectory"].to(device)
        B = gt_traj.shape[0]

        offset, scores = model(bev, status)

        loss_dict = compute_cosafe_wote_loss(
            offset, scores, gt_traj, anchors,
            offset_weight=config.offset_weight,
            score_weight=config.score_weight,
        )
        total_loss += loss_dict["total"].item() * B

        candidates = anchors.unsqueeze(0) + offset
        disp = candidates[..., :2] - gt_traj[:, None, :, :2]
        per_step_dist = disp.norm(dim=-1)
        ade_per_anchor = per_step_dist.mean(dim=-1)
        fde_per_anchor = per_step_dist[:, :, -1]
        total_min_ade += ade_per_anchor.min(dim=-1).values.sum().item()
        total_min_fde += fde_per_anchor.min(dim=-1).values.sum().item()
        n_samples += B

    return {
        "loss": total_loss / n_samples,
        "min_ade": total_min_ade / n_samples,
        "min_fde": total_min_fde / n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="CoSafe WoTE offset network training")

    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, required=True)
    parser.add_argument("--bev_dir", type=str, default=None,
                        help="Directory with pre-extracted BEV .pt files")
    parser.add_argument("--val_pkl_path", type=str, default=None)
    parser.add_argument("--val_bev_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--bev_channels", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ffn", type=int, default=1024)
    parser.add_argument("--num_keyval", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--status_dim", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--offset_weight", type=float, default=1.0)
    parser.add_argument("--score_weight", type=float, default=0.1)

    parser.add_argument("--t_plan", type=int, default=40)
    parser.add_argument("--dt_target", type=float, default=0.05)
    parser.add_argument("--stride_frames", type=int, default=5)

    parser.add_argument("--patience", type=int, default=20,
                        help="Early stop if val min_ADE doesn't improve for N val checks")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = CsWoTEDataset(
        pkl_path=args.pkl_path,
        bev_dir=args.bev_dir,
        t_plan=args.t_plan,
        dt_target=args.dt_target,
        stride_frames=args.stride_frames,
        bev_channels=args.bev_channels,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_pkl_path:
        val_dataset = CsWoTEDataset(
            pkl_path=args.val_pkl_path,
            bev_dir=args.val_bev_dir,
            t_plan=args.t_plan,
            dt_target=args.dt_target,
            stride_frames=args.stride_frames,
            bev_channels=args.bev_channels,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    model = CsWoTENet(
        anchors_path=args.anchors_path,
        bev_channels=args.bev_channels,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        num_keyval=args.num_keyval,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_decoder_layers,
        status_dim=args.status_dim,
    ).to(device)

    anchors = model.trajectory_anchors.detach()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters, {model.num_anchors} anchors")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosLR(
        optimizer, lr=args.lr, min_lr=args.min_lr,
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
    )

    best_min_ade = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, anchors, device, args
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        log = (
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"loss={train_metrics['total']:.4f} "
            f"(offset={train_metrics['offset']:.4f}, score={train_metrics['score']:.4f}) | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_loader, anchors, device, args)
            log += (
                f" | val_loss={val_metrics['loss']:.4f} "
                f"min_ADE={val_metrics['min_ade']:.3f} "
                f"min_FDE={val_metrics['min_fde']:.3f}"
            )
            if val_metrics["min_ade"] < best_min_ade:
                best_min_ade = val_metrics["min_ade"]
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "best.pth"),
                )
                log += " *best*"
            else:
                patience_counter += 1

            if args.patience > 0 and patience_counter >= args.patience:
                print(log)
                print(f"Early stopping: no improvement for {args.patience} "
                      f"val checks. Best min_ADE={best_min_ade:.3f}")
                break

        print(log)

        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"epoch_{epoch+1}.pth"),
            )

    torch.save(
        model.state_dict(),
        os.path.join(args.output_dir, "final.pth"),
    )
    print(f"Training complete. Best min_ADE={best_min_ade:.3f}. "
          f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
