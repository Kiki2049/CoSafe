"""
CoSafe WoTE offset prediction network.

Architecture matches WoTE_model.py's offset prediction pipeline:
  BEV encoder → anchor encoding → cross-attention → offset/score heads.
Replaces TransfuserBackbone with Conv stack for BEVerse pre-fusion BEV features.

Reference: WoTE/navsim/agents/WoTE/WoTE_model.py
"""

import math
import numpy as np
import torch
import torch.nn as nn


class TrajectoryOffsetHead(nn.Module):
    """Predict (x, y, heading) offsets. Identical to WoTE's TrajectoryOffsetHead."""

    def __init__(self, num_poses: int = 40, d_ffn: int = 1024, d_model: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self._num_poses = num_poses
        self._mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, num_poses * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bz, M, _ = x.shape
        poses = self._mlp(x).reshape(bz, M, self._num_poses, 3)
        poses[..., 2] = poses[..., 2].tanh() * math.pi
        return poses


class CsWoTENet(nn.Module):
    """CoSafe WoTE trajectory offset prediction network.

    Flow (identical to WoTE's offset prediction, replacing backbone only):
      1. BEV encoder → bev_tokens (B, num_keyval, d_model)
      2. Status encoding → ego_feat (B, 1, d_model)
      3. Anchor MLP + TransformerEncoder → anchor_feat (B, M, d_model)
      4. cat(ego, anchor) → MLP → query (B, M, d_model)
      5. TransformerDecoder(query, bev_tokens) → refined (B, M, d_model)
      6. OffsetHead → (B, M, T, 3)
      7. ScoreHead + softmax → (B, M)
    """

    def __init__(
        self,
        anchors_path: str,
        bev_channels: int = 64,
        d_model: int = 256,
        num_keyval: int = 64,
        d_ffn: int = 1024,
        num_heads: int = 8,
        num_decoder_layers: int = 3,
        decoder_dropout: float = 0.1,
        status_dim: int = 8,
        encoder_num_layers: int = 2,
        encoder_nhead: int = 8,
        encoder_d_ffn: int = 512,
        encoder_dropout: float = 0.1,
        score_hidden_dim: int = 128,
        feat_dropout: float = 0.1,
    ):
        super().__init__()

        anchors_np = np.load(anchors_path)
        self.num_anchors, self.num_poses, _ = anchors_np.shape
        self.trajectory_anchors = nn.Parameter(
            torch.tensor(anchors_np, dtype=torch.float32), requires_grad=False
        )
        anchor_flat_dim = self.num_poses * 3

        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(feat_dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(feat_dropout),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(feat_dropout),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        h = w = int(num_keyval ** 0.5)
        self.bev_pool = nn.AdaptiveAvgPool2d((h, w))
        self.bev_downscale = nn.Conv2d(512, d_model, kernel_size=1)
        self.keyval_embedding = nn.Embedding(num_keyval, d_model)

        self.status_encoding = nn.Linear(status_dim, d_model)

        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(anchor_flat_dim, score_hidden_dim),
            nn.ReLU(),
            nn.Linear(score_hidden_dim, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=encoder_nhead,
            dim_feedforward=encoder_d_ffn, dropout=encoder_dropout,
            batch_first=True,
        )
        self.cluster_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=encoder_num_layers
        )

        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(feat_dropout),
            nn.Linear(d_model, d_model),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_ffn, dropout=decoder_dropout,
            batch_first=True,
        )
        self.offset_tf_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers
        )

        self.offset_head = TrajectoryOffsetHead(
            num_poses=self.num_poses, d_ffn=d_ffn, d_model=d_model,
            dropout=feat_dropout,
        )

        self.offset_score_head = nn.Sequential(
            nn.Linear(d_model, score_hidden_dim),
            nn.ReLU(),
            nn.Dropout(feat_dropout),
            nn.Linear(score_hidden_dim, 1),
        )

    def _get_bev_tokens(self, bev_feature: torch.Tensor) -> torch.Tensor:
        """BEV feature map → BEV tokens. Replaces WoTE _process_backbone_features."""
        x = self.bev_encoder(bev_feature)
        x = self.bev_pool(x)
        x = self.bev_downscale(x)
        x = x.flatten(-2, -1).permute(0, 2, 1)
        x = x + self.keyval_embedding.weight[None, :, :]
        return x

    def forward(self, bev_feature: torch.Tensor, ego_status: torch.Tensor):
        """
        Args:
            bev_feature: (B, C, H, W) pre-fusion ego-centric BEV
            ego_status:  (B, status_dim)
        Returns:
            trajectory_offset: (B, M, T, 3)
            scores:            (B, M) softmax
        """
        B = bev_feature.shape[0]
        M = self.num_anchors

        bev_tokens = self._get_bev_tokens(bev_feature)

        ego_feat = self.status_encoding(ego_status)[:, None, :]

        anchors_flat = self.trajectory_anchors.reshape(M, -1)
        anchors_batch = anchors_flat.unsqueeze(0).expand(B, -1, -1)
        anchor_feat = self.mlp_planning_vb(anchors_batch)
        anchor_feat = self.cluster_encoder(anchor_feat)

        ego_expanded = ego_feat.expand(-1, M, -1)
        fused = torch.cat([ego_expanded, anchor_feat], dim=-1)
        query = self.encode_ego_feat_mlp(fused)

        query = self.offset_tf_decoder(query, bev_tokens)

        trajectory_offset = self.offset_head(query)

        scores = self.offset_score_head(query).squeeze(-1)
        scores = torch.softmax(scores, dim=-1)

        return trajectory_offset, scores
