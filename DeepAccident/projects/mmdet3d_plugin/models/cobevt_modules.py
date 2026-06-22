"""
CoBEVT (Cooperative BEV Transformer) fusion modules.
Migrated from HEAL/opencood for V2X cooperative perception.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum
from .v2x_fusion_adapter import Regroup, warp_affine_simple


class PreNormResidual(nn.Module):
    """Layer normalization with residual connection."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class FeedForward(nn.Module):
    """MLP block with GELU activation."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head attention for cooperative perception.
    Supports 3D structure: [agent, window_height, window_width]
    
    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        Number of agents (vehicles, timestamps, or views).
    window_size: int
        Spatial window size for local attention.
    """

    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        if mask is not None:
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))

        attn = self.attend(sim)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)


class SwapFusionBlockMask(nn.Module):
    """
    Swap Fusion Block with window and grid attention.
    Enables efficient long-range interaction for multi-agent cooperation.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out):
        super(SwapFusionBlockMask, self).__init__()

        self.window_size = window_size

        self.window_attention = PreNormResidual(input_dim,
                                                Attention(input_dim, dim_head,
                                                          drop_out,
                                                          agent_size,
                                                          window_size))
        self.window_ffd = PreNormResidual(input_dim,
                                          FeedForward(input_dim, mlp_dim,
                                                      drop_out))
        
        self.grid_attention = PreNormResidual(input_dim,
                                              Attention(input_dim, dim_head,
                                                        drop_out,
                                                        agent_size,
                                                        window_size))
        self.grid_ffd = PreNormResidual(input_dim,
                                        FeedForward(input_dim, mlp_dim,
                                                    drop_out))

    def forward(self, x, mask):
        
        mask_swap = mask
        mask_swap = rearrange(mask_swap,
                              'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')

        mask_swap = mask
        mask_swap = rearrange(mask_swap,
                              'b (w1 x) (w2 y) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.grid_attention(x, mask=mask_swap)
        x = self.grid_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (w1 x) (w2 y)')

        return x


class CoBEVT(nn.Module):
    """
    Cooperative BEV Transformer for V2X fusion.
    Uses swap window/grid attention for efficient multi-agent feature aggregation.
    
    Parameters
    ----------
    args : dict
        Configuration dictionary containing:
        - depth: Number of transformer layers
        - input_dim: Input channel dimension
        - mlp_dim: Hidden dimension for FFN
        - agent_size: Maximum number of agents
        - window_size: Window size for local attention
        - drop_out: Dropout rate
        - dim_head: Dimension per attention head
    """
    def __init__(self, args):
        super(CoBEVT, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args['depth']
        
        input_dim = args['input_dim']
        mlp_dim = args['mlp_dim']
        agent_size = args['agent_size']
        window_size = args['window_size']
        drop_out = args['drop_out']
        dim_head = args['dim_head']

        for i in range(self.depth):
            block = SwapFusionBlockMask(input_dim,
                                        mlp_dim,
                                        dim_head,
                                        window_size,
                                        agent_size,
                                        drop_out)
            self.layers.append(block)
        
        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )

    def forward(self, x, record_len, affine_matrix):
        """
        Forward pass for CoBEVT.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [sum(N), C, H, W]
        record_len : torch.Tensor
            Number of agents per batch [B]
        affine_matrix : torch.Tensor
            Transformation matrices [B, L, L, 4, 4]
            From pose_vec_to_affine(), where affine_matrix[b, ego, j] is the
            4x4 matrix to warp agent j's BEV to ego's coordinate frame.
            
        Returns
        -------
        torch.Tensor
            Fused features [B, C, H, W]
        """
        
        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]

        regroup_feature, mask = Regroup(x, record_len, L)
        
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])

        regroup_feature_new = []
        for b in range(B):
            ego = 0
            regroup_feature_new.append(
                warp_affine_simple(regroup_feature[b], affine_matrix[b, ego], (H, W))
            )
        regroup_feature = torch.stack(regroup_feature_new)

        x = regroup_feature
        for stage in self.layers:
            x = stage(x, mask=com_mask)
        
        return self.mlp_head(x)

