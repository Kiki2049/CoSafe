# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from .base_bev_backbone_resnet import ResNetBEVBackbone
from .resblock import ResNetModified, Bottleneck, BasicBlock
from .v2x_fusion_adapter import warp_affine_simple

def regroup(x, record_len):
    """Regroup features from concatenated format to list format."""
    import numpy as np
    if isinstance(record_len, torch.Tensor):
        record_len = record_len.cpu().numpy().tolist()
    cum_sum_len = np.cumsum(record_len)
    split_features = torch.tensor_split(x, cum_sum_len[:-1].tolist())
    return split_features


def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        Transformation matrices from pose_vec_to_affine()
        shape: (B, L, L, 4, 4) 
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                    scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            Transformation matrices from pose_vec_to_affine()
            [B, L, L, 4, 4]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor


        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        
        return fused_feature, occ_map_list 