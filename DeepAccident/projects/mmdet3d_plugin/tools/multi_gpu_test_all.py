import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..metrics import IntersectionOverUnion, PanopticMetric
from ..visualize import Visualizer

from ..datasets.utils.instance import predict_instance_segmentation_and_trajectories, predict_instance_segmentation_and_trajectories_accident
import math
from pprint import pprint
from scipy.optimize import linear_sum_assignment
from skimage.measure import find_contours, approximate_polygon
import matplotlib.pyplot as plt
import shapely.geometry as geo
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sympy import Point, Polygon
import numpy as np
import pdb

from .multi_gpu_test import collect_results_cpu, collect_results_gpu, poly_distance, matchingpair, eval_metrics, interpolate_centers, traj_mapping, create_mapping, _mapping, giou2d, PolyArea2D, flip_rotate_image


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    test_mode = data_loader.dataset.test_submission

    task_enable = model.module.pts_bbox_head.task_enbale
    det_enable = task_enable.get('3dod', False)
    map_enable = task_enable.get('map', False)
    motion_enable = task_enable.get('motion', False)
    det_results = []

    if test_mode:
        map_enable = False
        motion_enable = False

    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()

    if motion_enable:
        EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
        num_motion_class = 2

        motion_panoptic_metrics = {}
        motion_iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            motion_panoptic_metrics[key] = PanopticMetric(
                n_classes=num_motion_class, temporally_consistent=True).cuda()
            motion_iou_metrics[key] = IntersectionOverUnion(
                num_motion_class).cuda()
        motion_prediction_metrics = {"true_positive": [], "false_positive": [], "false_negative": [],
                                     "minADE": [], "pADE": [], "brierADE": [], "minFDE": [], "pFDE": [],
                                     "brierFDE": [], "MR": [], "p-MR": [], "true_positive_calculation": []}
        accident_metrics = {"true_positive": [], "false_positive": [], "false_negative": [], "id_error_sum": [],
                            "position_error_sum": [], "time_error_sum": []}

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    coordinate_system = dataset.coordinate_system
    if show:
        if test_mode:
            default_out_dir = 'test_visualize'
        else:
            default_out_dir = 'eval_visualize'

        out_dir = out_dir or default_out_dir
        visualizer = Visualizer(
            out_dir=out_dir, coordinate_system=coordinate_system)

    time.sleep(2)


    for i, data in enumerate(data_loader):
        data_prefix = data['img_metas'][0].data[0][0]['sample_idx']
        town_name = data_prefix.split('_type')[0].split('_')[-1]
        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                img_metas=data['img_metas'],
                img_inputs=data['img_inputs'],
                future_egomotions=data['future_egomotions'],
                img_is_valid=data['img_is_valid'][0],
            )
            probs = np.ones(1)

        if det_enable:
            det_results.extend(result['bbox_results'])

        if map_enable:
            pred_semantic_indices = result['pred_semantic_indices']
            target_semantic_indices = data['semantic_indices'][0].cuda()

            semantic_map_iou_val(pred_semantic_indices,
                                 target_semantic_indices)

        if motion_enable:
            motion_segmentation, motion_instance = result['motion_segmentation'], result['motion_instance']
            has_invalid_frame = data['has_invalid_frame'][0]
            if not has_invalid_frame.item():

                motion_targets = {
                    'motion_segmentation': data['motion_segmentation'][0],
                    'motion_instance': data['motion_instance'][0],
                    'instance_centerness': data['instance_centerness'][0],
                    'instance_offset': data['instance_offset'][0],
                    'instance_flow': data['instance_flow'][0],
                    'future_egomotion': data['future_egomotions'][0],
                }
                motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                    motion_targets, center_sampling_mode='bilinear')

                for key, grid in EVALUATION_RANGES.items():
                    limits = slice(grid[0], grid[1])
                    motion_panoptic_metrics[key](motion_instance[..., limits, limits].contiguous(
                    ), motion_labels['instance'][..., limits, limits].contiguous().cuda())

                    motion_iou_metrics[key](motion_segmentation[..., limits, limits].contiguous(
                    ), motion_labels['segmentation'][..., limits, limits].contiguous().cuda())

        if show:


            motion_targets = {
                'motion_segmentation': data['motion_segmentation'][0],
                'motion_instance': data['motion_instance'][0],
                'instance_centerness': data['instance_centerness'][0],
                'instance_offset': data['instance_offset'][0],
                'instance_flow': data['instance_flow'][0],
                'future_egomotion': data['future_egomotions'][0],
            }

            motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                motion_targets, center_sampling_mode='bilinear')

            img_metas = data['img_metas'][0].data[0][0]
            visualizer.visualize_beverse(
                img_metas=img_metas,
                bbox_results=result['bbox_results'][0],
                gt_bboxes_3d=data['gt_bboxes_3d'][0],
                gt_labels_3d=data['gt_labels_3d'][0],
                motion_labels=motion_labels,
                motion_preds=result['motion_predictions'],
                save_path='{:04d}_'.format(i + 1) + img_metas['sample_idx']
            )

        motion_targets = {
            'motion_segmentation': data['motion_segmentation'][0],
            'motion_instance': data['motion_instance'][0],
            'instance_centerness': data['instance_centerness'][0],
            'instance_offset': data['instance_offset'][0],
            'instance_flow': data['instance_flow'][0],
            'future_egomotion': data['future_egomotions'][0],
        }
        motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
            motion_targets, center_sampling_mode='bilinear')

        matchestraj = list()

        segmentation_binary = motion_labels['segmentation']
        segmentation = segmentation_binary.new_zeros(
            segmentation_binary.shape).repeat(1, 1, 2, 1, 1)
        segmentation[:, :, 0] = (segmentation_binary[:, :, 0] == 0)
        segmentation[:, :, 1] = (segmentation_binary[:, :, 0] == 1)
        motion_labels['segmentation'] = segmentation.float() * 10
        motion_labels['instance_center'] = motion_labels['centerness']
        motion_labels['instance_offset'] = motion_labels['offset']
        motion_labels['instance_flow'] = motion_labels['flow']

        consistent_instance_seg_label, matched_centers_label, segmentpixelscnt_label = predict_instance_segmentation_and_trajectories_accident(
            motion_labels, compute_matched_centers=True)


        consistent_instance_seg_pred_list, matched_centers_pred_list, segmentpixelscnt_pred_list = [], [], []
        for i in range(len(result['motion_predictions'])):
            consistent_instance_seg_pred, matched_centers_pred, segmentpixelscnt_pred = predict_instance_segmentation_and_trajectories_accident(
                result['motion_predictions'][i], compute_matched_centers=True)
            consistent_instance_seg_pred_list.append(consistent_instance_seg_pred)
            matched_centers_pred_list.append(matched_centers_pred)
            segmentpixelscnt_pred_list.append(segmentpixelscnt_pred)


        for i in range(len(result['motion_predictions'])):
            matches, distances, unmpred, unmgt = traj_mapping(matched_centers_pred_list[i], matched_centers_label,
                                                              consistent_instance_seg_pred_list[i],
                                                              consistent_instance_seg_label)

            matchestraj.append(matches)


        mappingall = dict()
        for gid in matched_centers_label.keys():
            mappingall[gid] = [-1]
            for idx, m in enumerate(matchestraj):
                for p, g in m:
                    if g == gid:
                        mappingall[gid][idx] = p
                        break

        metrics = [len(matches), len(unmpred), len(unmgt)]
        metrics += eval_metrics(matched_centers_pred, matched_centers_label, mappingall, probs)

        for val, metric in zip(metrics, motion_prediction_metrics.keys()):
            motion_prediction_metrics[metric].append(val)


        accident_metric_single = \
            eval_accidents_all(matched_centers_pred, matched_centers_label, consistent_instance_seg_pred,
                                      consistent_instance_seg_label, matchestraj)
        for val, metric in zip(accident_metric_single, accident_metrics.keys()):
            accident_metrics[metric].append(val)


        if rank == 0:
            for _ in range(data_loader.batch_size * world_size):
                prog_bar.update()

    if det_enable:
        if gpu_collect:
            det_results = collect_results_gpu(det_results, len(dataset))
        else:
            det_results = collect_results_cpu(
                det_results, len(dataset), tmpdir)

    if map_enable:
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        if rank == 0:
            print('\n[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
                len(dataset), len(dataset), scores, mIoU,
            ))

    if motion_enable:
        if rank == 0:
            print(
                '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(len(dataset), len(dataset)))

        for key, grid in EVALUATION_RANGES.items():
            results_str = 'grid = {}: '.format(key)

            panoptic_scores = motion_panoptic_metrics[key].compute()
            iou_scores = motion_iou_metrics[key].compute()

            if rank == 0:
                results_str += 'iou = {:.3f}, '.format(
                    iou_scores[1].item() * 100)
                for panoptic_key, value in panoptic_scores.items():
                    results_str += '{} = {:.3f}, '.format(
                        panoptic_key, value[1].item() * 100)
                print(results_str)

        print("Evaluation metrics for motion prediction:")
        total_tp = sum(motion_prediction_metrics['true_positive'])
        total_fp = sum(motion_prediction_metrics['false_positive'])
        total_fn = sum(motion_prediction_metrics['false_negative'])
        motion_accuracy = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        print("motion total tp: %d, total fn: %d, total total_fp: %d" % (total_tp, total_fn, total_fp))
        print("motion_accuracy: %.4f" % (motion_accuracy))

        total_tp_cal = sum(motion_prediction_metrics['true_positive_calculation'])

        if total_tp > 0:
            avg_Ade_error = sum(motion_prediction_metrics['minADE']) / total_tp_cal
            avg_pAde_error = sum(motion_prediction_metrics['pADE']) / total_tp_cal
            avg_brierAde_error = sum(motion_prediction_metrics['brierADE']) / total_tp_cal
            avg_Fde_error = sum(motion_prediction_metrics['minFDE']) / total_tp_cal
            avg_pFde_error = sum(motion_prediction_metrics['pFDE']) / total_tp_cal
            avg_brierFde_error = sum(motion_prediction_metrics['brierFDE']) / total_tp_cal
            avg_MR_error = sum(motion_prediction_metrics['MR']) / total_tp_cal
            avg_pMR_error = sum(motion_prediction_metrics['p-MR']) / total_tp_cal
        else:
            avg_Ade_error = 10000
            avg_pAde_error = 10000
            avg_brierAde_error = 10000
            avg_Fde_error = 10000
            avg_pFde_error = 10000
            avg_brierFde_error = 10000
            avg_MR_error = 1
            avg_pMR_error = 1

        print("avg_Ade_error: %.4f" % (avg_Ade_error))
        print("avg_pAde_error: %.4f" % (avg_pAde_error))
        print("avg_brierAde_error: %.4f" % (avg_brierAde_error))
        print("avg_Fde_error: %.4f" % (avg_Fde_error))
        print("avg_pFde_error: %.4f" % (avg_pFde_error))
        print("avg_brierFde_error: %.4f" % (avg_brierFde_error))
        print("avg_MR_error: %.4f" % (avg_MR_error))
        print("avg_pMR_error: %.4f \n" % (avg_pMR_error))

        print("Evaluation metrics for accident prediction:")
        total_tp = sum(accident_metrics['true_positive'])
        total_fp = sum(accident_metrics['false_positive'])
        total_fn = sum(accident_metrics['false_negative'])
        accident_accuracy = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        print("total tp: %d, total fn: %d, total total_fp: %d"%(total_tp, total_fn, total_fp))
        print("accident_accuracy: %.4f"%(accident_accuracy))

        if total_tp > 0:
            avg_id_error = sum(accident_metrics['id_error_sum']) / total_tp
            avg_position_error = sum(accident_metrics['position_error_sum']) / total_tp
            avg_time_error = sum(accident_metrics['time_error_sum']) / total_tp
        else:
            avg_id_error = 1000
            avg_position_error = 1000
            avg_time_error = 1000

        print("avg_id_error: %.4f"%(avg_id_error))
        print("avg_position_error: %.4f" % (avg_position_error))
        print("avg_time_error: %.4f \n" % (avg_time_error))

    return det_results


def eval_accidents_all(pred_trajs_list, gt_trajs, consistent_instance_seg_pred_list, consistent_instance_seg_label, matchestraj):

    pred_instance_giou_list = []
    number_of_predictions = len(pred_trajs_list)

    for i in range(number_of_predictions):
        pred_instance_giou1 = consistent_instance_seg_pred_list[i][0].clone().detach().cpu().numpy()
        pred_instance_giou_list.append(pred_instance_giou1)

    gt_instance_giou = consistent_instance_seg_label[0].clone().detach().cpu().numpy()
    sequence_length = max(len(t) for t in gt_trajs.values())

    dist_threshold_gt = 0.001
    dist_threshold_pred = 5

    bev_resolution = 0.5
    seconds_per_sample = 0.5

    true_positive = 0
    false_positive = 0
    false_negative = 0

    id_error_sum = 0
    position_error_sum = 0
    time_error_sum = 0

    max_time = 3 * seconds_per_sample
    max_position = 20


    for t in range(sequence_length):
        gt_accident = list()
        pred_accident_total = list()
        for idx, (gt_id_A, gt_traj_A) in enumerate(gt_trajs.items()):
            for gt_id_B, gt_traj_B in list(gt_trajs.items())[idx + 1:]:
                gtdist = poly_distance(gt_id_A, gt_id_B, t, gt_instance_giou)
                if gtdist < dist_threshold_gt:
                    gt_accident.append((t, gt_id_A, gt_id_B, gt_trajs[gt_id_A][t], gt_trajs[gt_id_B][t], gtdist))

    if len(gt_accident) > 1:
        gt_accident.sort(key=lambda x: x[-1])


    if len(pred_trajs_list[0].values()) == 0:
        if len(gt_accident) > 0:
            false_negative = 1
        else:
            false_negative = 0
        return [0], [0], [false_negative], [0], [0], [0]

    for pred_trajs, pred_instance_giou in zip(pred_trajs_list, pred_instance_giou_list):
        pred_accident = list()
        for idx, (pred_id_A, pred_traj_A) in enumerate(pred_trajs.items()):
            for pred_id_B, pred_traj_B in list(pred_trajs.items())[idx + 1:]:
                preddist = poly_distance(pred_id_A, pred_id_B, t, pred_instance_giou)
                if preddist < dist_threshold_pred:
                    pred_accident.append(
                        (t, pred_id_A, pred_id_B, pred_trajs[pred_id_A][t], pred_trajs[pred_id_B][t], preddist))
        if len(pred_accident) > 1:
            pred_accident.sort(key=lambda x: x[-1])
        if len(pred_accident) > 0:
            pred_accident = pred_accident[0]
        pred_accident_total.append(pred_accident)

    if len(gt_accident) > 0:
        gt_accident = gt_accident[0]

    pred_accident_total_valid = []
    for i in range(number_of_predictions):
        if len(pred_accident_total[i]) > 0:
            pred_accident_total_valid.append(i)

    if len(gt_accident) > 0:
        if len(pred_accident_total_valid) == 0:
            false_negative += 1
        else:
            best_pred_idx = -1
            best_id_error = 1
            best_position_err = float("inf")
            
            for i in pred_accident_total_valid:
                pred_accident = pred_accident_total[i]
                if matchingpair(pred_accident[1:3], gt_accident[1:3], matchestraj[i]):
                    if best_id_error > 0:
                        best_id_error = 0
                        best_pred_idx = i
                        position_err1 = np.sqrt(
                            (pred_accident[3][0] - gt_accident[3][0]) ** 2 + (pred_accident[3][1] - gt_accident[3][1]) ** 2) + \
                                        np.sqrt((pred_accident[4][0] - gt_accident[4][0]) ** 2 + (
                                                pred_accident[4][1] - gt_accident[4][1]) ** 2)
                        position_err2 = np.sqrt(
                            (pred_accident[3][0] - gt_accident[4][0]) ** 2 + (
                                    pred_accident[3][1] - gt_accident[4][1]) ** 2) + \
                                        np.sqrt((pred_accident[4][0] - gt_accident[3][0]) ** 2 + (
                                                pred_accident[4][1] - gt_accident[3][1]) ** 2)
                        best_position_err = min(position_err1, position_err2) * bev_resolution
                else:
                    if best_pred_idx == -1:
                        best_pred_idx = i
                        position_err1 = np.sqrt(
                            (pred_accident[3][0] - gt_accident[3][0]) ** 2 + (pred_accident[3][1] - gt_accident[3][1]) ** 2) + \
                                        np.sqrt((pred_accident[4][0] - gt_accident[4][0]) ** 2 + (
                                                pred_accident[4][1] - gt_accident[4][1]) ** 2)
                        position_err2 = np.sqrt(
                            (pred_accident[3][0] - gt_accident[4][0]) ** 2 + (
                                    pred_accident[3][1] - gt_accident[4][1]) ** 2) + \
                                        np.sqrt((pred_accident[4][0] - gt_accident[3][0]) ** 2 + (
                                                pred_accident[4][1] - gt_accident[3][1]) ** 2)
                        best_position_err = min(position_err1, position_err2) * bev_resolution

            if best_id_error == 0:
                true_positive += 1
                pred_accident = pred_accident_total[best_pred_idx]
                position_error_sum += min(best_position_err, max_position)
                time_error_sum += min(abs(pred_accident[0] - gt_accident[0]) * seconds_per_sample, max_time)
            else:
                false_positive += 1
                if best_pred_idx != -1:
                    pred_accident = pred_accident_total[best_pred_idx]
                    id_error_sum += 1
                    position_error_sum += min(best_position_err, max_position)
                    time_error_sum += min(abs(pred_accident[0] - gt_accident[0]) * seconds_per_sample, max_time)
    else:
        if len(pred_accident_total[-1]) != 0:
            false_positive += 1

    return [true_positive], [false_positive], [false_negative], [id_error_sum], [position_error_sum], [time_error_sum]
