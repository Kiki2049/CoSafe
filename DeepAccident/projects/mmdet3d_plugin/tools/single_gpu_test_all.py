from collections import defaultdict
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import pdb
import time

import numpy as np
import os
import matplotlib as mpl

import cv2

from ..metrics import IntersectionOverUnion, PanopticMetric, IntersectionOverUnion_separate, PanopticMetric_separate
from ..visualize import Visualizer

from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from .multi_gpu_test_all import eval_accidents_all, eval_metrics, traj_mapping
from ..datasets.utils.instance import predict_instance_segmentation_and_trajectories, predict_instance_segmentation_and_trajectories_accident

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    tp_save_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    dataset = data_loader.dataset
    test_mode = dataset.test_submission

    coordinate_system = dataset.coordinate_system

    prog_bar = mmcv.ProgressBar(len(dataset))
    if show:
        if test_mode:
            default_out_dir = 'test_visualize'
        else:
            default_out_dir = 'eval_visualize'

        out_dir = out_dir or default_out_dir
        visualizer = Visualizer(
            out_dir=out_dir, coordinate_system=coordinate_system)

    logging_interval = 5000

    task_enable = model.module.pts_bbox_head.task_enbale
    det_enable = task_enable.get('3dod', False)
    map_enable = task_enable.get('map', False)
    motion_enable = task_enable.get('motion', False)
    det_results = []

    do_separate_evaluation = True

    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion_separate(num_map_class)

    if motion_enable:
        EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
        num_motion_class = 2

        motion_panoptic_metrics = {}
        motion_iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            motion_panoptic_metrics[key] = PanopticMetric_separate(
                n_classes=num_motion_class, temporally_consistent=True)
            motion_iou_metrics[key] = IntersectionOverUnion_separate(num_motion_class)
        motion_prediction_metrics = {"true_positive": [], "false_positive": [], "false_negative": [],
                                     "minADE": [], "pADE": [], "brierADE": [], "minFDE": [], "pFDE": [],
                                     "brierFDE": [], "MR": [], "p-MR": [], "true_positive_calculation": []}
        accident_metrics = {"true_positive": [], "false_positive": [], "false_negative": [], "id_error_sum": [],
                            "position_error_sum": [], "time_error_sum": []}

        motion_eval_count = 0

    latencies = []
    tp_list = []


    ego_agent_idx = 0
    data_attribute_list = []

    for data_idx, data in enumerate(data_loader):
        if data_idx == 0:
            V2X_model = isinstance(data['img_metas'][0],list)
            not_V2X_model = not V2X_model


        data_prefix = data['img_metas'][0].data[0][0]['sample_idx'] if not_V2X_model else \
        data['img_metas'][ego_agent_idx][0].data[0][0]['sample_idx']
        meta_info = data['img_metas'][0].data[0][0] if not_V2X_model else \
            data['img_metas'][ego_agent_idx][0].data[0][0]

        seconds_before_end_of_scenario = (meta_info['scenario_length'] - meta_info['timestamp']) * 0.1
        accident_visibility = meta_info['accident_visibility']

        data_prefix = data['img_metas'][0].data[0][0]['sample_idx'] if not_V2X_model else \
            data['img_metas'][ego_agent_idx][0].data[0][0]['sample_idx']
        if '_infrastructure_' in data_prefix:
            vehicle_name = 'infrastructure'
        elif '_ego_vehicle_behind_' in data_prefix:
            vehicle_name = 'ego_vehicle_behind'
        elif '_other_vehicle_behind_' in data_prefix:
            vehicle_name = 'other_vehicle_behind'
        elif '_ego_vehicle_' in data_prefix:
            vehicle_name = 'ego_vehicle'
        elif '_other_vehicle_' in data_prefix:
            vehicle_name = 'other_vehicle'
        else:
            vehicle_name = 'unknown'

        data_attribute = dict(
            scenario_type=meta_info['scenario_type'],
            town_name=meta_info['town_name'],
            weather=meta_info['weather'],
            time_of_the_day=meta_info['time_of_the_day'],
            collision_status=meta_info['collision_status'],
            junction_type=meta_info['junction_type'],
            trajectory_type=meta_info['trajectory_type'],
            accident_visibility=str(accident_visibility),
            seconds_before_end_of_scenario=str(seconds_before_end_of_scenario),
            vehicle_name=vehicle_name)
        data_attribute_list.append(data_attribute)

        with torch.no_grad():
            if test_mode:
                motion_distribution_targets = None
            else:
                motion_distribution_targets = {
                    'motion_segmentation': data['motion_segmentation'][0] if not_V2X_model else data['motion_segmentation'][ego_agent_idx][0],
                    'motion_instance': data['motion_instance'][0] if not_V2X_model else data['motion_instance'][ego_agent_idx][0],
                    'instance_centerness': data['instance_centerness'][0] if not_V2X_model else data['instance_centerness'][ego_agent_idx][0],
                    'instance_offset': data['instance_offset'][0] if not_V2X_model else data['instance_offset'][ego_agent_idx][0],
                    'instance_flow': data['instance_flow'][0] if not_V2X_model else data['instance_flow'][ego_agent_idx][0],
                    'future_egomotion': data['future_egomotions'][0] if not_V2X_model else data['future_egomotions'][ego_agent_idx][0],
                }

            if not_V2X_model:
                result = model(
                    return_loss=False,
                    rescale=True,
                    img_metas=data['img_metas'],
                    img_inputs=data['img_inputs'],
                    future_egomotions=data['future_egomotions'],
                    motion_targets=motion_distribution_targets,
                    img_is_valid=data['img_is_valid'][0],
                )
            else:
                result = model(
                    return_loss=False,
                    rescale=True,
                    img_metas=data['img_metas'],
                    img_inputs=data['img_inputs'],
                    future_egomotions=data['future_egomotions'],
                    motion_targets=motion_distribution_targets,
                    img_is_valid=[img_valid[0] for img_valid in data['img_is_valid']],
                    relative_pose_to_ego=[relative_pose[0] for relative_pose in data['relative_pose_to_ego']],
                )

            time_stats = result['time_stats']
            num_input_frame = data['img_inputs'][0][0].shape[1] if not_V2X_model else data['img_inputs'][ego_agent_idx][0][0].shape[1]
            latency = (time_stats['t_BEV'] - time_stats['t0']) / \
                num_input_frame + time_stats['t_end'] - time_stats['t_BEV']

            latencies.append(latency)

            probs = np.ones(len(result['motion_predictions']))

        if det_enable:
            det_results.extend(result['bbox_results'])

        if motion_enable:
            motion_targets = {
                'motion_segmentation': data['motion_segmentation'][0] if not_V2X_model else
                data['motion_segmentation'][ego_agent_idx][0],
                'motion_instance': data['motion_instance'][0] if not_V2X_model else
                data['motion_instance'][ego_agent_idx][0],
                'instance_centerness': data['instance_centerness'][0] if not_V2X_model else
                data['instance_centerness'][ego_agent_idx][0],
                'instance_offset': data['instance_offset'][0] if not_V2X_model else
                data['instance_offset'][ego_agent_idx][0],
                'instance_flow': data['instance_flow'][0] if not_V2X_model else data['instance_flow'][ego_agent_idx][0],
                'future_egomotion': data['future_egomotions'][0] if not_V2X_model else
                data['future_egomotions'][ego_agent_idx][0],
            }

            motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                motion_targets, center_sampling_mode='bilinear')

            if not_V2X_model:
                flow_start_idx = data['future_egomotions'][0].shape[1] - data['motion_segmentation'][0].shape[1]
            else:
                flow_start_idx = data['future_egomotions'][ego_agent_idx][0].shape[1] - \
                                 data['motion_segmentation'][ego_agent_idx][0].shape[1]
            img_metas = data['img_metas'][0].data[0][0] if not_V2X_model else data['img_metas'][ego_agent_idx][0].data[0][0]

            fisrt_invalid_frame = (img_metas['scenario_length'] - img_metas['timestamp']) // 5 + 1

            if fisrt_invalid_frame != motion_labels['segmentation'].shape[2]:
                motion_labels['segmentation'] = motion_labels['segmentation'][:, :fisrt_invalid_frame]
                motion_labels['instance'] = motion_labels['instance'][:, :fisrt_invalid_frame]
                motion_labels['centerness'] = motion_labels['centerness'][:, :fisrt_invalid_frame]
                motion_labels['offset'] = motion_labels['offset'][:, :fisrt_invalid_frame]
                motion_labels['flow'] = motion_labels['flow'][:, :fisrt_invalid_frame]

                if len(result['motion_segmentation']) > 1:
                    for pred_idx in range(len(result['motion_segmentation'])):
                        result['motion_predictions'][pred_idx]['segmentation'] = result['motion_predictions'][pred_idx]['segmentation'][:,
                                                                       :fisrt_invalid_frame]
                        result['motion_predictions'][pred_idx]['instance_center'] = result['motion_predictions'][pred_idx][
                                                                              'instance_center'][:,
                                                                          :fisrt_invalid_frame]
                        result['motion_predictions'][pred_idx]['instance_offset'] = result['motion_predictions'][pred_idx][
                                                                              'instance_offset'][:,
                                                                          :fisrt_invalid_frame]
                        result['motion_predictions'][pred_idx]['instance_flow'] = result['motion_predictions'][pred_idx]['instance_flow'][:,
                                                                        :fisrt_invalid_frame]
                        result['motion_segmentation'][pred_idx] = result['motion_segmentation'][pred_idx][:, :fisrt_invalid_frame]
                        result['motion_instance'][pred_idx] = result['motion_instance'][pred_idx][:, :fisrt_invalid_frame]
                else:
                    result['motion_predictions']['segmentation'] = result['motion_predictions']['segmentation'][:,
                                                                   :fisrt_invalid_frame]
                    result['motion_predictions']['instance_center'] = result['motion_predictions']['instance_center'][:,
                                                                      :fisrt_invalid_frame]
                    result['motion_predictions']['instance_offset'] = result['motion_predictions']['instance_offset'][:,
                                                                      :fisrt_invalid_frame]
                    result['motion_predictions']['instance_flow'] = result['motion_predictions']['instance_flow'][:,
                                                                    :fisrt_invalid_frame]
                    result['motion_segmentation'] = result['motion_segmentation'][:, :fisrt_invalid_frame]
                    result['motion_instance'] = result['motion_instance'][:, :fisrt_invalid_frame]

            if len(result['motion_segmentation']) > 1:
                motion_segmentation, motion_instance = result['motion_segmentation'][-1], result['motion_instance'][-1]
            else:
                motion_segmentation, motion_instance = result['motion_segmentation'], result['motion_instance']

            has_invalid_frame = data['has_invalid_frame'][0] if not_V2X_model else data['has_invalid_frame'][ego_agent_idx][0]
            if True:
                motion_eval_count += 1
                if not test_mode:
                    for key, grid in EVALUATION_RANGES.items():
                        limits = slice(grid[0], grid[1])
                        motion_panoptic_metrics[key].update(motion_instance[..., limits, limits].contiguous(
                        ), motion_labels['instance'][..., limits, limits].contiguous().cuda())

                        motion_iou_metrics[key].update(motion_segmentation[..., limits, limits].contiguous(
                        ), motion_labels['segmentation'][..., limits, limits].contiguous().cuda())
                else:
                    motion_labels = None

        segmentation_binary = motion_labels['segmentation']
        segmentation = segmentation_binary.new_zeros(
            segmentation_binary.shape).repeat(1, 1, 2, 1, 1)
        segmentation[:, :, 0] = (segmentation_binary[:, :, 0] == 0)
        segmentation[:, :, 1] = (segmentation_binary[:, :, 0] == 1)
        motion_labels['segmentation'] = segmentation.float() * 10
        motion_labels['instance_center'] = motion_labels['centerness']
        motion_labels['instance_offset'] = motion_labels['offset']
        motion_labels['instance_flow'] = motion_labels['flow']

        for _ in range(data_loader.batch_size):
            prog_bar.update()


        if show:
            if True:
                img_metas = data['img_metas'][0].data[0][0] if not_V2X_model else data['img_metas'][ego_agent_idx][0].data[0][0]
                visualizer.visualize_deepaccident(
                    img_metas=img_metas,
                    bbox_results=result['bbox_results'][0],
                    gt_bboxes_3d=data['gt_bboxes_3d'][0] if not_V2X_model else data['gt_bboxes_3d'][ego_agent_idx][0],
                    gt_labels_3d=data['gt_labels_3d'][0] if not_V2X_model else data['gt_labels_3d'][ego_agent_idx][0],
                    motion_labels=motion_labels,
                    motion_preds=result['motion_predictions'],
                    save_path='{:04d}_'.format(data_idx+1)+img_metas['sample_idx']
                )

        matchestraj = list()

        consistent_instance_seg_label, matched_centers_label, segmentpixelscnt_label = predict_instance_segmentation_and_trajectories_accident(
            motion_labels, compute_matched_centers=True)

        consistent_instance_seg_pred_list, matched_centers_pred_list, segmentpixelscnt_pred_list = [], [], []
        for idx in range(len(result['motion_predictions'])):
            consistent_instance_seg_pred, matched_centers_pred, segmentpixelscnt_pred = predict_instance_segmentation_and_trajectories_accident(
                result['motion_predictions'][idx], compute_matched_centers=True)
            consistent_instance_seg_pred_list.append(consistent_instance_seg_pred)
            matched_centers_pred_list.append(matched_centers_pred)
            segmentpixelscnt_pred_list.append(segmentpixelscnt_pred)

        for idx in range(len(result['motion_predictions'])):
            matches, distances, unmpred, unmgt = traj_mapping(matched_centers_pred_list[idx], matched_centers_label,
                                                              consistent_instance_seg_pred_list[idx],
                                                              consistent_instance_seg_label)

            matchestraj.append(matches)


        mappingall = dict()
        for gid in matched_centers_label.keys():
            mappingall[gid] = [-1] * len(result['motion_predictions'])
            for idx, m in enumerate(matchestraj):
                for p, g in m:
                    if g == gid:
                        mappingall[gid][idx] = p
                        break

        metrics = [len(matches), len(unmpred), len(unmgt)]

        metrics += eval_metrics(matched_centers_pred_list, matched_centers_label, mappingall, probs)

        for val, metric in zip(metrics, motion_prediction_metrics.keys()):
            motion_prediction_metrics[metric].append(val)

        accident_metric_single = \
            eval_accidents_all(matched_centers_pred_list, matched_centers_label, consistent_instance_seg_pred_list,
                           consistent_instance_seg_label, matchestraj)
        if 1 in accident_metric_single[0]:
            print('true positive prediction!!!')
            print('tp: ', accident_metric_single[0])
            print('id_error: ', accident_metric_single[-3])
            print('position_error: ', accident_metric_single[-2])
            print('time_error: ', accident_metric_single[-1])
            key_list = ['accident_visibility', 'scenario_type', 'town_name', 'weather', 'time_of_the_day', 'collision_status', 'junction_type', 'trajectory_type']
            tp_single = '%04d_'%(data_idx+1) + data_prefix

            for key_name in key_list:
                tp_single += ' ' + data_attribute[key_name]
            print(tp_single)
            tp_list.append(tp_single)

        for val, metric in zip(accident_metric_single, accident_metrics.keys()):
            accident_metrics[metric].append(val)

        if (data_idx + 1) % logging_interval == 0:
            if map_enable:
                scores = semantic_map_iou_val.compute()
                mIoU = sum(scores[1:]) / (len(scores) - 1)
                print('[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
                    data_idx + 1, len(dataset), scores, mIoU,
                ))

            if motion_enable:
                print(
                    '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(motion_eval_count, len(dataset)))

                for key, grid in EVALUATION_RANGES.items():
                    results_str = 'grid = {}: '.format(key)

                    panoptic_scores = motion_panoptic_metrics[key].compute()
                    iou_scores = motion_iou_metrics[key].compute()

                    results_str += 'iou = {:.3f}, '.format(
                        iou_scores[1].item() * 100)

                    for panoptic_key, value in panoptic_scores.items():
                        results_str += '{} = {:.3f}, '.format(
                            panoptic_key, value[1].item() * 100)

                    print(results_str)

            robust_latencies = latencies[20:]
            avg_latency = sum(robust_latencies) / len(robust_latencies)
            print(
                ", average forward time = {:.2f}, fps = {:.2f}".format(
                    avg_latency,
                    1 / avg_latency,
                )
            )

    if map_enable:
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        print('\n[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
            len(dataset), len(dataset), scores, mIoU,
        ))

    attributes_counter = {}
    for key_counter in data_attribute_list[0].keys():
        attributes_counter[key_counter] = {}
    for data_attribute_idx, data_attribute_single in enumerate(data_attribute_list):
        for attribute_name, attribute_val in data_attribute_single.items():
            if attribute_val not in attributes_counter[attribute_name].keys():
                attributes_counter[attribute_name][attribute_val] = [data_attribute_idx]
            else:
                attributes_counter[attribute_name][attribute_val].append(data_attribute_idx)
    print(attributes_counter)

    for attribute_name, attribute_val in attributes_counter.items():
        print('Evaluation divided by %s' % attribute_name)
        for attribute_val_single, selected_index in attribute_val.items():
            print('\nEvaluation on %s: %s!!!!!!!' % (attribute_name, attribute_val_single))
            print('\n[Data length for this attribute value {:04d} / {:04d}]'.format(len(selected_index), len(dataset)))
        print('----------')

    if motion_enable:
        if do_separate_evaluation:
            for attribute_name, attribute_val in attributes_counter.items():
                print('Evaluation divided by %s' % attribute_name)
                for attribute_val_single, selected_index in attribute_val.items():
                    print('\nEvaluation on %s: %s!!!!!!!' % (attribute_name, attribute_val_single))
                    print('\n[Validation {:04d} / {:04d}]: motion metrics: '.format(len(selected_index), len(dataset)))
                    for key, grid in EVALUATION_RANGES.items():
                        results_str = 'grid = {}: '.format(key)
                        panoptic_scores = motion_panoptic_metrics[key].compute(selected_index)
                        iou_scores = motion_iou_metrics[key].compute(selected_index)

                        results_str += 'iou = {:.3f}, '.format(
                            iou_scores[1].item() * 100)

                        for panoptic_key, value in panoptic_scores.items():
                            results_str += '{} = {:.3f}, '.format(
                                panoptic_key, value[1].item() * 100)

                        print(results_str)

                    evalute_based_on_attribute_val(motion_prediction_metrics, accident_metrics, selected_index,
                                                   attribute_name, attribute_val_single)

        print('\nEvaluation on all evaluated data: %s!!!!!!!')
        print('\n[Validation {:04d} / {:04d}]: motion metrics: '.format(motion_eval_count, len(dataset)))
        for key, grid in EVALUATION_RANGES.items():
            results_str = 'grid = {}: '.format(key)

            panoptic_scores = motion_panoptic_metrics[key].compute()
            iou_scores = motion_iou_metrics[key].compute()

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
        print("motion total tp: %d, total fn: %d, total_fp: %d" % (total_tp, total_fn, total_fp))
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
        accident_true_positive = np.array(accident_metrics['true_positive'])
        accident_false_positive = np.array(accident_metrics['false_positive'])
        accident_false_negative = np.array(accident_metrics['false_negative'])
        accident_id_error_sum = np.array(accident_metrics['id_error_sum'])
        accident_position_error_sum = np.array(accident_metrics['position_error_sum'])
        accident_time_error_sum = np.array(accident_metrics['time_error_sum'])

        total_tp = np.sum(accident_true_positive)
        total_fp = np.sum(accident_false_positive)
        total_fn = np.sum(accident_false_negative)

        accident_accuracy = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        print("total tp: %d, total fn: %d, total_fp: %d" % (total_tp, total_fn, total_fp))
        print("accident_accuracy: %.4f" % (accident_accuracy))

        if total_tp > 0:
            avg_id_error = np.sum(accident_id_error_sum) / total_tp
            avg_position_error = np.sum(accident_position_error_sum) / total_tp
            avg_time_error = np.sum(accident_time_error_sum) / total_tp
        else:
            avg_id_error = 1
            avg_position_error = 20
            avg_time_error = 1

        print("avg_id_error: %.4f" % (avg_id_error))
        print("avg_position_error: %.4f" % (avg_position_error))
        print("avg_time_error: %.4f \n" % (avg_time_error))

        if tp_save_dir:
            tp_file_name = tp_save_dir + '/accident_tp_list.txt'
            mmcv.mkdir_or_exist(tp_save_dir)
        else:
            tp_file_name = './accident_tp_list.txt'
        with open(tp_file_name, 'w') as f:
            for accident_tp in tp_list:
                f.write('%s\n'%(accident_tp))

    if not do_separate_evaluation:
        return det_results
    else:
        return det_results, attributes_counter


def evalute_based_on_attribute_val(motion_prediction_metrics, accident_metrics, selected_index, attribute_name, attribute_val):
    print("Evaluation metrics for motion prediction:")
    total_tp = sum([motion_prediction_metrics['true_positive'][selected_index_single] for selected_index_single in selected_index])
    total_fp = sum([motion_prediction_metrics['false_positive'][selected_index_single] for selected_index_single in selected_index])
    total_fn = sum([motion_prediction_metrics['false_negative'][selected_index_single] for selected_index_single in selected_index])
    motion_accuracy = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
    print("motion total tp: %d, total fn: %d, total_fp: %d" % (total_tp, total_fn, total_fp))
    print("motion_accuracy: %.4f" % (motion_accuracy))

    total_tp_cal = sum([motion_prediction_metrics['true_positive_calculation'][selected_index_single] for selected_index_single in selected_index])

    if total_tp > 0:
        avg_Ade_error = sum([motion_prediction_metrics['minADE'][selected_index_single] for selected_index_single in selected_index]) / total_tp_cal
        avg_pAde_error = sum([motion_prediction_metrics['pADE'][selected_index_single] for selected_index_single in selected_index]) / total_tp_cal
        avg_brierAde_error = sum([motion_prediction_metrics['brierADE'][selected_index_single] for selected_index_single in selected_index])/ total_tp_cal
        avg_Fde_error = sum([motion_prediction_metrics['minFDE'][selected_index_single] for selected_index_single in selected_index]) / total_tp_cal
        avg_pFde_error = sum([motion_prediction_metrics['pFDE'][selected_index_single] for selected_index_single in selected_index]) / total_tp_cal
        avg_brierFde_error = sum([motion_prediction_metrics['brierFDE'][selected_index_single] for selected_index_single in selected_index])/ total_tp_cal
        avg_MR_error = sum([motion_prediction_metrics['MR'][selected_index_single] for selected_index_single in selected_index]) / total_tp_cal
        avg_pMR_error = sum([motion_prediction_metrics['p-MR'][selected_index_single] for selected_index_single in selected_index]) / total_tp_cal
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
    accident_true_positive = np.array([accident_metrics['true_positive'][selected_index_single] for selected_index_single in selected_index])
    accident_false_positive = np.array([accident_metrics['false_positive'][selected_index_single] for selected_index_single in selected_index])
    accident_false_negative = np.array([accident_metrics['false_negative'][selected_index_single] for selected_index_single in selected_index])
    accident_id_error_sum = np.array([accident_metrics['id_error_sum'][selected_index_single] for selected_index_single in selected_index])
    accident_position_error_sum = np.array([accident_metrics['position_error_sum'][selected_index_single] for selected_index_single in selected_index])
    accident_time_error_sum = np.array([accident_metrics['time_error_sum'][selected_index_single] for selected_index_single in selected_index])

    total_tp = np.sum(accident_true_positive)
    total_fp = np.sum(accident_false_positive)
    total_fn = np.sum(accident_false_negative)

    accident_accuracy = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
    print("total tp: %d, total fn: %d, total_fp: %d" % (total_tp, total_fn, total_fp))
    print("accident_accuracy: %.4f" % (accident_accuracy))

    if total_tp > 0:
        avg_id_error = np.sum(accident_id_error_sum) / total_tp
        avg_position_error = np.sum(accident_position_error_sum) / total_tp
        avg_time_error = np.sum(accident_time_error_sum) / total_tp
    else:
        avg_id_error = 1
        avg_position_error = 20
        avg_time_error = 1

    print("avg_id_error: %.4f" % (avg_id_error))
    print("avg_position_error: %.4f" % (avg_position_error))
    print("avg_time_error: %.4f \n" % (avg_time_error))
