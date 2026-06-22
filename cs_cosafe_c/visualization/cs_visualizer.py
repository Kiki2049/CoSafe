"""BEV visualization aligned with DeepAccident motion_gt / motion_pred format.

Produces three image types per inference step:
- motion_gt_{step}.png:      Road background + CARLA GT vehicle rotated bboxes + velocity arrows
- motion_pred_{step}_{k}.png: Road background + model-predicted instances + trajectories (NavSim style)
- seg_heatmap_{step}.png:    Raw segmentation heatmap (debug)

References:
- DeepAccident visualizer: projects/mmdet3d_plugin/visualize/visualizer_navsim.py
- plot_motion_prediction_with_map_navsim (line 184)
- flip_rotate_image (line 300)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from cs_cosafe_c.perception.cs_bev_transform import world_to_bev


_EGO_COLOR = (0, 200, 0)


def _flip_rotate_image(image):
    """Rotate image 90 degrees CCW (same as flip_rotate_image in visualizer_navsim.py:300)."""
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.ROTATE_90)
    return np.array(pil_img)


class CsVisualizer:
    """BEV prediction visualizer for CoSafe.

    Matches DeepAccident work_dirs visualization output:
    - motion_gt: road + CARLA GT vehicles as colored rotated bboxes
    - motion_pred: road + NavSim-style predicted instances + trajectories
    - seg_heatmap: raw sigmoid output for model diagnosis
    """

    def __init__(self, save_dir: str, canvas_size: Tuple[int, int] = (200, 200)):
        self._save_dir = save_dir
        self._canvas_size = canvas_size
        os.makedirs(save_dir, exist_ok=True)
        self._road_bg = None
        self._plot_pred_fn = None
        self._plot_label_fn = None
        self._flip_rotate_fn = None

    def setup_road_bg(self, rsu_location, grid_conf):
        """Generate and cache road background. Called once from cs_rsu_module.setup()."""
        from cs_cosafe_c.visualization.cs_road_background import generate_road_background
        self._road_bg = generate_road_background(rsu_location, grid_conf)
        print("[CoSafe Vis] Road background generated: shape={}".format(
            self._road_bg.shape))

    def _get_navsim_fns(self):
        """Lazy import of NavSim visualization functions."""
        if self._plot_pred_fn is None:
            from mmdet3d_plugin.visualize.visualizer_navsim import (
                plot_motion_prediction_with_map_navsim,
                flip_rotate_image,
            )
            self._plot_pred_fn = plot_motion_prediction_with_map_navsim
            self._flip_rotate_fn = flip_rotate_image
        return self._plot_pred_fn, self._flip_rotate_fn

    def _get_road_bg(self):
        """Return road background or white fallback."""
        if self._road_bg is not None:
            return self._road_bg.copy()
        h, w = self._canvas_size
        return np.full((h, w, 3), 255, dtype=np.uint8)

    def render(
        self,
        motion_predictions,
        rsu_location=None,
        grid_conf=None,
        ego_vehicle_ids=None,
        step: int = 0,
        rsu_only_motion_predictions=None,
    ) -> str:
        """Full render pipeline producing motion_gt + motion_pred + seg_heatmap.

        Args:
            motion_predictions: list of prediction dicts (K VAE samples) or single dict.
                Each dict has keys: 'segmentation', 'instance_center',
                'instance_offset', 'instance_flow'.
                Can be None (GT-only rendering).
            rsu_location: carla.Location of RSU (needed for GT vehicle drawing).
            grid_conf: motion grid config dict (needed for GT vehicle drawing).
            ego_vehicle_ids: set of vehicle IDs to highlight as ego.
            step: simulation step number for filename.
            rsu_only_motion_predictions: optional RSU-only predictions for comparison.

        Returns:
            path: saved GT image path.
        """
        gt_path = ""
        pred_path = ""

        if rsu_location is not None and grid_conf is not None:
            gt_path = self._render_motion_gt(
                rsu_location, grid_conf, ego_vehicle_ids, step)

        if motion_predictions is not None:
            pred_path = self._render_motion_pred(motion_predictions, step)

        if rsu_only_motion_predictions is not None:
            self._render_motion_pred(
                rsu_only_motion_predictions, step, prefix='motion_pred_rsu_only')

        return gt_path or pred_path or ""


    def _draw_gt_vehicles(self, canvas, rsu_location, grid_conf,
                          ego_vehicle_ids, future_frames=4, frame_dt=0.5):
        """Draw CARLA GT vehicle bboxes + trajectory dots onto canvas (in-place).

        Returns the canvas (pre flip_rotate) for further overlay.
        """
        import carla
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        ego_pose = (rsu_location.x, rsu_location.y, 0.0)
        bev_range = abs(grid_conf['xbound'][0])

        world = CarlaDataProvider.get_world()
        vehicles = world.get_actors().filter('vehicle.*')

        if ego_vehicle_ids is None:
            ego_vehicle_ids = set()

        from mmdet3d_plugin.visualize.visualizer_navsim import NAVSIM_INSTANCE_COLORS

        color_idx = 0
        for veh in vehicles:
            veh_loc = veh.get_location()
            if veh_loc.distance(rsu_location) > bev_range + 5:
                continue

            bb = veh.bounding_box.extent
            corners_local = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
            ]
            veh.get_transform().transform(corners_local)

            bev_corners = []
            for c in corners_local:
                px, py = world_to_bev(c.x, c.y, grid_conf, ego_pose)
                bev_corners.append([int(round(px)), int(round(py))])
            bev_corners = np.array(bev_corners, dtype=np.int32)

            is_ego = veh.id in ego_vehicle_ids
            if is_ego:
                color = _EGO_COLOR
            else:
                color = tuple(int(c) for c in
                              NAVSIM_INSTANCE_COLORS[color_idx % len(NAVSIM_INSTANCE_COLORS)])
                color_idx += 1

            cv2.fillPoly(canvas, [bev_corners], color)
            cv2.drawContours(canvas, [bev_corners], 0, (0, 0, 0), 1)

            vel = veh.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2) ** 0.5
            if speed > 0.5:
                for t in range(1, future_frames + 1):
                    fut_x = veh_loc.x + vel.x * t * frame_dt
                    fut_y = veh_loc.y + vel.y * t * frame_dt
                    fpx, fpy = world_to_bev(fut_x, fut_y, grid_conf, ego_pose)
                    pt = (int(round(fpx)), int(round(fpy)))
                    cv2.circle(canvas, pt, 2, color, -1)
                    cv2.circle(canvas, pt, 2, (0, 0, 0), 1)

        return canvas

    def _render_motion_gt(self, rsu_location, grid_conf, ego_vehicle_ids, step,
                          future_frames=4, frame_dt=0.5):
        """Render motion_gt image: road bg + CARLA GT vehicle bboxes + trajectory dots."""
        canvas = self._get_road_bg()
        canvas = self._draw_gt_vehicles(
            canvas, rsu_location, grid_conf, ego_vehicle_ids,
            future_frames, frame_dt)
        canvas = _flip_rotate_image(canvas)

        path = os.path.join(self._save_dir, 'motion_gt_{:06d}.png'.format(step))
        cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return path


    def _render_motion_pred(self, motion_predictions, step, prefix='motion_pred'):
        """Render motion_pred images using NavSim plot_motion_prediction_with_map_navsim."""
        plot_pred_fn, flip_rotate_fn = self._get_navsim_fns()
        save_path = None

        if isinstance(motion_predictions, list):
            preds_list = motion_predictions
        else:
            preds_list = [motion_predictions]

        for k, motion_pred in enumerate(preds_list):
            try:
                road_bg = self._get_road_bg()
                vis_image = plot_pred_fn(motion_pred, road_bg)
                vis_image = flip_rotate_fn(vis_image)

                path = os.path.join(
                    self._save_dir,
                    '{}_{:06d}_{}.png'.format(prefix, step, k + 1))
                cv2.imwrite(path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                if k == 0:
                    save_path = path
            except Exception as e:
                print("[CoSafe Vis] Error rendering {} {}: {}".format(
                    prefix, k + 1, e))
                fallback = self._get_road_bg()
                fallback = flip_rotate_fn(fallback)
                cv2.putText(fallback, "No instances detected",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (100, 100, 100), 1)
                path = os.path.join(
                    self._save_dir,
                    '{}_{:06d}_{}.png'.format(prefix, step, k + 1))
                cv2.imwrite(path, cv2.cvtColor(fallback, cv2.COLOR_RGB2BGR))
                if k == 0:
                    save_path = path

        if save_path:
            print("[CoSafe Vis] Saved {}: {}".format(prefix, save_path))
        return save_path or ""


    def _save_seg_heatmap(self, pred_dict, step):
        """Save raw segmentation heatmap for debugging.

        Always produces a visible image even when no instances are detected,
        helping diagnose whether the model produces meaningful output.
        """
        try:
            seg = pred_dict.get('segmentation')
            if seg is None:
                return
            seg_t0 = seg[0, 0].detach()
            fg_prob = seg_t0[1].sigmoid().cpu().numpy()
            heatmap = (fg_prob * 255).clip(0, 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_color = _flip_rotate_image(
                cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))

            path = os.path.join(
                self._save_dir, 'seg_heatmap_{:06d}.png'.format(step))
            cv2.imwrite(path, cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print("[CoSafe Vis] Heatmap error: {}".format(e))


    def render_risks(self, perception_output, risk_assessment,
                     grid_conf, rsu_location, step: int = 0,
                     ego_vehicle_ids=None) -> str:
        """Render risk overlay on top of GT vehicle canvas.

        Draws GT vehicles first (same as motion_gt), then overlays:
        - Flagged agents: cyan circle
        - Collision risks: red lines between agent pairs
        - Traffic violations: orange markers
        - Deadlock: purple X markers
        - Single stuck: purple X markers
        """
        if risk_assessment is None or not risk_assessment.risks:
            return ""

        canvas = self._get_road_bg()
        canvas = self._draw_gt_vehicles(
            canvas, rsu_location, grid_conf, ego_vehicle_ids)

        ego_pose = (rsu_location.x, rsu_location.y, 0.0)

        agent_positions = {}
        if perception_output and perception_output.scenes:
            scene = perception_output.scenes[-1]
            for idx, v in enumerate(scene):
                aid = v.instance_id if v.instance_id >= 0 else idx
                px, py = world_to_bev(
                    v.position_world[0], v.position_world[1],
                    grid_conf, ego_pose)
                agent_positions[aid] = (int(round(px)), int(round(py)))

        for aid in risk_assessment.flagged_agents:
            pos = agent_positions.get(aid)
            if pos:
                cv2.circle(canvas, pos, 8, (0, 200, 200), 2)

        for risk in risk_assessment.risks:
            pos_i = agent_positions.get(risk.agent_i)

            if risk.risk_type == 'collision' and risk.agent_j >= 0:
                pos_j = agent_positions.get(risk.agent_j)
                if pos_i and pos_j:
                    cv2.line(canvas, pos_i, pos_j, (255, 0, 0), 2)
                    mid = ((pos_i[0] + pos_j[0]) // 2,
                           (pos_i[1] + pos_j[1]) // 2)
                    cv2.putText(canvas, "{:.1f}".format(risk.severity),
                                mid, cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, (255, 255, 255), 1)

            elif risk.risk_type == 'red_light':
                if pos_i:
                    cv2.circle(canvas, pos_i, 6, (255, 140, 0), -1)
                    cv2.circle(canvas, pos_i, 6, (0, 0, 0), 1)

            elif risk.risk_type in ('deadlock', 'single_stuck'):
                for pos in [pos_i, agent_positions.get(risk.agent_j)]:
                    if pos:
                        x, y = pos
                        cv2.line(canvas, (x - 4, y - 4), (x + 4, y + 4),
                                 (180, 0, 180), 2)
                        cv2.line(canvas, (x - 4, y + 4), (x + 4, y - 4),
                                 (180, 0, 180), 2)

        canvas = _flip_rotate_image(canvas)

        path = os.path.join(
            self._save_dir, 'risk_overlay_{:06d}.png'.format(step))
        cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return path


    def render_wote_candidates(self, wote_results, grid_conf, rsu_location,
                               step: int = 0, ego_vehicle_ids=None) -> str:
        """Render WoTE candidate trajectories on BEV road background.

        For each agent:
        - All M candidates: plasma colormap by score
        - Best candidate: solid red line with dot markers (ELLIS_5[0])
        - Agent position: colored square marker

        Style adapted from cs_wote_vis.py predict rendering to cv2.

        Args:
            wote_results: dict {agent_id: (candidates, scores, vehicle_state)}
                candidates: (M, T, 3) ego-local coords.
                scores: (M,) softmax scores.
                vehicle_state: CsVehicleState.
            grid_conf: motion grid config dict.
            rsu_location: carla.Location of RSU.
            step: simulation step number.
            ego_vehicle_ids: set of vehicle IDs to highlight.

        Returns:
            path: saved image path.
        """
        import math

        if not wote_results or rsu_location is None:
            return ""

        canvas = self._get_road_bg()
        ego_pose = (rsu_location.x, rsu_location.y, 0.0)

        if ego_vehicle_ids is None:
            ego_vehicle_ids = set()

        selected_color = (222, 112, 97)
        bg_color = (180, 180, 180)
        bg_alpha = 0.4

        for agent_id, (candidates, scores, vstate, *_rest) in wote_results.items():
            M = candidates.shape[0]

            wx, wy = vstate.position_world
            heading = vstate.heading_rad
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)

            best_idx = int(np.argmax(scores))

            score_min, score_max = scores.min(), scores.max()
            score_range = score_max - score_min if score_max > score_min else 1.0

            overlay = canvas.copy()
            order = np.argsort(scores)
            for rank, i in enumerate(order):
                if i == best_idx:
                    continue

                traj = candidates[i]
                bev_pts = []
                for t in range(traj.shape[0]):
                    x_local, y_local = traj[t, 0], traj[t, 1]
                    x_world = wx + x_local * cos_h - y_local * sin_h
                    y_world = wy + x_local * sin_h + y_local * cos_h
                    px, py = world_to_bev(x_world, y_world, grid_conf, ego_pose)
                    bev_pts.append((int(round(px)), int(round(py))))

                if len(bev_pts) < 2:
                    continue

                pts_arr = np.array(bev_pts, dtype=np.int32)
                cv2.polylines(overlay, [pts_arr], isClosed=False,
                              color=bg_color, thickness=1)

            cv2.addWeighted(overlay, bg_alpha, canvas, 1.0 - bg_alpha,
                            0, canvas)

            traj_best = candidates[best_idx]
            bev_pts_best = []
            for t in range(traj_best.shape[0]):
                x_local, y_local = traj_best[t, 0], traj_best[t, 1]
                x_world = wx + x_local * cos_h - y_local * sin_h
                y_world = wy + x_local * sin_h + y_local * cos_h
                px, py = world_to_bev(x_world, y_world, grid_conf, ego_pose)
                bev_pts_best.append((int(round(px)), int(round(py))))

            if len(bev_pts_best) >= 2:
                pts_arr_best = np.array(bev_pts_best, dtype=np.int32)
                cv2.polylines(canvas, [pts_arr_best], isClosed=False,
                              color=selected_color, thickness=2)
                for pt in bev_pts_best:
                    cv2.circle(canvas, pt, 2, selected_color, -1)
                    cv2.circle(canvas, pt, 2, (0, 0, 0), 1)

            import carla as carla_mod
            is_ego = agent_id in ego_vehicle_ids
            marker_color = _EGO_COLOR if is_ego else selected_color
            carla_veh = self._find_carla_vehicle_for_vis(agent_id)
            if carla_veh is not None:
                bb = carla_veh.bounding_box.extent
                corners_local = [
                    carla_mod.Location(x=-bb.x, y=-bb.y),
                    carla_mod.Location(x=bb.x, y=-bb.y),
                    carla_mod.Location(x=bb.x, y=bb.y),
                    carla_mod.Location(x=-bb.x, y=bb.y),
                ]
                carla_veh.get_transform().transform(corners_local)
                bev_corners = []
                for c in corners_local:
                    cpx, cpy = world_to_bev(c.x, c.y, grid_conf, ego_pose)
                    bev_corners.append([int(round(cpx)), int(round(cpy))])
                bev_corners = np.array(bev_corners, dtype=np.int32)
            else:
                apx, apy = world_to_bev(wx, wy, grid_conf, ego_pose)
                h = 3
                bev_corners = np.array([
                    [int(apx) - h, int(apy) - h],
                    [int(apx) + h, int(apy) - h],
                    [int(apx) + h, int(apy) + h],
                    [int(apx) - h, int(apy) + h],
                ], dtype=np.int32)
            cv2.fillPoly(canvas, [bev_corners], marker_color)
            cv2.drawContours(canvas, [bev_corners], 0, (0, 0, 0), 1)

        canvas = _flip_rotate_image(canvas)

        path = os.path.join(
            self._save_dir, 'wote_candidates_{:06d}.png'.format(step))
        cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print("[CoSafe Vis] Saved WoTE candidates: {}".format(path))
        return path

    @staticmethod
    def _find_carla_vehicle_for_vis(actor_id):
        """Look up a CARLA vehicle actor by ID for visualization bbox."""
        try:
            from srunner.scenariomanager.carla_data_provider import (
                CarlaDataProvider)
            world = CarlaDataProvider.get_world()
            actor = world.get_actor(actor_id)
            if actor is not None and actor.is_alive:
                return actor
        except Exception:
            pass
        return None


    _AGENT_COLORS = [
        (41, 128, 185),
        (231, 76, 60),
        (39, 174, 96),
        (243, 156, 18),
        (142, 68, 173),
        (22, 160, 133),
        (211, 84, 0),
        (44, 62, 80),
    ]

    def render_selection(self, selection_result, wote_results, grid_conf,
                         rsu_location, step: int = 0,
                         ego_vehicle_ids=None,
                         vehicle_targets=None) -> str:
        """Render Phase 4 selection: all candidates (faint) + selected (solid).

        For each agent:
        - All M candidates: faint gray polylines
        - Selected trajectory: solid colored polyline (per-agent color)
        - Vehicle bbox: filled with agent color
        - Target point: star marker (if available)
        - Label: agent ID, priority rank, cost

        Args:
            selection_result: CsSelectionResult from Phase 4.
            wote_results: dict {agent_id: (candidates, scores, vehicle_state)}.
            grid_conf: motion grid config dict.
            rsu_location: carla.Location of RSU.
            step: simulation step number.
            ego_vehicle_ids: set of vehicle IDs to highlight.
            vehicle_targets: dict {agent_id: (x, y)} world-frame target points.

        Returns:
            path: saved image path.
        """
        import math

        if selection_result is None or rsu_location is None:
            return ""

        canvas = self._get_road_bg()
        ego_pose = (rsu_location.x, rsu_location.y, 0.0)

        if ego_vehicle_ids is None:
            ego_vehicle_ids = set()

        canvas = self._draw_gt_vehicles(
            canvas, rsu_location, grid_conf, ego_vehicle_ids)

        priority_rank = {}
        for rank, aid in enumerate(selection_result.priority_order):
            priority_rank[aid] = rank

        bg_color = (200, 200, 200)
        bg_alpha = 0.15

        for color_idx, agent_id in enumerate(selection_result.priority_order):
            if wote_results is None or agent_id not in wote_results:
                continue

            candidates, scores, vstate = wote_results[agent_id][:3]
            wx, wy = vstate.position_world
            heading = vstate.heading_rad
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            M = candidates.shape[0]

            chosen_idx = selection_result.assignments.get(agent_id, -1)
            agent_color = self._AGENT_COLORS[color_idx % len(self._AGENT_COLORS)]

            overlay = canvas.copy()
            for m in range(M):
                if m == chosen_idx:
                    continue
                traj = candidates[m]
                bev_pts = []
                for t in range(traj.shape[0]):
                    x_local, y_local = traj[t, 0], traj[t, 1]
                    x_world = wx + x_local * cos_h - y_local * sin_h
                    y_world = wy + x_local * sin_h + y_local * cos_h
                    px, py = world_to_bev(x_world, y_world, grid_conf, ego_pose)
                    bev_pts.append((int(round(px)), int(round(py))))
                if len(bev_pts) >= 2:
                    pts_arr = np.array(bev_pts, dtype=np.int32)
                    cv2.polylines(overlay, [pts_arr], isClosed=False,
                                  color=bg_color, thickness=1)

            cv2.addWeighted(overlay, bg_alpha, canvas, 1.0 - bg_alpha,
                            0, canvas)

            if chosen_idx >= 0:
                if chosen_idx < M:
                    traj_sel = candidates[chosen_idx]
                else:
                    traj_sel = selection_result.trajectories_local.get(agent_id)
                    if traj_sel is None:
                        traj_sel = np.zeros((candidates.shape[1], 3))
                bev_pts_sel = []
                for t in range(traj_sel.shape[0]):
                    x_local, y_local = traj_sel[t, 0], traj_sel[t, 1]
                    x_world = wx + x_local * cos_h - y_local * sin_h
                    y_world = wy + x_local * sin_h + y_local * cos_h
                    px, py = world_to_bev(x_world, y_world, grid_conf, ego_pose)
                    bev_pts_sel.append((int(round(px)), int(round(py))))
                if len(bev_pts_sel) >= 2:
                    pts_arr = np.array(bev_pts_sel, dtype=np.int32)
                    cv2.polylines(canvas, [pts_arr], isClosed=False,
                                  color=agent_color, thickness=2)
                    for pt in bev_pts_sel:
                        cv2.circle(canvas, pt, 2, agent_color, -1)
                        cv2.circle(canvas, pt, 2, (0, 0, 0), 1)

            import carla as carla_mod
            carla_veh = self._find_carla_vehicle_for_vis(agent_id)
            if carla_veh is not None:
                bb = carla_veh.bounding_box.extent
                corners_local = [
                    carla_mod.Location(x=-bb.x, y=-bb.y),
                    carla_mod.Location(x=bb.x, y=-bb.y),
                    carla_mod.Location(x=bb.x, y=bb.y),
                    carla_mod.Location(x=-bb.x, y=bb.y),
                ]
                carla_veh.get_transform().transform(corners_local)
                bev_corners = []
                for c in corners_local:
                    cpx, cpy = world_to_bev(c.x, c.y, grid_conf, ego_pose)
                    bev_corners.append([int(round(cpx)), int(round(cpy))])
                bev_corners = np.array(bev_corners, dtype=np.int32)
            else:
                apx, apy = world_to_bev(wx, wy, grid_conf, ego_pose)
                h = 3
                bev_corners = np.array([
                    [int(apx) - h, int(apy) - h],
                    [int(apx) + h, int(apy) - h],
                    [int(apx) + h, int(apy) + h],
                    [int(apx) - h, int(apy) + h],
                ], dtype=np.int32)
            cv2.fillPoly(canvas, [bev_corners], agent_color)
            cv2.drawContours(canvas, [bev_corners], 0, (0, 0, 0), 1)


            rank = priority_rank.get(agent_id, -1)
            cost = selection_result.costs.get(agent_id, 0.0)
            label = "id{} P{} c={:.2f}".format(agent_id, rank + 1, cost)
            apx, apy = world_to_bev(wx, wy, grid_conf, ego_pose)
            cv2.putText(canvas, label,
                        (int(apx) + 5, int(apy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255), 1)

        canvas = _flip_rotate_image(canvas)

        path = os.path.join(
            self._save_dir, 'selection_{:06d}.png'.format(step))
        cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print("[CoSafe Vis] Saved selection: {}".format(path))

        self._render_ref_paths(
            selection_result, wote_results, grid_conf, rsu_location,
            step, ego_vehicle_ids)

        self.render_pool(
            selection_result, wote_results, grid_conf, rsu_location,
            step, ego_vehicle_ids)

        return path

    def render_pool(self, selection_result, wote_results, grid_conf,
                    rsu_location, step: int = 0,
                    ego_vehicle_ids=None) -> str:
        """Render pool candidates per agent: GO (colored) vs non-GO (red dashed).

        One sub-image per agent so trajectories don't overlap across agents.
        Also renders a combined image for quick overview.

        Color coding:
        - GO candidates: agent color, alpha proportional to rank (best=opaque)
        - Non-GO (stop): red dashed
        - Selected: thick white outline
        - Reference path: cyan dashed
        """
        import math

        pool_info = getattr(selection_result, 'pool_info', None)
        if pool_info is None or rsu_location is None:
            return ""

        if ego_vehicle_ids is None:
            ego_vehicle_ids = set()

        ego_pose = (rsu_location.x, rsu_location.y, 0.0)
        agents = selection_result.priority_order

        per_agent_canvases = {}

        for color_idx, agent_id in enumerate(agents):
            if wote_results is None or agent_id not in wote_results or agent_id not in pool_info:
                continue

            vstate = wote_results[agent_id][2]
            info = pool_info[agent_id]
            go_set = set(info['go'])
            non_go_set = set(info['non_go'])
            pool = info['pool']
            stop_idx = info['stop_idx']
            costs = info.get('costs', {})
            local_cands = info['local_candidates']
            chosen_idx = selection_result.assignments.get(agent_id, -1)

            wx, wy = vstate.position_world
            heading = vstate.heading_rad
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)

            agent_color = self._AGENT_COLORS[color_idx % len(self._AGENT_COLORS)]

            canvas = self._get_road_bg()
            canvas = self._draw_gt_vehicles(
                canvas, rsu_location, grid_conf, ego_vehicle_ids)

            go_by_cost = sorted(
                [(m, costs.get(m, 0.0)) for m in pool if m in go_set],
                key=lambda x: x[1])

            n_go = len(go_by_cost)
            for rank, (m, cost) in enumerate(go_by_cost):
                alpha = 0.8 - 0.6 * (rank / max(n_go - 1, 1)) if n_go > 1 else 0.8
                traj = self._get_pool_traj_world(
                    m, local_cands, wx, wy, cos_h, sin_h)
                bev_pts = self._traj_to_bev(traj, grid_conf, ego_pose)
                if len(bev_pts) >= 2:
                    overlay = canvas.copy()
                    pts_arr = np.array(bev_pts, dtype=np.int32)
                    cv2.polylines(overlay, [pts_arr], isClosed=False,
                                  color=agent_color, thickness=2)
                    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha,
                                    0, canvas)
                    cv2.putText(canvas, "{:.2f}".format(cost),
                                (bev_pts[-1][0] + 2, bev_pts[-1][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                agent_color, 1)

            for m in pool:
                if m in go_set:
                    continue
                traj = self._get_pool_traj_world(
                    m, local_cands, wx, wy, cos_h, sin_h)
                bev_pts = self._traj_to_bev(traj, grid_conf, ego_pose)
                if len(bev_pts) >= 2:
                    for i in range(0, len(bev_pts) - 1, 2):
                        cv2.line(canvas, bev_pts[i],
                                 bev_pts[min(i + 1, len(bev_pts) - 1)],
                                 (60, 60, 230), 1)
                    label = "stop" if m == stop_idx else "ngo"
                    cv2.putText(canvas, label,
                                (bev_pts[-1][0] + 2, bev_pts[-1][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                (60, 60, 230), 1)

            if chosen_idx >= 0:
                traj = self._get_pool_traj_world(
                    chosen_idx, local_cands, wx, wy, cos_h, sin_h)
                bev_pts = self._traj_to_bev(traj, grid_conf, ego_pose)
                if len(bev_pts) >= 2:
                    pts_arr = np.array(bev_pts, dtype=np.int32)
                    cv2.polylines(canvas, [pts_arr], isClosed=False,
                                  color=(255, 255, 255), thickness=3)
                    cv2.polylines(canvas, [pts_arr], isClosed=False,
                                  color=agent_color, thickness=1)

            ref_paths = getattr(selection_result, 'reference_paths', None)
            if ref_paths and agent_id in ref_paths:
                ref_path = ref_paths[agent_id]
                if ref_path and len(ref_path) >= 2:
                    ref_bev = []
                    for rx, ry in ref_path:
                        rpx, rpy = world_to_bev(rx, ry, grid_conf, ego_pose)
                        ref_bev.append((int(round(rpx)), int(round(rpy))))
                    for i in range(0, len(ref_bev) - 1, 2):
                        cv2.line(canvas, ref_bev[i],
                                 ref_bev[min(i + 1, len(ref_bev) - 1)],
                                 (255, 255, 0), 1)

            apx, apy = world_to_bev(wx, wy, grid_conf, ego_pose)
            cv2.circle(canvas, (int(apx), int(apy)), 4, agent_color, -1)
            cv2.circle(canvas, (int(apx), int(apy)), 4, (0, 0, 0), 1)
            label = "id{} go={} ngo={} sel={}".format(
                agent_id, len(go_set), len(non_go_set), chosen_idx)
            cv2.putText(canvas, label,
                        (int(apx) + 5, int(apy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255), 1)

            canvas = _flip_rotate_image(canvas)
            per_agent_canvases[agent_id] = canvas

            path = os.path.join(
                self._save_dir,
                'pool_{:06d}_agent{}.png'.format(step, agent_id))
            cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        if per_agent_canvases:
            imgs = [per_agent_canvases[a] for a in agents
                    if a in per_agent_canvases]
            if imgs:
                n = len(imgs)
                cols = min(2, n)
                rows = (n + cols - 1) // cols
                h, w = imgs[0].shape[:2]
                combined = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
                for i, img in enumerate(imgs):
                    r, c = divmod(i, cols)
                    combined[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
                cpath = os.path.join(
                    self._save_dir, 'pool_{:06d}.png'.format(step))
                cv2.imwrite(cpath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                print("[CoSafe Vis] Saved pool: {}".format(cpath))
                return cpath

        return ""

    def _get_pool_traj_world(self, m, local_candidates,
                              wx, wy, cos_h, sin_h):
        """Get world-frame trajectory points for a pool candidate.

        Args:
            m: candidate index.
            local_candidates: (M+1, T, 3) array including stop trajectory.

        Returns list of (x_world, y_world).
        """
        traj_local = local_candidates[m]
        pts = []
        for t in range(traj_local.shape[0]):
            xl, yl = traj_local[t, 0], traj_local[t, 1]
            xw = wx + xl * cos_h - yl * sin_h
            yw = wy + xl * sin_h + yl * cos_h
            pts.append((xw, yw))
        return pts

    def _traj_to_bev(self, world_pts, grid_conf, ego_pose):
        """Convert list of (x_world, y_world) to BEV pixel coords."""
        bev_pts = []
        for xw, yw in world_pts:
            px, py = world_to_bev(xw, yw, grid_conf, ego_pose)
            bev_pts.append((int(round(px)), int(round(py))))
        return bev_pts

    def _render_ref_paths(self, selection_result, wote_results, grid_conf,
                          rsu_location, step, ego_vehicle_ids):
        """Render a dedicated reference-path-only image for debugging."""
        import math

        ref_paths = getattr(selection_result, 'reference_paths', None)
        if not ref_paths or rsu_location is None:
            return

        canvas = self._get_road_bg()
        ego_pose = (rsu_location.x, rsu_location.y, 0.0)
        if ego_vehicle_ids is None:
            ego_vehicle_ids = set()

        canvas = self._draw_gt_vehicles(
            canvas, rsu_location, grid_conf, ego_vehicle_ids)

        for color_idx, agent_id in enumerate(selection_result.priority_order):
            agent_color = self._AGENT_COLORS[color_idx % len(self._AGENT_COLORS)]
            ref_path = ref_paths.get(agent_id)
            if not ref_path or len(ref_path) < 2:
                continue

            bev_pts = []
            for rx, ry in ref_path:
                rpx, rpy = world_to_bev(rx, ry, grid_conf, ego_pose)
                bev_pts.append((int(round(rpx)), int(round(rpy))))

            pts_arr = np.array(bev_pts, dtype=np.int32)
            cv2.polylines(canvas, [pts_arr], isClosed=False,
                          color=agent_color, thickness=2)
            for i, pt in enumerate(bev_pts):
                cv2.circle(canvas, pt, 2, agent_color, -1)
                cv2.circle(canvas, pt, 2, (0, 0, 0), 1)

            cv2.putText(canvas, "S", bev_pts[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(canvas, "E", bev_pts[-1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            if wote_results is not None and agent_id in wote_results:
                vstate = wote_results[agent_id][2]
                wx, wy = vstate.position_world
                apx, apy = world_to_bev(wx, wy, grid_conf, ego_pose)
                cv2.putText(canvas,
                            "id{} pts={}".format(agent_id, len(ref_path)),
                            (int(apx) + 5, int(apy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255), 1)

        canvas = _flip_rotate_image(canvas)
        path = os.path.join(
            self._save_dir, 'ref_{:06d}.png'.format(step))
        cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def save_risk_report(self, risk_assessment, step: int = 0) -> str:
        """Save JSON report of risks and flagged agents."""
        if risk_assessment is None or not risk_assessment.risks:
            return ""

        report = {
            'step': risk_assessment.step,
            'perception_mode': risk_assessment.perception_mode,
            'num_scenes': risk_assessment.num_scenes,
            'flagged_agents': sorted(risk_assessment.flagged_agents),
            'risks': [
                {
                    'risk_type': r.risk_type,
                    'severity': r.severity,
                    'agent_i': r.agent_i,
                    'agent_j': r.agent_j,
                    'timestep': r.timestep,
                    'time_seconds': r.time_seconds,
                    'details': r.details,
                }
                for r in risk_assessment.risks
            ],
        }

        path = os.path.join(
            self._save_dir, 'risk_report_{:06d}.json'.format(step))
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return path
