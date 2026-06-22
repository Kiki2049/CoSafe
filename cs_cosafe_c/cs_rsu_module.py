"""CoSafe RSU module: the central safety middleware.

Phase 0: pass-through only — returns ADS controls unchanged.
Phase 1: cooperative perception via BEVerse V2X (BEV fusion + K=6 prediction).
Future phases will implement the full MAPE-K pipeline here.
"""

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import carla

from cs_cosafe_c.cs_config import CsCoSafeConfig


@dataclass
class CsVehicleState:
    """State of a single detected vehicle in BEV and world coordinates."""

    instance_id: int
    position_bev: Tuple[float, float]
    position_world: Tuple[float, float]
    trajectory_bev: List[Tuple[float, float]]
    trajectory_world: List[Tuple[float, float]]
    is_ads_vehicle: bool
    ads_vehicle_id: int = -1
    heading_rad: float = 0.0
    speed: float = 0.0


@dataclass
class CsPerceptionOutput:
    """Structured perception output for downstream planning/safety modules.

    Two modes:
    - 'gt':   ground-truth from CARLA actors. scenes has 1 element.
    - 'pred': model predictions (K=6 VAE samples). scenes has K elements.

    Usage:
        output.scenes[-1]  # mean/best scene (last in VAE order)
        for scene in output.scenes:  # iterate all K possibilities
            for v in scene: ...
    """

    scenes: List[List[CsVehicleState]]
    mode: str
    step: int = 0
    ego_vehicle_ids: List[int] = field(default_factory=list)


class CoSafeRSU:
    """Road-Side Unit safety module.

    Phase 0: no-op pass-through.
    Phase 1: runs BEVerse V2X perception at configured interval,
             instance tracking, and optional visualization.
             Still returns ADS actions unchanged (no override yet).
    """

    def __init__(self, config: CsCoSafeConfig, ego_vehicles_num: int):
        self._config = config
        self._ego_vehicles_num = ego_vehicles_num

        self._perception = None
        self._tracker = None
        self._visualizer = None

        self._risk_oracle = None
        self._last_risk_assessment = None

        self._wote_generator = None
        self._last_wote_results = None

        self._candidate_selector = None
        self._last_selection = None
        self._last_selection_step = -1

        self._mpc_controller = None

        self._ego_vehicles = None
        self._rsu_location = None
        self._last_predictions = None
        self._step_count = 0
        self._carla_map = None

        self._vehicle_commands = {}

        self._vehicle_targets = {}

        self._vehicle_reference_paths = {}

        self._warmup_ticks = 0

    @staticmethod
    def _format_command_name(command):
        cmd_names = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            5: 'CHANGELEFT',
            6: 'CHANGERIGHT',
        }
        return cmd_names.get(command, 'None')

    def _log_selection_summary(self, selection_result, wote_results):
        """Print compact per-agent selection diagnostics."""
        if selection_result is None or not wote_results:
            return

        for agent_id in selection_result.priority_order:
            if agent_id not in selection_result.assignments:
                continue
            if agent_id not in wote_results:
                continue

            chosen_idx = selection_result.assignments[agent_id]
            total_cost = selection_result.costs.get(agent_id, float('nan'))
            _cands, scores, _vstate, command = wote_results[agent_id]
            score = float('nan')
            if scores is not None and 0 <= chosen_idx < len(scores):
                score = float(scores[chosen_idx])

            print(
                "[CoSafe][Selection] step={} agent={} priority={} "
                "cmd={}({}) selected_idx={} total_cost={:.3f} wote_score={:.4f}".format(
                    self._step_count,
                    agent_id,
                    selection_result.priority_order.index(agent_id) + 1,
                    command,
                    self._format_command_name(command),
                    chosen_idx,
                    float(total_cost),
                    score,
                )
            )

    def _log_mpc_status(self, agent_id, status):
        """Print compact MPC execution diagnostics for one agent."""
        if not status:
            print(
                "[CoSafe][MPC] step={} agent={} status=unknown".format(
                    self._step_count, agent_id)
            )
            return

        msg = "[CoSafe][MPC] step={} agent={} status={}".format(
            self._step_count, agent_id, status.get('status'))

        if 'progress' in status:
            msg += " progress={:.3f}".format(float(status['progress']))
        if 'return_status' in status:
            msg += " return_status={}".format(status['return_status'])
        if 'error' in status:
            msg += " error={}".format(status['error'])
        if status.get('status') == 'ok':
            msg += (
                " acc={:.3f} steer_rad={:.3f} throttle={:.3f} "
                "brake={:.3f} steer={:.3f}".format(
                    float(status.get('acc', 0.0)),
                    float(status.get('steer_rad', 0.0)),
                    float(status.get('throttle', 0.0)),
                    float(status.get('brake', 0.0)),
                    float(status.get('steer', 0.0)),
                )
            )

        print(msg)

    def setup(self, ego_vehicles=None, rsu_location=None, save_root=None):
        """Initialize RSU sub-modules.

        Args:
            ego_vehicles: list of carla.Actor (needed for Phase 1+)
            rsu_location: carla.Location of the RSU
            save_root: root save directory (typically $SAVE_PATH).
                       Visualization saves to {save_root}/cosafe/cosafe_vis/.
                       Follows the same convention as cs_scenario_manager.py
                       which saves debug images to {save_root}/cosafe/rsu/ etc.
        """
        self._ego_vehicles = ego_vehicles
        self._rsu_location = rsu_location

        if self._config.perception.enable:
            from cs_cosafe_c.perception.cs_beverse_inference import CsBeverseInference
            from cs_cosafe_c.perception.cs_instance_tracker import CsInstanceTracker
            from cs_cosafe_c.visualization.cs_visualizer import CsVisualizer

            self._perception = CsBeverseInference(
                self._config.perception, self._config.camera,
                self._config.bev)
            self._perception.load_model()

            self._tracker = CsInstanceTracker(
                self._config.bev,
                match_threshold=10.0,
            )

            if self._config.perception.save_visualization:
                if save_root:
                    vis_dir = os.path.join(
                        str(save_root), 'cosafe', 'cosafe_vis')
                else:
                    vis_dir = self._config.perception.visualization_dir
                os.makedirs(vis_dir, exist_ok=True)
                grid = self._config.bev.motion_grid
                h = int((grid['ybound'][1] - grid['ybound'][0]) / grid['ybound'][2])
                w = int((grid['xbound'][1] - grid['xbound'][0]) / grid['xbound'][2])
                self._visualizer = CsVisualizer(
                    save_dir=vis_dir, canvas_size=(h, w))

                if rsu_location is not None:
                    self._visualizer.setup_road_bg(rsu_location, grid)

                print("[CoSafe] Visualization dir: {}".format(vis_dir))

            print("[CoSafe] RSU module initialized (Phase 1: cooperative perception)")
        else:
            print("[CoSafe] RSU module initialized (Phase 0: pass-through)")

        if self._config.risk.enable:
            from cs_cosafe_c.safety.cs_risk_oracle import CsRiskOracle

            self._risk_oracle = CsRiskOracle(self._config.risk)
            self._risk_oracle.setup()
            print("[CoSafe] Risk assessment initialized (Phase 2: safety check)")

        if self._config.wote.enable:
            from cs_cosafe_c.planners.cs_wote_candidates import CsWoTECandidateGenerator

            self._wote_generator = CsWoTECandidateGenerator(
                self._config.wote, device=self._config.perception.device)
            print("[CoSafe] WoTE candidate generator initialized (Phase 3)")

        if self._config.selector.enable:
            from cs_cosafe_c.planners.cs_candidate_selector import CsCandidateSelector
            self._candidate_selector = CsCandidateSelector(
                self._config.selector, self._config.risk)
            print("[CoSafe] Candidate selector initialized (Phase 4)")

        if self._config.mpc.enable:
            from cs_cosafe_c.controllers.cs_mpc_controller import CsMpcController
            self._mpc_controller = CsMpcController(
                self._config.mpc, self._config.bicycle)
            print("[CoSafe] MPC controller initialized (Phase 5)")

    def _get_ego_vehicle_ids(self):
        """Return set of ego vehicle CARLA actor IDs for visualization highlighting."""
        ids = set()
        if self._ego_vehicles:
            for v in self._ego_vehicles:
                if v is not None and v.is_alive:
                    ids.add(v.id)
        return ids

    def step(self, sensor_data, ads_actions, ads_trajectories=None):
        """Process one timestep of the MAPE-K pipeline.

        Phase 0: returns ads_actions unchanged.
        Phase 1: runs perception, stores predictions, still returns
                 ads_actions unchanged (no override yet).

        Args:
            sensor_data: dict from CsSensorManager.tick_all()
            ads_actions: List[carla.VehicleControl] from ADS agent
            ads_trajectories: None (ADS agents don't expose trajectories)

        Returns:
            List[carla.VehicleControl]: final actions (same as ads_actions in Phase 1)
        """
        self._step_count += 1
        import time as _time
        _t_step_start = _time.perf_counter()
        _phase_times = {}

        if self._config.perception.enable and self._perception is not None:
            predictions = self._perception.run_inference(
                sensor_data, self._ego_vehicles, self._rsu_location)

        _t0 = _time.perf_counter()
        if self._config.perception.enable and self._perception is not None:
            if predictions is not None:
                self._last_predictions = predictions

                all_vehicles = self._get_all_vehicles_in_range()
                if self._tracker is not None:
                    self._tracker.update_gt_positions(
                        all_vehicles, self._rsu_location)

                instance_seg = predictions.get('instance_seg')
                matched_centers = predictions.get('matched_centers')
                if instance_seg is not None and self._tracker is not None:
                    matching = self._tracker.update(
                        instance_seg, matched_centers,
                        all_vehicles, self._rsu_location)
                    predictions['matching'] = matching
        _phase_times['P1b_tracking'] = _time.perf_counter() - _t0

        if self._config.perception.enable and self._perception is not None:
            is_inference_tick = (
                self._step_count % self._config.perception.inference_interval == 0)

            if self._visualizer is not None and is_inference_tick:
                grid_conf = self._config.bev.motion_grid

                is_warmed_up = self._step_count > self._warmup_ticks
                motion_preds_raw = None
                rsu_only_preds_raw = None
                if is_warmed_up and predictions is not None:
                    motion_preds_raw = predictions.get('motion_predictions')
                    rsu_only_preds_raw = predictions.get(
                        'rsu_only_motion_predictions')

                self._visualizer.render(
                    motion_preds_raw,
                    rsu_location=self._rsu_location,
                    grid_conf=grid_conf,
                    ego_vehicle_ids=self._get_ego_vehicle_ids(),
                    step=self._step_count,
                    rsu_only_motion_predictions=rsu_only_preds_raw,
                )

        _t0 = _time.perf_counter()
        if (self._config.risk.enable
                and self._risk_oracle is not None
                and self._step_count % self._config.perception.inference_interval == 0):
            perception_output = self.get_perception_output(
                self._config.perception_mode)
            if perception_output is not None:
                self._last_risk_assessment = self._risk_oracle.assess(
                    perception_output)
        _phase_times['P2_risk'] = _time.perf_counter() - _t0

        if (self._config.risk.enable
                and self._risk_oracle is not None
                and self._step_count % self._config.perception.inference_interval == 0):
            if self._visualizer is not None and self._last_risk_assessment is not None:
                perception_output = self.get_perception_output(
                    self._config.perception_mode)
                if perception_output is not None:
                    grid_conf = self._config.bev.motion_grid
                    self._visualizer.render_risks(
                        perception_output,
                        self._last_risk_assessment,
                        grid_conf,
                        self._rsu_location,
                        step=self._step_count,
                        ego_vehicle_ids=self._get_ego_vehicle_ids(),
                    )
                    self._visualizer.save_risk_report(
                        self._last_risk_assessment,
                        step=self._step_count,
                    )

        _t0 = _time.perf_counter()
        if (self._wote_generator is not None
                and self._last_predictions is not None
                and self._step_count % self._config.perception.inference_interval == 0):
            flagged = self.get_flagged_agents()
            ego_ids = self._get_ego_vehicle_ids()
            flagged_ads = flagged & ego_ids
            if not flagged_ads:
                self._last_wote_results = None
                self._last_selection = None
            else:
                from cs_cosafe_c.planners.cs_wote_candidates import build_ego_status

                per_agent_bev = self._perception.get_per_agent_bev()
                wote_results = {}
                for agent_id in flagged_ads:
                    vehicle_state = self._find_vehicle_state(agent_id)
                    if vehicle_state is None:
                        continue
                    carla_veh = self._find_carla_vehicle(agent_id)
                    command = self._get_vehicle_command(agent_id)
                    print("[CoSafe] Agent {} command: {} ({})".format(
                        agent_id, command, self._format_command_name(command)))
                    ego_status = build_ego_status(
                        vehicle_state, carla_vehicle=carla_veh,
                        command=command)
                    if agent_id in per_agent_bev:
                        candidates, scores = self._wote_generator.generate(
                            per_agent_bev[agent_id], ego_status)
                    else:
                        candidates, scores = self._wote_generator.generate_fallback()
                    wote_results[agent_id] = (candidates, scores, vehicle_state, command)
                self._last_wote_results = wote_results


        _phase_times['P3_wote'] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        if (self._candidate_selector is not None
                and self._last_wote_results
                and self._step_count % self._config.perception.inference_interval == 0):
            perception_output = self.get_perception_output(
                self._config.perception_mode)
            if perception_output is not None:
                flagged_ads = self.get_flagged_agents() & self._get_ego_vehicle_ids()
                if flagged_ads:
                    self._last_selection = self._candidate_selector.solve(
                        self._last_wote_results,
                        perception_output,
                        self._last_risk_assessment,
                        flagged_ads,
                        vehicle_targets=self._vehicle_targets,
                        vehicle_reference_paths=self._vehicle_reference_paths,
                    )
                    if self._last_selection is not None:
                        self._last_selection_step = self._step_count
                    self._log_selection_summary(
                        self._last_selection, self._last_wote_results)
                    if self._last_selection is not None and self._mpc_controller is not None:
                        for agent_id in flagged_ads:
                            if agent_id not in self._last_selection.assignments:
                                continue
                            ref_traj = self._last_selection.trajectories_world.get(agent_id)
                            carla_veh = self._find_carla_vehicle(agent_id)
                            if ref_traj is not None and carla_veh is not None:
                                self._mpc_controller.update_reference(
                                    agent_id, ref_traj, carla_veh)

        if self._candidate_selector is not None and hasattr(self._candidate_selector, '_last_core_time'):
            _phase_times['P4_selection'] = self._candidate_selector._last_core_time
        else:
            _phase_times['P4_selection'] = _time.perf_counter() - _t0

        if (self._candidate_selector is not None
                and self._step_count % self._config.perception.inference_interval == 0):
            if self._visualizer is not None and self._last_selection is not None:
                self._visualizer.render_selection(
                    self._last_selection,
                    self._last_wote_results,
                    self._config.bev.motion_grid,
                    self._rsu_location,
                    step=self._step_count,
                    ego_vehicle_ids=self._get_ego_vehicle_ids(),
                    vehicle_targets=self._vehicle_targets,
                )

        _t0 = _time.perf_counter()
        if self._mpc_controller is not None:
            flagged_ads = self.get_flagged_agents() & self._get_ego_vehicle_ids()
            stop_agents = (self._last_selection.stop_agents
                           if self._last_selection is not None else set())

            for agent_id in flagged_ads:
                carla_veh = self._find_carla_vehicle(agent_id)
                if carla_veh is None:
                    continue

                if agent_id in stop_agents:
                    brake_control = carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=1.0)
                    for i, ev in enumerate(self._ego_vehicles or []):
                        if ev is not None and ev.id == agent_id:
                            ads_actions[i] = brake_control
                            break
                    continue

                mpc_control = self._mpc_controller.compute_control(
                    carla_veh, agent_id)
                self._log_mpc_status(
                    agent_id,
                    self._mpc_controller.get_last_status(agent_id),
                )

                if mpc_control is not None:
                    for i, ev in enumerate(self._ego_vehicles or []):
                        if ev is not None and ev.id == agent_id:
                            ads_actions[i] = mpc_control
                            break

        _phase_times['P5_mpc'] = _time.perf_counter() - _t0

        if self._step_count % self._config.perception.inference_interval == 0:
            parts = " ".join("{}={:.1f}ms".format(k, v * 1000)
                             for k, v in _phase_times.items())
            _t_sum = sum(_phase_times.values())
            print("[CoSafe][Timing] step={} sum={:.1f}ms {}".format(
                self._step_count, _t_sum * 1000, parts))

        return ads_actions

    def get_predictions(self):
        """Return the latest raw perception predictions (for external consumers)."""
        return self._last_predictions

    def get_risk_assessment(self) -> Optional['CsRiskAssessment']:
        """Return the latest risk assessment (Phase 2).

        Returns:
            CsRiskAssessment with fields:
                risks: List[CsRiskEvent]  — individual risk events
                    Each CsRiskEvent has:
                        risk_type: 'collision' | 'red_light' | 'deadlock' | 'single_stuck'
                        severity: float (0-1, collision probability; 1.0 for others)
                        agent_i: int (vehicle ID)
                        agent_j: int (second vehicle ID, -1 for single-agent risks)
                        timestep: int (future frame index, -1 for deadlock/single_stuck)
                        time_seconds: float (time to risk event)
                        details: dict (risk-type-specific info)
                step: int — assessment step number
                perception_mode: 'gt' | 'pred'
                num_scenes: int — K (1 for GT, 6 for pred)
                flagged_agents: Set[int] — union of all risky agents + w_hold cooldown
            None if no assessment has been run yet.
        """
        return self._last_risk_assessment

    def get_flagged_agents(self) -> set:
        """Return the set of agent IDs that Phase 3 should jointly optimize.

        This is the primary Phase 2 → Phase 3 interface.

        Agent IDs:
            - GT mode: CARLA actor ID (veh.id)
            - Pred mode: CARLA actor ID (matched) or BEV instance_id (unmatched)

        Includes:
            - Agents with active risks this frame (collision / red_light /
              deadlock / single_stuck)
            - Agents still within w_hold=5s cooldown after a risk disappears,
              preventing control oscillation between ADS and safety override

        Returns:
            Set[int] of agent IDs, or empty set if no risks detected.
        """
        if self._last_risk_assessment is not None:
            return self._last_risk_assessment.flagged_agents
        return set()

    def get_wote_candidates(self):
        """Return latest WoTE candidate results for downstream CBS.

        Returns:
            dict: {agent_id: (candidates, scores, vehicle_state, command)} or None.
                candidates: (M, T, 3) numpy, ego-local coords.
                scores: (M,) numpy.
                vehicle_state: CsVehicleState.
                command: int (CARLA RoadOption value 1-6) or None.
        """
        return self._last_wote_results

    def get_selection(self):
        """Return latest trajectory selection (Phase 4) for downstream MPC.

        Returns:
            CsSelectionResult with fields:
                assignments: Dict[int, int]  — agent_id -> candidate index
                trajectories_local: Dict[int, np.ndarray]  — (T, 3) ego-local
                trajectories_world: Dict[int, np.ndarray]  — (T, 3) world frame
                costs: Dict[int, float]  — total cost per agent
                priority_order: List[int]  — agents sorted by urgency
            None if no selection has been made.
        """
        return self._last_selection

    def _find_vehicle_state(self, agent_id) -> Optional[CsVehicleState]:
        """Look up CsVehicleState for an ADS vehicle by CARLA actor ID.

        Uses only the mean scene (scenes[-1]) to avoid ambiguity across
        K VAE samples.  Falls back to direct CARLA actor query.

        Works for both GT and pred modes:
        - GT mode: instance_id == veh.id (always CARLA actor ID)
        - Pred mode: after flagged & ego_ids intersection, agent_id is
          always a matched CARLA actor ID (unmatched BEV-only instances
          are never ADS vehicles)
        """
        perception_output = self.get_perception_output(
            self._config.perception_mode)
        if perception_output is not None and perception_output.scenes:
            for v in perception_output.scenes[-1]:
                if v.instance_id == agent_id:
                    return v

        from cs_cosafe_c.perception.cs_bev_transform import world_to_bev
        if self._ego_vehicles:
            for ev in self._ego_vehicles:
                if ev is not None and ev.is_alive and ev.id == agent_id:
                    vel = ev.get_velocity()
                    speed = (vel.x ** 2 + vel.y ** 2) ** 0.5
                    yaw = math.radians(ev.get_transform().rotation.yaw)
                    loc = ev.get_location()
                    grid_conf = self._config.bev.motion_grid
                    ego_pose = (self._rsu_location.x,
                                self._rsu_location.y, 0.0)
                    px, py = world_to_bev(loc.x, loc.y, grid_conf, ego_pose)
                    return CsVehicleState(
                        instance_id=agent_id,
                        position_bev=(px, py),
                        position_world=(loc.x, loc.y),
                        trajectory_bev=[],
                        trajectory_world=[],
                        is_ads_vehicle=True,
                        ads_vehicle_id=agent_id,
                        heading_rad=yaw,
                        speed=speed,
                    )
        return None

    def _find_carla_vehicle(self, agent_id):
        """Find the carla.Vehicle actor for a given agent ID, or None."""
        for veh in self._get_all_vehicles_in_range():
            if veh.id == agent_id:
                return veh
        return None

    def set_vehicle_commands(self, commands):
        """Receive driving commands uploaded by ADS vehicles each tick.

        In the V2X cooperative architecture, vehicles upload their routing
        commands to the RSU.  Called from CsScenarioManager after the ADS
        agent produces its tick data.

        Args:
            commands: dict {carla_actor_id: int} where int is CARLA
                RoadOption value (1=LEFT, 2=RIGHT, 3=STRAIGHT, 4=FOLLOW,
                5=CHANGE_LEFT, 6=CHANGE_RIGHT).
        """
        self._vehicle_commands = commands

    def _get_vehicle_command(self, agent_id):
        """Get the latest driving command for a vehicle, or None.

        Returns:
            int (CARLA RoadOption value) or None if unknown.
        """
        return self._vehicle_commands.get(agent_id)

    def set_vehicle_targets(self, targets):
        """Receive target points uploaded by ADS vehicles each tick.

        Args:
            targets: dict {carla_actor_id: (x, y)} world coordinates.
        """
        self._vehicle_targets = targets

    def _get_vehicle_target(self, agent_id):
        """Get the latest target point for a vehicle, or None.

        Returns:
            (x, y) world coordinates or None.
        """
        return self._vehicle_targets.get(agent_id)

    def _should_reselect(self, flagged_ads):
        """Decide whether Phase 4 should reselect trajectories.

        Reselect when:
        1. No prior selection exists.
        2. MPC progress >= 20 waypoints (1.0s executed out of 2.0s trajectory).
        3. New agents joined the flagged set since last selection.
        4. Risk Oracle reports collision involving currently-selected agents.
        """
        if self._last_selection is None or self._last_selection_step < 0:
            return True

        if self._mpc_controller is not None:
            for agent_id in flagged_ads:
                if agent_id not in self._last_selection.assignments:
                    continue
                ticks = self._mpc_controller._ticks_since_ref.get(agent_id, 0)
                progress = ticks * self._config.mpc.dt / self._config.wote.dt_plan
                if progress >= 20.0:
                    return True

        prev_agents = set(self._last_selection.assignments.keys())
        if not flagged_ads.issubset(prev_agents):
            return True

        if self._last_risk_assessment is not None:
            selected = set(self._last_selection.assignments.keys())
            for r in self._last_risk_assessment.risks:
                if (r.risk_type == 'collision'
                        and r.agent_i in selected
                        and r.agent_j in selected):
                    return True

        return False

    def set_vehicle_reference_paths(self, ref_paths):
        """Receive reference paths uploaded by ADS vehicles each tick.

        Args:
            ref_paths: dict {carla_actor_id: [(x, y), ...]} world coordinates.
                Each path has ~10-30 waypoints covering ~30m ahead.
        """
        self._vehicle_reference_paths = ref_paths

    def get_perception_output(self, mode: str = 'pred') -> Optional[CsPerceptionOutput]:
        """Get structured perception output for downstream modules.

        Args:
            mode: 'pred' for model predictions, 'gt' for CARLA ground truth.

        Returns:
            CsPerceptionOutput with vehicle states, or None if not available.
        """
        if mode == 'gt':
            return self._build_gt_output()
        elif mode == 'pred':
            return self._build_pred_output()
        else:
            raise ValueError("mode must be 'gt' or 'pred', got '{}'".format(mode))

    def _build_gt_output(self) -> Optional[CsPerceptionOutput]:
        """Build perception output from CARLA ground truth actors."""
        import math
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from cs_cosafe_c.perception.cs_bev_transform import world_to_bev

        if self._rsu_location is None:
            return None

        grid_conf = self._config.bev.motion_grid
        ego_pose = (self._rsu_location.x, self._rsu_location.y, 0.0)
        bev_range = abs(grid_conf['xbound'][0])
        frame_dt = self._config.perception.inference_interval / 20.0
        future_frames = int(self._config.risk.collision_horizon_s / frame_dt)

        ego_ids = self._get_ego_vehicle_ids()
        world = CarlaDataProvider.get_world()
        all_vehicles = world.get_actors().filter('vehicle.*')

        vehicles = []
        for veh in all_vehicles:
            veh_loc = veh.get_location()
            if veh_loc.distance(self._rsu_location) > bev_range + 5:
                continue

            vel = veh.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2) ** 0.5
            yaw = math.radians(veh.get_transform().rotation.yaw)
            px, py = world_to_bev(veh_loc.x, veh_loc.y, grid_conf, ego_pose)

            traj_bev = []
            traj_world = []
            ctrl = veh.get_control()
            steer_rad = ctrl.steer * math.radians(70.0)
            wheelbase = self._config.bicycle.lf + self._config.bicycle.lr
            cur_x, cur_y, cur_yaw = veh_loc.x, veh_loc.y, yaw
            cur_speed = speed
            for t in range(1, future_frames + 1):
                if ctrl.brake > 0.01:
                    acc = -ctrl.brake * self._config.selector.brake_decel
                elif ctrl.throttle > 0.01:
                    acc = ctrl.throttle * 3.0
                else:
                    acc = 0.0
                cur_speed = max(0.0, cur_speed + acc * frame_dt)
                cur_yaw = cur_yaw + (cur_speed * math.tan(steer_rad) / wheelbase) * frame_dt
                cur_x = cur_x + cur_speed * math.cos(cur_yaw) * frame_dt
                cur_y = cur_y + cur_speed * math.sin(cur_yaw) * frame_dt
                traj_world.append((cur_x, cur_y))
                fpx, fpy = world_to_bev(cur_x, cur_y, grid_conf, ego_pose)
                traj_bev.append((fpx, fpy))

            is_ads = veh.id in ego_ids
            ads_vid = -1
            if is_ads and self._ego_vehicles:
                for i, ev in enumerate(self._ego_vehicles):
                    if ev is not None and ev.id == veh.id:
                        ads_vid = i
                        break

            vehicles.append(CsVehicleState(
                instance_id=veh.id,
                position_bev=(px, py),
                position_world=(veh_loc.x, veh_loc.y),
                trajectory_bev=traj_bev,
                trajectory_world=traj_world,
                is_ads_vehicle=is_ads,
                ads_vehicle_id=ads_vid,
                heading_rad=yaw,
                speed=speed,
            ))

        return CsPerceptionOutput(
            scenes=[vehicles],
            mode='gt',
            step=self._step_count,
            ego_vehicle_ids=[v.id for v in (self._ego_vehicles or [])
                             if v is not None and v.is_alive],
        )

    def _build_pred_output(self) -> Optional[CsPerceptionOutput]:
        """Build perception output from model predictions + instance tracker."""
        from cs_cosafe_c.perception.cs_bev_transform import bev_to_world, world_to_bev

        if self._last_predictions is None:
            return None

        grid_conf = self._config.bev.motion_grid
        ego_pose = (self._rsu_location.x, self._rsu_location.y, 0.0)
        matching = self._last_predictions.get('matching', {})
        ego_ids = self._get_ego_vehicle_ids()
        filter_cfg = self._config.perception

        bev_res = grid_conf['xbound'][2]
        frame_dt = self._config.perception.inference_interval / 20.0

        def _centers_to_vehicles(centers, seg_counts=None, inst_flows=None):
            vehs = []
            dropped = {
                'weak_pixels': 0,
                'short_traj': 0,
                'jumping': 0,
                'off_lane': 0,
            }
            if not centers:
                return vehs, dropped
            for inst_id, trajectory in centers.items():
                if not trajectory or len(trajectory) == 0:
                    continue
                pos_bev = (trajectory[0][0], trajectory[0][1])
                pos_world = bev_to_world(pos_bev[0], pos_bev[1], grid_conf, ego_pose)
                traj_bev = [(p[0], p[1]) for p in trajectory[1:]]
                traj_world = [bev_to_world(p[0], p[1], grid_conf, ego_pose)
                              for p in trajectory[1:]]
                carla_id = matching.get(inst_id, -1)
                is_ads = carla_id in ego_ids if carla_id >= 0 else False
                if not self._should_keep_pred_vehicle(
                        carla_id=carla_id,
                        position_world=pos_world,
                        trajectory_world=traj_world,
                        segment_pixels=(seg_counts or {}).get(inst_id),
                        cfg=filter_cfg,
                        dropped=dropped):
                    continue

                speed = 0.0
                heading = 0.0
                if is_ads and self._ego_vehicles:
                    for ev in self._ego_vehicles:
                        if ev is not None and ev.is_alive and ev.id == carla_id:
                            vel = ev.get_velocity()
                            speed = math.hypot(vel.x, vel.y)
                            heading = math.radians(ev.get_transform().rotation.yaw)
                            gt_loc = ev.get_transform().location
                            pos_world = (gt_loc.x, gt_loc.y)
                            pos_bev = world_to_bev(gt_loc.x, gt_loc.y, grid_conf, ego_pose)
                            break
                else:
                    flow = (inst_flows or {}).get(inst_id)
                    if flow is not None:
                        d_row, d_col = flow
                        dx_m = d_col * bev_res
                        dy_m = d_row * bev_res
                        speed = math.hypot(dx_m, dy_m) / frame_dt
                        if abs(dx_m) > 1e-6 or abs(dy_m) > 1e-6:
                            heading = math.atan2(dy_m, dx_m)

                vehicle_id = carla_id if carla_id >= 0 else inst_id
                vehs.append(CsVehicleState(
                    instance_id=vehicle_id,
                    position_bev=pos_bev,
                    position_world=pos_world,
                    trajectory_bev=traj_bev,
                    trajectory_world=traj_world,
                    is_ads_vehicle=is_ads,
                    ads_vehicle_id=carla_id,
                    heading_rad=heading,
                    speed=speed,
                ))
            return vehs, dropped

        all_samples = self._last_predictions.get('all_samples', [])
        scenes = []
        scene_drop_stats = []
        for sample in all_samples:
            mc_k = sample.get('matched_centers', {})
            sc_k = sample.get('segmentpixelscnt', {})
            if_k = sample.get('instance_flows', {})
            vehicles_k, dropped_k = _centers_to_vehicles(mc_k, sc_k, if_k)
            scenes.append(vehicles_k)
            scene_drop_stats.append(dropped_k)

        if not scenes:
            matched_centers = self._last_predictions.get('matched_centers')
            seg_counts = self._last_predictions.get('segmentpixelscnt', {})
            inst_flows = self._last_predictions.get('instance_flows', {})
            vehicles_0, dropped_0 = _centers_to_vehicles(
                matched_centers, seg_counts, inst_flows)
            scenes = [vehicles_0]
            scene_drop_stats = [dropped_0]

        total_dropped = {
            'weak_pixels': 0,
            'short_traj': 0,
            'jumping': 0,
            'off_lane': 0,
        }
        for stats in scene_drop_stats:
            for key in total_dropped:
                total_dropped[key] += stats.get(key, 0)
        if any(total_dropped.values()):
            print("[CoSafe] Pred safety filter dropped instances: {}".format(
                total_dropped))

        if scenes and scenes[-1]:
            parts = []
            for v in scenes[-1]:
                tag = "ads" if v.is_ads_vehicle else "npc"
                parts.append("id{}({})={:.2f}m/s h={:.1f}deg".format(
                    v.instance_id, tag, v.speed,
                    math.degrees(v.heading_rad)))
            print("[CoSafe] Pred flow speeds: {}".format(", ".join(parts)))

        return CsPerceptionOutput(
            scenes=scenes,
            mode='pred',
            step=self._step_count,
            ego_vehicle_ids=[v.id for v in (self._ego_vehicles or [])
                             if v is not None and v.is_alive],
        )

    def _should_keep_pred_vehicle(
        self,
        carla_id: int,
        position_world: Tuple[float, float],
        trajectory_world: List[Tuple[float, float]],
        segment_pixels: Optional[List[int]],
        cfg,
        dropped: Dict[str, int],
    ) -> bool:
        """Gate unmatched predicted instances before safety consumes them."""
        if carla_id >= 0:
            return True

        if not cfg.pred_filter_unmatched:
            return True

        if segment_pixels:
            if segment_pixels[0] < cfg.pred_min_present_pixels:
                dropped['weak_pixels'] += 1
                return False

        if len(trajectory_world) + 1 < cfg.pred_min_trajectory_points:
            dropped['short_traj'] += 1
            return False

        if self._has_unphysical_motion(
                position_world, trajectory_world,
                cfg.pred_max_step_displacement_m):
            dropped['jumping'] += 1
            return False

        if (cfg.pred_require_driving_lane
                and not self._is_near_driving_lane(
                    position_world, cfg.pred_driving_lane_tolerance_m)):
            dropped['off_lane'] += 1
            return False

        return True

    @staticmethod
    def _has_unphysical_motion(
        position_world: Tuple[float, float],
        trajectory_world: List[Tuple[float, float]],
        max_step_displacement_m: float,
    ) -> bool:
        points = [position_world] + list(trajectory_world)
        for idx in range(len(points) - 1):
            dx = points[idx + 1][0] - points[idx][0]
            dy = points[idx + 1][1] - points[idx][1]
            if math.hypot(dx, dy) > max_step_displacement_m:
                return True
        return False

    def _get_carla_map(self):
        if self._carla_map is None:
            try:
                from srunner.scenariomanager.carla_data_provider import (
                    CarlaDataProvider)
                self._carla_map = CarlaDataProvider.get_map()
            except Exception:
                self._carla_map = None
        return self._carla_map

    def _is_near_driving_lane(
        self,
        position_world: Tuple[float, float],
        tolerance_m: float,
    ) -> bool:
        carla_map = self._get_carla_map()
        if carla_map is None:
            return True

        try:
            query_loc = carla.Location(
                x=position_world[0], y=position_world[1], z=0.0)
            waypoint = carla_map.get_waypoint(
                query_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                return False
            wp_loc = waypoint.transform.location
            return math.hypot(
                wp_loc.x - position_world[0],
                wp_loc.y - position_world[1],
            ) <= tolerance_m
        except Exception:
            return True

    def _get_all_vehicles_in_range(self):
        """Get all CARLA vehicles within BEV range of the RSU.

        Used for instance tracking: matches BEV predicted instances against
        ALL vehicles (ADS + NPC), not just ego vehicles.
        """
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        if self._rsu_location is None:
            return list(self._ego_vehicles or [])

        bev_range = abs(self._config.bev.motion_grid['xbound'][0])
        world = CarlaDataProvider.get_world()
        all_vehicles = world.get_actors().filter('vehicle.*')

        in_range = []
        for veh in all_vehicles:
            if not veh.is_alive:
                continue
            if veh.get_location().distance(self._rsu_location) <= bev_range + 5:
                in_range.append(veh)
        return in_range

    def destroy(self):
        """Cleanup RSU sub-modules."""
        if self._perception is not None:
            self._perception.destroy()
            self._perception = None
        self._tracker = None
        self._visualizer = None
        self._last_predictions = None
        if self._risk_oracle is not None:
            self._risk_oracle.destroy()
            self._risk_oracle = None
        self._last_risk_assessment = None
        self._wote_generator = None
        self._last_wote_results = None
        self._candidate_selector = None
        self._last_selection = None
        self._last_selection_step = -1
        self._vehicle_reference_paths = {}
        if self._mpc_controller is not None:
            self._mpc_controller.reset()
            self._mpc_controller = None
        print("[CoSafe] RSU module destroyed")
