"""CoSafe configuration dataclasses.

All default values are derived from:
- DeepAccident configs (BEV grids, camera layout)
- UDMC vehicle_obs.py (bicycle model parameters)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
_DEEPACCIDENT_ROOT = os.environ.get(
    'COSAFE_DEEPACCIDENT_ROOT', os.path.join(_REPO_ROOT, 'DeepAccident'))


@dataclass
class CsBevConfig:
    """BEV grid configuration from DeepAccident."""

    det_grid: Dict[str, List[float]] = field(default_factory=lambda: {
        'xbound': [-51.2, 51.2, 0.8],
        'ybound': [-51.2, 51.2, 0.8],
        'zbound': [-30.0, 30.0, 60.0],
        'dbound': [1.0, 60.0, 1.0],
    })

    motion_grid: Dict[str, List[float]] = field(default_factory=lambda: {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-30.0, 30.0, 60.0],
        'dbound': [1.0, 60.0, 1.0],
    })


@dataclass
class CsCameraConfig:
    """6-camera layout configuration from DeepAccident."""

    camera_names: List[str] = field(default_factory=lambda: [
        'Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight',
        'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight',
    ])

    rsu_width: int = 1600
    rsu_height: int = 900
    rsu_fovs: Tuple[float, ...] = (70.0, 70.0, 70.0, 110.0, 110.0, 110.0)
    rsu_pitch: float = 0.0
    rsu_height_m: float = 3.15
    rsu_yaws: Tuple[float, ...] = (-55.0, 0.0, 55.0, -110.0, 180.0, 110.0)

    veh_width: int = 1600
    veh_height: int = 900
    veh_fovs: Tuple[float, ...] = (70.0, 70.0, 70.0, 110.0, 110.0, 110.0)
    veh_pitch: float = 0.0
    veh_positions: Tuple[Tuple[float, float, float], ...] = (
        (0.27, -0.55, 1.60),
        (0.80,  0.00, 1.60),
        (0.27,  0.55, 1.60),
        (-0.32, -0.55, 1.60),
        (-2.00,  0.00, 1.60),
        (-0.32,  0.55, 1.60),
    )
    veh_yaws: Tuple[float, ...] = (-55.0, 0.0, 55.0, -110.0, 180.0, 110.0)

    sensor_tick: float = 0.0

    max_vehicle_cameras: int = 4


@dataclass
class CsBicycleModelConfig:
    """Bicycle model parameters from UDMC vehicle_obs.py (SLSQP + 70-deg steering)."""

    kf: float = -102129.8307648713
    kr: float = -89999.99208226385
    lf: float = 1.2868463998073212
    lr: float = 1.6031536001974758
    m: float = 1699.9998612883219
    Iz: float = 2699.993738447328

    @property
    def Lk(self) -> float:
        return self.lf * self.kf - self.lr * self.kr


@dataclass
class CsMpcConfig:
    """MPC configuration for future safety controller."""

    enable: bool = True
    horizon: int = 20
    dt: float = 0.05
    Q: List[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 1.0, 0.0])
    R: List[float] = field(default_factory=lambda: [1.0, 1.0])
    Rd: List[float] = field(default_factory=lambda: [1.0, 3.0])
    steer_bound: float = 0.7
    acc_lbound: float = -6.0
    acc_ubound: float = 3.0
    jerk_bound: float = 0.9
    steer_rate_bound: float = 0.2


@dataclass
class CsRsuConfig:
    """RSU placement configuration."""

    height: float = 3.15
    lane_side: str = 'right'
    distance: float = 12.0
    use_junction_center: bool = True


@dataclass
class CsPerceptionConfig:
    """BEVerse V2X perception configuration for Phase 1.

    Default values derived from:
    - DeepAccident/projects/configs/DeepAccident_tiny_v2x_4agents_infra.py
    - DeepAccident/projects/configs/DeepAccident_singleframe_tiny_v2x.py
    """

    enable: bool = True

    model_config_path: str = os.path.join(
        _DEEPACCIDENT_ROOT, 'projects/configs/DeepAccident_tiny_v2x_4agents_infra.py')
    checkpoint_path: str = os.path.join(
        _DEEPACCIDENT_ROOT,
        'work_dirs/DeepAccident_tiny_v2x_4agents_infra_BEVT/latest.pth')
    device: str = 'cuda:0'

    inference_interval: int = 5

    receptive_field: int = 3
    future_frames: int = 4

    max_agents: int = 5

    orig_height: int = 900
    orig_width: int = 1600
    final_dim: Tuple[int, int] = (256, 704)
    img_norm_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_norm_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    enable_rsu_only_inference: bool = False

    cooperative_perception: bool = True

    pred_filter_unmatched: bool = True
    pred_min_present_pixels: int = 10
    pred_min_trajectory_points: int = 2
    pred_max_step_displacement_m: float = 8.0
    pred_require_driving_lane: bool = True
    pred_driving_lane_tolerance_m: float = 2.5

    save_visualization: bool = True
    visualization_dir: str = 'cosafe_vis'


@dataclass
class CsRiskConfig:
    """Risk assessment configuration for Phase 2."""

    enable: bool = True
    enable_collision: bool = True
    enable_traffic: bool = True
    enable_deadlock: bool = True
    vehicle_length: float = 4.5
    vehicle_width: float = 2.0
    collision_margin: float = 0.2
    v_creep: float = 0.1
    traffic_light_range: float = 50.0
    w_stall: float = 5.0
    v_stall: float = 0.5
    rho_stall: float = 0.8
    w_conflict: float = 8.0
    d_proj: float = 15.0
    collision_horizon_s: float = 1.0
    frame_dt: float = 0.25
    w_hold: float = 5.0


@dataclass
class CsWoTEConfig:
    """WoTE candidate generation configuration for Phase 3."""

    anchors_path: str = os.path.join(
        _PKG_DIR, 'training/kmeans_output_base/k128/trajectory_anchors_128.npy')
    checkpoint_path: str = os.path.join(_PKG_DIR, 'training/output/best.pth')
    num_anchors: int = 128
    t_plan: int = 40
    dt_plan: float = 0.05
    bev_channels: int = 64
    d_model: int = 256
    enable: bool = True


@dataclass
class CsSelectorConfig:
    """Phase 4: Route-aware candidate trajectory selection.

    Step A: Hard pruning (CollisionFree, Drivable) + Route Corridor Filtering.
    Step B: 4-state pairwise conflict table (INDEPENDENT / COLLISION /
            A_YIELDS / B_YIELDS) with cross-frame yield memory, then
            branch-and-bound optimal joint search.
    """
    bg_check_steps: int = 40
    bg_collision_margin: float = 0.2
    road_check_steps: int = 20
    drivable_skip_steps: int = 5
    enable_drivable_check: bool = True
    corridor_width: float = 3.5
    heading_tolerance: float = 0.8
    corridor_eval_steps: List[int] = field(default_factory=lambda: [5, 10, 20])
    ref_path_extend_m: float = 10.0
    fallback_target_dist: float = 10.0
    fallback_angle_tol: float = 0.5
    pool_size: int = 15
    diversity_min_dist: float = 2.0
    diversity_eval_steps: List[int] = field(default_factory=lambda: [10, 20, 39])
    non_go_cost: float = 3.5
    ref_weight: float = 1.0
    bg_weight: float = 1.0
    bg_clearance_decay: float = 2.0
    agent_weight: float = 1.0
    agent_clearance_decay: float = 1.5
    conflict_margin: float = 0.0
    conflict_margin_rate: float = 0.015
    conflict_check_steps: int = 30
    yield_min_gap: int = 20
    brake_decel: float = 10.0
    snap_static_threshold: float = 2.5
    enable: bool = True
    selector_mode: str = 'bnb'


@dataclass
class CsCoSafeConfig:
    """Top-level CoSafe configuration."""

    bev: CsBevConfig = field(default_factory=CsBevConfig)
    camera: CsCameraConfig = field(default_factory=CsCameraConfig)
    bicycle: CsBicycleModelConfig = field(default_factory=CsBicycleModelConfig)
    mpc: CsMpcConfig = field(default_factory=CsMpcConfig)
    rsu: CsRsuConfig = field(default_factory=CsRsuConfig)
    perception: CsPerceptionConfig = field(default_factory=CsPerceptionConfig)
    risk: CsRiskConfig = field(default_factory=CsRiskConfig)
    wote: CsWoTEConfig = field(default_factory=CsWoTEConfig)
    selector: CsSelectorConfig = field(default_factory=CsSelectorConfig)

    perception_mode: str = 'gt'

    def __post_init__(self):
        """Derive frame_dt from inference_interval to keep them in sync."""
        self.risk.frame_dt = self.perception.inference_interval / 20.0
