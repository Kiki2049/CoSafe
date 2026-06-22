"""CasADi MPC controller for trajectory tracking (Phase 5).

Given a reference trajectory (t_plan steps @ dt_plan, world frame) from Phase 4
candidate selection, solves a nonlinear MPC problem at 20Hz to produce
carla.VehicleControl commands.

Solver structure follows UDMC vehicle_obs.py:
- 6-state bicycle model (cs_bicycle_model.py)
- Multi-shooting with IPOPT
- Hard jerk/steer-rate constraints (not just Rd soft penalty)
- Per-vehicle warm-start and previous-control tracking

Reference trajectory handling:
- WoTE outputs (40, 3) starting at t+0.05s (no t=0 current pose).
- On inference ticks, update_reference() prepends the vehicle's current
  pose as t=0 and stores the fixed (41, 3) extended trajectory.
- On every tick, compute_control() uses spatial projection to find where
  the vehicle currently is on the stored trajectory, then interpolates
  N+1 MPC reference points ahead from there.
"""

import math

import casadi as ca
import numpy as np

from cs_cosafe_c.cs_config import CsBicycleModelConfig, CsMpcConfig
from cs_cosafe_c.controllers.cs_bicycle_model import BicycleModel


class CsMpcController:
    """MPC trajectory-tracking controller.

    Builds the NLP once at construction; reuses it every call with different
    parameters (current state + reference + previous control).

    Per-vehicle state (warm-start, previous control, stored trajectory) is
    tracked via dicts keyed by agent_id, so multi-vehicle MPC calls don't
    interfere.
    """

    def __init__(self, mpc_config: CsMpcConfig,
                 bicycle_config: CsBicycleModelConfig):
        self._cfg = mpc_config
        self._model = BicycleModel(bicycle_config, dt=mpc_config.dt)
        self._solver = None
        self._N = mpc_config.horizon
        self._n_s = 6
        self._n_c = 2
        self._lbx = None
        self._ubx = None
        self._lbg = None
        self._ubg = None

        self._warm_starts = {}
        self._prev_controls = {}
        self._ref_trajectories = {}
        self._last_status = {}
        self._ticks_since_ref = {}

        self._build_solver()


    def _build_solver(self):
        """Build the CasADi NLP for multi-shooting MPC.

        Follows UDMC vehicle_obs.py solver_basis() + solver_add_cost() +
        solver_add_bounds() structure.

        Parameter vector P layout:
            P[0 : n_s*(N+1)]  = reference states (column-major, col 0 = init)
            P[n_s*(N+1) : end] = u00 (previous applied control, 2 values)

        Constraint vector g layout:
            [0 : n_s*(N+1)]              = equality: initial state + dynamics
            [n_s*(N+1) : +2]             = first-step rate: U[:,0] - u00
            [+2 : +2*(N-1)]              = jerk: U[:,i+1] - U[:,i]
        """
        f = self._model.get_casadi_function()
        N = self._N
        n_s = self._n_s
        n_c = self._n_c

        U = ca.SX.sym('U', n_c, N)
        X = ca.SX.sym('X', n_s, N + 1)

        n_p = n_s * (N + 1) + n_c
        P_sym = ca.SX.sym('P', n_p)
        P_ref = ca.reshape(P_sym[:n_s * (N + 1)], n_s, N + 1)
        u00 = P_sym[n_s * (N + 1):]

        Q = np.diag(self._cfg.Q)
        R = np.diag(self._cfg.R)
        Rd = np.diag(self._cfg.Rd)

        obj = 0
        g = []

        g.append(X[:, 0] - P_ref[:, 0])

        for i in range(N):
            err = X[0:5, i] - P_ref[0:5, i]
            obj += ca.mtimes([err.T, Q, err])
            obj += ca.mtimes([U[:, i].T, R, U[:, i]])
            if i < N - 1:
                du = U[:, i] - U[:, i + 1]
                obj += ca.mtimes([du.T, Rd, du])
            x_next = f(X[:, i], U[:, i])
            g.append(X[:, i + 1] - x_next)

        g.append(U[:, 0] - u00)
        for i in range(N - 1):
            g.append(U[:, i + 1] - U[:, i])

        opt_variables = ca.vertcat(
            ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

        lbx = []
        ubx = []
        for _ in range(N):
            lbx += [self._cfg.acc_lbound, -self._cfg.steer_bound]
            ubx += [self._cfg.acc_ubound, self._cfg.steer_bound]
        for _ in range(N + 1):
            lbx += [-1e6, -1e6, -1e6, -1e6, -1e6, -1e6]
            ubx += [1e6, 1e6, 1e6, 1e6, 1e6, 1e6]

        lbg = []
        ubg = []
        n_eq = n_s * (N + 1)
        lbg += [0.0] * n_eq
        ubg += [0.0] * n_eq
        lbg += [-self._cfg.acc_ubound, -self._cfg.steer_bound]
        ubg += [self._cfg.acc_ubound, self._cfg.steer_bound]
        for _ in range(N - 1):
            lbg += [-self._cfg.jerk_bound, -self._cfg.steer_rate_bound]
            ubg += [self._cfg.jerk_bound, self._cfg.steer_rate_bound]

        nlp = {
            'f': obj,
            'x': opt_variables,
            'p': P_sym,
            'g': ca.vertcat(*g),
        }
        opts = {
            'ipopt.max_iter': 200,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
        }
        self._solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._lbx = lbx
        self._ubx = ubx
        self._lbg = lbg
        self._ubg = ubg
        self._n_p = n_p


    def update_reference(self, agent_id, ref_trajectory_world, carla_vehicle):
        """Store a new reference trajectory for an agent (inference tick).

        Prepends the vehicle's current pose as t=0 anchor, creating a fixed
        (41, 3) extended trajectory that compute_control() projects onto.

        Called once per inference tick (every 5 ticks) when Phase 4 produces
        a new selection.  Between inference ticks, compute_control() uses
        spatial projection on this stored trajectory.

        Args:
            agent_id: CARLA actor ID.
            ref_trajectory_world: (40, 3) [x, y, heading], world frame.
                WoTE output starting at t+0.05s.
            carla_vehicle: carla.Vehicle actor (for current pose).
        """
        state = self._get_vehicle_state_6(carla_vehicle)
        current_pose = np.array([[state[0], state[1], state[2]]])
        self._ref_trajectories[agent_id] = np.vstack(
            [current_pose, ref_trajectory_world])
        self._ticks_since_ref[agent_id] = 0


    @staticmethod
    def _project_onto_trajectory(ref_world, current_pos):
        """Find fractional index of the closest point on the trajectory.

        Projects current_pos onto each segment [i, i+1] and returns the
        fractional index (e.g. 3.7 = 70% between waypoint 3 and 4).

        Args:
            ref_world: (T, 3) array [x, y, heading].
            current_pos: (2,) array [x, y].

        Returns:
            float: fractional index in [0, T-1].
        """
        T = ref_world.shape[0]
        best_dist = float('inf')
        best_frac = 0.0

        for i in range(T - 1):
            ax, ay = ref_world[i, 0], ref_world[i, 1]
            bx, by = ref_world[i + 1, 0], ref_world[i + 1, 1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy

            if seg_len_sq < 1e-10:
                t = 0.0
            else:
                t = ((current_pos[0] - ax) * dx
                     + (current_pos[1] - ay) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))

            px = ax + t * dx
            py = ay + t * dy
            dist = math.hypot(current_pos[0] - px, current_pos[1] - py)

            if dist < best_dist:
                best_dist = dist
                best_frac = i + t

        return best_frac

    def _interpolate_reference(self, ref_world, progress):
        """Build MPC reference from time-based progress on stored trajectory.

        Uses the given progress (fractional waypoint index) as the current
        position on the trajectory, then samples N+1 reference points ahead
        at dt_mpc intervals.

        Args:
            ref_world: (41, 3) fixed extended trajectory (from update_reference).
            progress: float, fractional waypoint index (time-based).

        Returns:
            (N+1, 6) reference states [x, y, yaw, vx, vy, omega].
        """
        dt_plan = 0.05
        dt_mpc = self._cfg.dt
        T_plan = ref_world.shape[0]
        t_max = (T_plan - 1) * dt_plan

        t_start = progress * dt_plan

        ref_states = np.zeros((self._N + 1, self._n_s))
        for j in range(self._N + 1):
            t_query = t_start + j * dt_mpc
            t_query = min(t_query, t_max)

            idx = t_query / dt_plan
            i0 = int(idx)
            i0 = min(i0, T_plan - 1)
            i1 = min(i0 + 1, T_plan - 1)
            alpha = idx - i0

            ref_states[j, 0] = (1 - alpha) * ref_world[i0, 0] + alpha * ref_world[i1, 0]
            ref_states[j, 1] = (1 - alpha) * ref_world[i0, 1] + alpha * ref_world[i1, 1]
            h0 = ref_world[i0, 2]
            h1 = ref_world[i1, 2]
            dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
            ref_states[j, 2] = h0 + alpha * dh
            if i1 > i0:
                dx = ref_world[i1, 0] - ref_world[i0, 0]
                dy = ref_world[i1, 1] - ref_world[i0, 1]
                ref_states[j, 3] = np.hypot(dx, dy) / dt_plan
            elif j > 0:
                ref_states[j, 3] = ref_states[j - 1, 3]

        return ref_states


    @staticmethod
    def _get_vehicle_state_6(carla_vehicle):
        """Extract (x, y, yaw, vx, vy, omega) from a CARLA vehicle actor."""
        loc = carla_vehicle.get_location()
        yaw = math.radians(carla_vehicle.get_transform().rotation.yaw)
        vel = carla_vehicle.get_velocity()
        ang = carla_vehicle.get_angular_velocity()

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        vx = vel.x * cos_y + vel.y * sin_y
        vy = -vel.x * sin_y + vel.y * cos_y
        omega = math.radians(ang.z)

        vx = max(vx, 0.1)

        return np.array([loc.x, loc.y, yaw, vx, vy, omega])

    def _set_status(self, agent_id, status, **kwargs):
        """Record the latest MPC execution status for one vehicle."""
        info = {'status': status}
        info.update(kwargs)
        self._last_status[agent_id] = info


    def compute_control(self, carla_vehicle, agent_id):
        """Compute MPC control for one vehicle (called every tick).

        Uses spatial projection on the stored extended trajectory
        (set by update_reference on inference ticks) to find where
        the vehicle is, then solves the NLP.

        Args:
            carla_vehicle: carla.Vehicle actor.
            agent_id: int, CARLA actor ID.

        Returns:
            carla.VehicleControl or None if no trajectory or solver fails.
        """
        ref_ext = self._ref_trajectories.get(agent_id)
        if ref_ext is None:
            self._set_status(agent_id, 'no_reference')
            return None

        state = self._get_vehicle_state_6(carla_vehicle)

        ticks = self._ticks_since_ref.get(agent_id, 0)
        self._ticks_since_ref[agent_id] = ticks + 1
        dt_plan = 0.05
        progress = ticks * self._cfg.dt / dt_plan

        ref = self._interpolate_reference(ref_ext, progress)

        T_plan = ref_ext.shape[0]
        if progress >= T_plan - 1.5:
            self._set_status(
                agent_id,
                'reference_exhausted',
                progress=float(progress),
                ref_points=int(T_plan),
            )
            return None

        P_ref = ref.T.copy()
        P_ref[:, 0] = state

        u00 = self._prev_controls.get(agent_id, np.zeros(self._n_c))
        p_flat = np.concatenate([P_ref.flatten(order='F'), u00])

        n_opt = self._n_c * self._N + self._n_s * (self._N + 1)
        x0 = self._warm_starts.get(agent_id)
        if x0 is None or len(x0) != n_opt:
            x0 = np.zeros(n_opt)

        try:
            sol = self._solver(
                x0=x0, p=p_flat,
                lbx=self._lbx, ubx=self._ubx,
                lbg=self._lbg, ubg=self._ubg)
            stats = self._solver.stats()
            if not stats['success']:
                self._warm_starts.pop(agent_id, None)
                self._set_status(
                    agent_id,
                    'solver_failed',
                    progress=float(progress),
                    return_status=str(stats.get('return_status', 'unknown')),
                )
                return None
        except Exception as e:
            self._warm_starts.pop(agent_id, None)
            self._set_status(
                agent_id,
                'solver_exception',
                progress=float(progress),
                error=str(e),
            )
            return None

        x_opt = np.array(sol['x']).flatten()
        self._warm_starts[agent_id] = x_opt

        acc = float(x_opt[0])
        steer_rad = float(x_opt[1])

        self._prev_controls[agent_id] = np.array([acc, steer_rad])

        control = self._to_carla_control(acc, steer_rad)
        self._set_status(
            agent_id,
            'ok',
            progress=float(progress),
            acc=acc,
            steer_rad=steer_rad,
            throttle=float(control.throttle),
            brake=float(control.brake),
            steer=float(control.steer),
        )

        return control

    def _to_carla_control(self, acc, steer_rad):
        """Convert (acceleration, steering_angle_rad) to carla.VehicleControl."""
        import carla

        control = carla.VehicleControl()

        control.steer = float(np.clip(
            steer_rad / self._cfg.steer_bound, -1.0, 1.0))

        if acc >= 0:
            control.throttle = float(np.clip(
                acc / self._cfg.acc_ubound, 0.0, 0.75))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = float(np.clip(
                -acc / (-self._cfg.acc_lbound), 0.0, 1.0))

        return control

    def reset(self, agent_id=None):
        """Clear per-vehicle state.

        Args:
            agent_id: clear state for this vehicle only, or all if None.
        """
        if agent_id is None:
            self._warm_starts.clear()
            self._prev_controls.clear()
            self._ref_trajectories.clear()
            self._last_status.clear()
            self._ticks_since_ref.clear()
        else:
            self._warm_starts.pop(agent_id, None)
            self._prev_controls.pop(agent_id, None)
            self._ref_trajectories.pop(agent_id, None)
            self._last_status.pop(agent_id, None)
            self._ticks_since_ref.pop(agent_id, None)

    def get_last_status(self, agent_id=None):
        """Return the latest MPC execution status."""
        if agent_id is None:
            return dict(self._last_status)
        return self._last_status.get(agent_id)
