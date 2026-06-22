"""CasADi symbolic bicycle model for MPC.

State: (x, y, yaw, vx, vy, omega)
Control: (acc, steer)
Parameters from UDMC vehicle_obs.py (SLSQP + 70-deg steering transformation).
"""

import math

import casadi as ca
import numpy as np

from cs_cosafe_c.cs_config import CsBicycleModelConfig


class BicycleModel:
    """6-state bicycle model with CasADi symbolic dynamics.

    Dynamics from UDMC vehicle_obs.py lines 186-200:
        x'  = x + (vx*cos(yaw) - vy*sin(yaw)) * dt
        y'  = y + (vy*cos(yaw) + vx*sin(yaw)) * dt
        yaw'= yaw + omega * dt
        vx' = vx + acc * dt
        vy' = (m*vx*vy + dt*Lk*omega - dt*kf*steer*vx - dt*m*vx^2*omega)
              / (m*vx - dt*(kf+kr))
        omega' = (Iz*vx*omega + dt*Lk*vy - dt*lf*kf*steer*vx)
                 / (Iz*vx - dt*(lf^2*kf + lr^2*kr))
    """

    def __init__(self, config: CsBicycleModelConfig = None, dt: float = 0.05):
        if config is None:
            config = CsBicycleModelConfig()
        self._config = config
        self._dt = dt

        self._kf = config.kf
        self._kr = config.kr
        self._lf = config.lf
        self._lr = config.lr
        self._m = config.m
        self._Iz = config.Iz
        self._Lk = config.Lk

        self._f = self._build_casadi_function()

    def _build_casadi_function(self):
        """Create the CasADi Function implementing bicycle model dynamics."""
        cx = ca.SX.sym('cx')
        cy = ca.SX.sym('cy')
        cyaw = ca.SX.sym('cyaw')
        cvx = ca.SX.sym('cvx')
        cvy = ca.SX.sym('cvy')
        comega = ca.SX.sym('comega')

        states = ca.vertcat(cx, cy, cyaw, cvx, cvy, comega)

        cacc = ca.SX.sym('cacc')
        csteer = ca.SX.sym('csteer')
        controls = ca.vertcat(cacc, csteer)

        dt = self._dt
        kf = self._kf
        kr = self._kr
        lf = self._lf
        lr = self._lr
        m = self._m
        Iz = self._Iz
        Lk = self._Lk

        rhs = ca.vertcat(
            cx + (cvx * ca.cos(cyaw) - cvy * ca.sin(cyaw)) * dt,
            cy + (cvy * ca.cos(cyaw) + cvx * ca.sin(cyaw)) * dt,
            cyaw + comega * dt,
            cvx + dt * cacc,
            (m * cvx * cvy + dt * Lk * comega - dt * kf * csteer * cvx
             - dt * m * (cvx * cvx) * comega) / (m * cvx - dt * (kf + kr)),
            (Iz * cvx * comega + dt * Lk * cvy - dt * lf * kf * csteer * cvx)
            / (Iz * cvx - dt * (lf * lf * kf + lr * lr * kr)),
        )

        f = ca.Function('f', [states, controls], [rhs],
                         ['input_state', 'control_input'], ['rhs'])
        return f

    def predict(self, state, control, dt=None):
        """Predict next state given current state and control.

        Args:
            state: (x, y, yaw, vx, vy, omega) — 6-element sequence
            control: (acc, steer) — 2-element sequence
            dt: timestep override (uses self._dt if None)

        Returns:
            tuple: (x', y', yaw', vx', vy', omega') next state
        """
        if dt is not None and dt != self._dt:
            old_dt = self._dt
            self._dt = dt
            f = self._build_casadi_function()
            self._dt = old_dt
        else:
            f = self._f

        result = f(np.array(state), np.array(control))
        next_state = np.array(result).flatten()
        return tuple(next_state)

    def get_casadi_function(self):
        """Return the CasADi Function object for use in MPC solvers."""
        return self._f

    @property
    def dt(self):
        return self._dt

    @property
    def n_states(self):
        return 6

    @property
    def n_controls(self):
        return 2
