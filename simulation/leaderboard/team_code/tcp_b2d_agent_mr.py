import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLMDRIVER_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
_TCP_PKG = os.path.join(_COLMDRIVER_ROOT, 'Bench2DriveZoo', 'team_code')
if _TCP_PKG not in sys.path:
    sys.path.insert(0, _TCP_PKG)

from _mr_multi_ego import MultiEgoMRWrapper
from Bench2DriveZoo.team_code.tcp_b2d_agent_mr import TCPAgentMR as _InnerTCPMR


def get_entry_point():
    return 'TCPAgentMR'


class _PerSensorAgentName:
    """Stand-in for AutonomousAgent's `agent_name` string.

    `AgentWrapper.setup_sensors` (leaderboard/autoagents/agent_wrapper.py:161)
    decides camera lens distortion via `self._agent.agent_name in ['TCP',...]`.
    `new.ckpt` was trained with `lens_circle_multiplier=3.0` (TCP), but the
    BEVFormer 6 surround cameras need the standard `1.0` (otherwise their
    edges go OOD relative to b2d training).

    We can't change `agent_name` per-sensor through the public API, so we
    install this proxy: it inspects the call stack for the `sensor_spec`
    local that `setup_sensors` iterates over, and only reports 'TCP' when
    the spec being processed is the TCP single-camera (id='rgb'). Outside
    `setup_sensors` (e.g. logs) it behaves like the default name.
    """

    __slots__ = ('_default',)

    def __init__(self, default='auto_agent'):
        self._default = default

    def _current(self):
        import sys
        frame = sys._getframe(1)
        while frame is not None:
            spec = frame.f_locals.get('sensor_spec')
            if isinstance(spec, dict) and spec.get('id') == 'rgb':
                return 'TCP'
            frame = frame.f_back
        return self._default

    def __eq__(self, other):
        return self._current() == other

    def __ne__(self, other):
        return self._current() != other

    def __hash__(self):
        return hash(self._default)

    def __str__(self):
        return self._current()

    def __repr__(self):
        return repr(self._current())


class TCPAgentMR(MultiEgoMRWrapper):
    INNER_CLS = _InnerTCPMR
    _SHARED_ATTRS = ('net', 'bevformer_pipeline', 'inference_only_pipeline', '_im_transform')
    _CACHED_FORWARD_ATTRS = ('net', 'bev_model')

    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        super().setup(path_to_conf_file, ego_vehicles_num)
        self.agent_name = _PerSensorAgentName('auto_agent')
