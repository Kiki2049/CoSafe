from _mr_multi_ego import MultiEgoMRWrapper
from Bench2DriveZoo.team_code.uniad_b2d_agent_mr import UniadAgentMR as _InnerUniadMR


def get_entry_point():
    return 'UniadAgentMR'


class UniadAgentMR(MultiEgoMRWrapper):
    INNER_CLS = _InnerUniadMR
