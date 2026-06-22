from _mr_multi_ego import MultiEgoMRWrapper
from Bench2DriveZoo.team_code.vad_b2d_agent_mr import VadAgentMR as _InnerVadMR


def get_entry_point():
    return 'VadAgentMR'


class VadAgentMR(MultiEgoMRWrapper):
    INNER_CLS = _InnerVadMR
