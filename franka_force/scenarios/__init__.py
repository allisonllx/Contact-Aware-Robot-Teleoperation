from .hit_floor import HitFloorScenario
from .peg_in_hole import PegInHoleScenario
from .push_block import PushBlockScenario

_SCENARIO_REGISTRY = {
    HitFloorScenario.name: HitFloorScenario(),
    PushBlockScenario.name: PushBlockScenario(),
    PegInHoleScenario.name: PegInHoleScenario(),
}

SCENARIOS = tuple(_SCENARIO_REGISTRY)


def get_scenario(name):
    return _SCENARIO_REGISTRY[name]
