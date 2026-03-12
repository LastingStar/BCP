from configs.config import SimulationConfig
from models.mission_models import PathPlanResult, SimulationState, MissionResult


def test_mission_models_instantiation():
    cfg = SimulationConfig()

    state = SimulationState(
        current_time_s=0.0,
        position_xyz=(0.0, 0.0, 100.0),
        remaining_energy_j=cfg.battery_capacity_j
    )

    plan = PathPlanResult(success=True)
    result = MissionResult(success=False, final_state=state)

    assert cfg.mission_update_interval_s == 30.0
    assert state.remaining_energy_j == cfg.battery_capacity_j
    assert plan.success is True
    assert result.final_state.position_xyz == (0.0, 0.0, 100.0)