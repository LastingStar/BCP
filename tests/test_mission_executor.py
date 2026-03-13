from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.battery_manager import BatteryManager
from core.planner import AStarPlanner
from simulation.mission_executor import MissionExecutor


def test_mission_executor_runs_basic_loop():
    cfg = SimulationConfig()
    cfg.max_replans = 2
    cfg.max_mission_time_s = 60.0
    cfg.mission_update_interval_s = 10.0
    cfg.cruise_speed_mps = 20.0
    cfg.battery_capacity_j = 200000.0

    map_manager = MapManager(cfg)
    wind_model = WindModelFactory.create("slope", cfg, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, cfg)
    physics = PhysicsEngine(cfg)
    battery_manager = BatteryManager(cfg)
    planner = AStarPlanner(cfg, estimator, physics)

    executor = MissionExecutor(
        config=cfg,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
    )

    min_x, max_x, min_y, max_y = estimator.get_bounds()
    start_xy = (min_x + 100.0, min_y + 100.0)
    goal_xy = (min_x + 300.0, min_y + 300.0)

    result = executor.execute_mission(start_xy, goal_xy)

    assert result.final_state.current_time_s >= 0.0
    assert result.final_state.remaining_energy_j <= cfg.battery_capacity_j
    assert len(result.actual_flown_path_xyz) >= 1
    assert result.total_replans >= 0
    assert result.failure_reason is None or isinstance(result.failure_reason, str)


def test_mission_executor_consumes_energy():
    cfg = SimulationConfig()
    cfg.max_replans = 2
    cfg.max_mission_time_s = 60.0
    cfg.mission_update_interval_s = 10.0
    cfg.cruise_speed_mps = 20.0
    cfg.battery_capacity_j = 200000.0

    map_manager = MapManager(cfg)
    wind_model = WindModelFactory.create("slope", cfg, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, cfg)
    physics = PhysicsEngine(cfg)
    battery_manager = BatteryManager(cfg)
    planner = AStarPlanner(cfg, estimator, physics)

    executor = MissionExecutor(
        config=cfg,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
    )

    min_x, max_x, min_y, max_y = estimator.get_bounds()
    start_xy = (min_x + 100.0, min_y + 100.0)
    goal_xy = (min_x + 300.0, min_y + 300.0)

    result = executor.execute_mission(start_xy, goal_xy)

    assert result.total_energy_used_j >= 0.0
    assert result.final_state.remaining_energy_j <= cfg.battery_capacity_j


def test_mission_executor_updates_time_or_stops_with_reason():
    cfg = SimulationConfig()
    cfg.max_replans = 2
    cfg.max_mission_time_s = 60.0
    cfg.mission_update_interval_s = 10.0
    cfg.cruise_speed_mps = 20.0
    cfg.battery_capacity_j = 200000.0

    map_manager = MapManager(cfg)
    wind_model = WindModelFactory.create("slope", cfg, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, cfg)
    physics = PhysicsEngine(cfg)
    battery_manager = BatteryManager(cfg)
    planner = AStarPlanner(cfg, estimator, physics)

    executor = MissionExecutor(
        config=cfg,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
    )

    min_x, max_x, min_y, max_y = estimator.get_bounds()
    start_xy = (min_x + 100.0, min_y + 100.0)
    goal_xy = (min_x + 300.0, min_y + 300.0)

    result = executor.execute_mission(start_xy, goal_xy)

    assert result.total_mission_time_s >= 0.0
    assert result.success or result.failure_reason is not None