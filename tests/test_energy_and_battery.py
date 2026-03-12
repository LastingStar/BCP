import numpy as np

from configs.config import SimulationConfig
from core.physics import PhysicsEngine
from core.battery_manager import BatteryManager


def test_segment_energy_positive():
    cfg = SimulationConfig()
    physics = PhysicsEngine(cfg)

    p0 = np.array([0.0, 0.0, 100.0])
    p1 = np.array([100.0, 0.0, 100.0])
    wind = np.array([0.0, 0.0, 0.0])

    energy_j, time_s, power_w = physics.estimate_segment_energy(
        p0, p1, wind, cfg.cruise_speed_mps
    )

    assert energy_j > 0.0
    assert time_s > 0.0
    assert power_w > 0.0


def test_headwind_costs_more_than_tailwind():
    cfg = SimulationConfig()
    physics = PhysicsEngine(cfg)

    p0 = np.array([0.0, 0.0, 100.0])
    p1 = np.array([120.0, 0.0, 100.0])

    tailwind = np.array([5.0, 0.0, 0.0])
    headwind = np.array([-5.0, 0.0, 0.0])

    energy_tail, _, _ = physics.estimate_segment_energy(
        p0, p1, tailwind, cfg.cruise_speed_mps
    )
    energy_head, _, _ = physics.estimate_segment_energy(
        p0, p1, headwind, cfg.cruise_speed_mps
    )

    assert energy_head > energy_tail


def test_climb_costs_more_than_level_flight():
    cfg = SimulationConfig()
    physics = PhysicsEngine(cfg)

    p0 = np.array([0.0, 0.0, 100.0])
    p1_level = np.array([100.0, 0.0, 100.0])
    p1_climb = np.array([100.0, 0.0, 130.0])

    wind = np.array([0.0, 0.0, 0.0])

    energy_level, _, _ = physics.estimate_segment_energy(
        p0, p1_level, wind, cfg.cruise_speed_mps
    )
    energy_climb, _, _ = physics.estimate_segment_energy(
        p0, p1_climb, wind, cfg.cruise_speed_mps
    )

    assert energy_climb > energy_level


def test_battery_path_feasibility():
    cfg = SimulationConfig()
    battery = BatteryManager(cfg)

    remaining_energy_j = 300000.0
    estimated_path_energy_j = 250000.0

    feasible = battery.is_path_feasible(
        remaining_energy_j=remaining_energy_j,
        estimated_path_energy_j=estimated_path_energy_j,
    )

    assert feasible is False


def test_battery_consumption_update():
    cfg = SimulationConfig()
    battery = BatteryManager(cfg)

    remaining_energy_j = 500000.0
    used_energy_j = 120000.0

    updated_energy_j = battery.consume_energy(
        remaining_energy_j=remaining_energy_j,
        used_energy_j=used_energy_j,
    )

    assert updated_energy_j == 380000.0