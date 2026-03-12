import numpy as np

from configs.config import SimulationConfig
from environment.wind_models import WindModelFactory
from environment.map_manager import MapManager
from core.estimator import StateEstimator


def test_background_wind_changes_with_time():
    cfg = SimulationConfig()
    wind_model = WindModelFactory.create("slope", cfg)

    grad = (0.0, 0.0)
    z0 = 0.1
    z_agl = 50.0

    w0 = wind_model.get_wind(0.0, 0.0, z_agl, grad, z0, t_s=0.0)
    w1 = wind_model.get_wind(0.0, 0.0, z_agl, grad, z0, t_s=cfg.wind_time_scale_s / 4.0)

    assert not np.allclose(w0, w1)


def test_estimator_supports_time_argument():
    cfg = SimulationConfig()
    map_manager = MapManager(cfg)
    wind_model = WindModelFactory.create("slope", cfg)
    estimator = StateEstimator(map_manager, wind_model, cfg)

    min_x, max_x, min_y, max_y = estimator.get_bounds()
    x = 0.5 * (min_x + max_x)
    y = 0.5 * (min_y + max_y)

    z_abs = estimator.get_altitude(x, y) + 50.0

    w0 = estimator.get_wind(x, y, z=z_abs, t_s=0.0)
    w1 = estimator.get_wind(x, y, z=z_abs, t_s=cfg.wind_time_scale_s / 4.0)

    assert w0.shape == (2,)
    assert w1.shape == (2,)
    assert not np.allclose(w0, w1)


def test_wind_is_bounded_by_max_wind_speed():
    cfg = SimulationConfig()
    wind_model = WindModelFactory.create("slope", cfg)

    grad = (100.0, 100.0)  # 人为构造大梯度
    z0 = 0.1
    z_agl = 100.0

    w = wind_model.get_wind(0.0, 0.0, z_agl, grad, z0, t_s=0.0)

    assert np.all(np.abs(w) <= cfg.max_wind_speed + 1e-9)