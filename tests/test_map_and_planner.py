import unittest
import numpy as np
from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import BaseWindModel
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner


class MapManagerTests(unittest.TestCase):
    def setUp(self):
        # force fake map by pointing to invalid path and small size
        self.cfg = SimulationConfig()
        self.cfg.map_path = 'nonexistent.file'
        self.cfg.target_size = (20, 20)
        self.mapm = MapManager(self.cfg)

    def test_bounds_and_altitude(self):
        min_x, max_x, min_y, max_y = self.mapm.get_bounds()
        self.assertTrue(min_x < max_x)
        self.assertTrue(min_y < max_y)
        # altitude at center should lie between min_alt and max_alt
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        alt = self.mapm.get_altitude(cx, cy)
        self.assertGreaterEqual(alt, self.cfg.min_alt)
        self.assertLessEqual(alt, self.cfg.max_alt)

    def test_gradient_and_roughness(self):
        # pick some points within bounds
        for x in [min(self.mapm.x), max(self.mapm.x)]:
            for y in [min(self.mapm.y), max(self.mapm.y)]:
                gx, gy = self.mapm.get_gradient(x, y)
                self.assertIsInstance(gx, float)
                self.assertIsInstance(gy, float)
                # gradient magnitude should be finite
                self.assertFalse(np.isnan(gx) or np.isnan(gy))
                z0 = self.mapm.get_roughness(x, y)
                self.assertGreaterEqual(z0, 0.0)


class PlannerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimulationConfig()
        self.cfg.map_path = 'nonexistent.file'
        self.cfg.target_size = (20, 20)
        # limit steps so test finishes quickly
        self.cfg.max_steps = 5000
        self.mapm = MapManager(self.cfg)
        # create a trivial wind model that always returns zero wind
        class ZeroWind(BaseWindModel):
            def get_wind(self, x, y, z, terrain_gradient, z0):
                return np.array([0.0, 0.0])

        self.wind = ZeroWind()
        self.est = StateEstimator(self.mapm, self.wind, self.cfg)
        self.physics = PhysicsEngine(self.cfg)
        self.planner = AStarPlanner(self.cfg, self.est, self.physics)
        bounds = self.est.get_bounds()
        # pick a location near the center and a nearby goal one step away
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        self.start = (cx, cy)
        # step a couple of grid cells horizontally
        step = self.mapm.resolution
        self.goal = (cx + step * 2, cy)

    def test_trivial_path(self):
        # start == goal should return immediate path
        res = self.planner.search(self.start, self.start)
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 1)

    def test_estimator_basic(self):
        # estimator should return wind vector and a non-negative risk value
        cx, cy = self.start
        wind = self.est.get_wind(cx, cy, 10.0)
        self.assertEqual(wind.shape, (2,))
        risk = self.est.get_risk(cx, cy)
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0.0)
    def test_nonempty_path(self):
        res = self.planner.search(self.start, self.goal)
        # ensure search returns either a path list or None, but does not crash
        self.assertTrue(res is None or isinstance(res, list))
        if isinstance(res, list):
            self.assertGreaterEqual(len(res), 1)


if __name__ == '__main__':
    unittest.main()
