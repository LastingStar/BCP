import numpy as np
import unittest
from configs.config import SimulationConfig
from core.physics import PhysicsEngine


class PhysicsEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.engine = PhysicsEngine(self.config)

    def test_power_for_speed_no_wind(self):
        # with zero wind, power should equal drag + base
        v_air = 10.0
        p = self.engine.power_for_speed(v_air)
        self.assertGreater(p, 0)
        # check monotonic increase
        self.assertGreater(self.engine.power_for_speed(v_air + 1), p)

    def test_find_feasible_speed_feasible(self):
        v_ground = np.array([5.0, 0.0])
        v_wind = np.array([0.0, 0.0])
        feasible, power, used = self.engine.find_feasible_speed(v_ground, v_wind)
        self.assertTrue(feasible)
        self.assertAlmostEqual(used, np.linalg.norm(v_ground))
        self.assertLessEqual(power, self.config.max_power)

    def test_find_feasible_speed_unfeasible(self):
        # use an extreme headwind that would push required airspeed
        # far beyond the drone's max power capability
        v_ground = np.array([5.0, 0.0])
        v_wind = np.array([200.0, 0.0])  # >> max_power threshold
        feasible, power, used = self.engine.find_feasible_speed(v_ground, v_wind)
        self.assertFalse(feasible)
        self.assertEqual(power, float('inf'))
        self.assertEqual(used, 0.0)


if __name__ == '__main__':
    unittest.main()
