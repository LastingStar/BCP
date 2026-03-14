"""
物理引擎测试模块

此模块测试物理引擎的功率计算和速度优化功能，验证空气动力学
模型在不同风速和负载条件下的正确性。

测试类：
- PhysicsEngineTests: 物理引擎测试
"""

import numpy as np
import unittest
from configs.config import SimulationConfig
from core.physics import PhysicsEngine


class PhysicsEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.engine = PhysicsEngine(self.config)

    def test_power_for_speed_no_wind(self):
        # 无风时，功率应等于阻力 + 基础功率
        v_air = 10.0
        p = self.engine.power_for_speed(v_air)
        self.assertGreater(p, 0)
        # 检查单调递增
        self.assertGreater(self.engine.power_for_speed(v_air + 1), p)

    def test_find_feasible_speed_feasible(self):
        v_ground = np.array([5.0, 0.0])
        v_wind = np.array([0.0, 0.0])
        feasible, power, used = self.engine.find_feasible_speed(v_ground, v_wind)
        self.assertTrue(feasible)
        self.assertAlmostEqual(used, np.linalg.norm(v_ground))
        self.assertLessEqual(power, self.config.max_power)

    def test_find_feasible_speed_unfeasible(self):
        # 使用极端的逆风，会将所需空速推到
        # 远超无人机最大功率能力
        v_ground = np.array([5.0, 0.0])
        v_wind = np.array([200.0, 0.0])  # >> max_power threshold
        feasible, power, used = self.engine.find_feasible_speed(v_ground, v_wind)
        self.assertFalse(feasible)
        self.assertEqual(power, float('inf'))
        self.assertEqual(used, 0.0)


if __name__ == '__main__':
    unittest.main()
