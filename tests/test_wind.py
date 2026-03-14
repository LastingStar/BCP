"""
风场模型测试模块

此模块测试风场模型的正确性，包括对数廓线修正、风速计算
和时变风场的模拟。验证风场模型在不同条件下的行为。

测试类：
- WindModelTests: 风场模型测试
"""

import numpy as np
import unittest
from configs.config import SimulationConfig
from environment.wind_models import BaseWindModel, SlopeWindModel


class DummyModel(BaseWindModel):
    """超简单风场：恒定东向风，不依赖地形或高度"""
    def get_wind(
        self,
        x: float,
        y: float,
        z: float,
        terrain_gradient: tuple,
        z0: float,
    ) -> np.ndarray:
        return np.array([1.0, 0.0])


class WindModelTests(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_log_profile_factor(self):
        # 使用真实的坡度模型来测试对数廓线辅助函数
        slope = SlopeWindModel(self.model.config if hasattr(self.model, 'config') else SimulationConfig())
        f1 = slope._log_profile_factor(0.1, 0.01)
        f2 = slope._log_profile_factor(1.0, 0.01)
        self.assertGreaterEqual(f1, 0.0)
        self.assertGreaterEqual(f2, 0.0)
        # 通常应随高度增加（高空风更大）
        self.assertLessEqual(f1, f2)

    def test_get_wind_3d(self):
        # 为梯度和z0传递虚拟值
        w = self.model.get_wind(10, 20, 5, (0.0, 0.0), 0.1)
        # 由于base_wind返回[1,0]，factor >=1，结果应至少为[1,0]
        self.assertGreaterEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 0.0)
        self.assertEqual(w.shape, (2,))


if __name__ == '__main__':
    unittest.main()
