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
        # Use a real slope model to exercise the log profile helper
        slope = SlopeWindModel(self.model.config if hasattr(self.model, 'config') else SimulationConfig())
        f1 = slope._log_profile_factor(0.1, 0.01)
        f2 = slope._log_profile_factor(1.0, 0.01)
        self.assertGreaterEqual(f1, 0.0)
        self.assertGreaterEqual(f2, 0.0)
        # should generally increase with height (more wind aloft)
        self.assertLessEqual(f1, f2)

    def test_get_wind_3d(self):
        # pass dummy values for gradient/z0
        w = self.model.get_wind(10, 20, 5, (0.0, 0.0), 0.1)
        # since base_wind returns [1,0], factor >=1, result should be at least [1,0]
        self.assertGreaterEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 0.0)
        self.assertEqual(w.shape, (2,))


if __name__ == '__main__':
    unittest.main()
