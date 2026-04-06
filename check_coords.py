from configs.config import SimulationConfig
from environment.map_manager import MapManager

config = SimulationConfig()
map_manager = MapManager(config)

print("bounds:", map_manager.get_bounds())
print("center:", (0.0, 0.0))

try:
    print("alt(center):", map_manager.get_altitude(0.0, 0.0))
except Exception as e:
    print("alt(center) error:", e)