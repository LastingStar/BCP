from configs.config import SimulationConfig


class BatteryManager:
    """
    电池管理器：负责电量安全余量、消耗更新与路径可行性判断。
    """

    def __init__(self, config: SimulationConfig):
        self.capacity_j = config.battery_capacity_j
        self.reserve_ratio = config.reserve_energy_ratio

    def get_min_reserve_energy_j(self) -> float:
        """
        返回最小安全保底电量（J）。
        """
        return self.capacity_j * self.reserve_ratio

    def get_usable_energy_j(self) -> float:
        """
        返回可使用电量（J），即总容量减去保底电量。
        """
        return self.capacity_j - self.get_min_reserve_energy_j()

    def can_consume(self, remaining_energy_j: float, required_energy_j: float) -> bool:
        """
        判断当前剩余电量是否允许继续消耗 required_energy_j，
        且消耗后仍保留安全余量。
        """
        return (remaining_energy_j - required_energy_j) >= self.get_min_reserve_energy_j()

    def consume_energy(self, remaining_energy_j: float, used_energy_j: float) -> float:
        """
        更新剩余电量。
        """
        return max(0.0, remaining_energy_j - used_energy_j)

    def is_path_feasible(
        self,
        remaining_energy_j: float,
        estimated_path_energy_j: float,
    ) -> bool:
        """
        判断在当前剩余电量下，一条新路径是否可执行。
        """
        return self.can_consume(remaining_energy_j, estimated_path_energy_j)