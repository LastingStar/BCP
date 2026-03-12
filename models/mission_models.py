from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point3D = Tuple[float, float, float]


@dataclass
class PathPlanResult:
    """
    Result of a single path planning call.
    """
    success: bool
    path_xyz: List[Point3D] = field(default_factory=list)
    total_cost: float = 0.0
    estimated_energy_j: float = 0.0
    risk_cost: float = 0.0
    planning_time_s: float = 0.0
    expanded_nodes: int = 0
    failure_reason: Optional[str] = None


@dataclass
class SimulationState:
    """
    Runtime mission state during dynamic execution.
    """
    current_time_s: float
    position_xyz: Point3D
    remaining_energy_j: float
    traveled_path_xyz: List[Point3D] = field(default_factory=list)
    replans_count: int = 0
    total_energy_used_j: float = 0.0
    is_goal_reached: bool = False
    is_mission_failed: bool = False
    failure_reason: Optional[str] = None


@dataclass
class MissionResult:
    """
    Final result of the whole mission execution.
    """
    success: bool
    final_state: SimulationState
    initial_no_wind_path_xyz: List[Point3D] = field(default_factory=list)
    initial_wind_path_xyz: List[Point3D] = field(default_factory=list)
    replanned_paths_xyz: List[List[Point3D]] = field(default_factory=list)
    actual_flown_path_xyz: List[Point3D] = field(default_factory=list)
    total_replans: int = 0
    total_mission_time_s: float = 0.0
    total_energy_used_j: float = 0.0
    failure_reason: Optional[str] = None