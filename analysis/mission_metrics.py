"""
任务指标分析模块

此模块提供任务结果的统计分析功能，包括路径长度、时间、
能量消耗、风险评估等指标的计算和格式化输出。

主要功能：
- 路径几何计算
- 任务性能指标
- 结果摘要和格式化
"""

from typing import Dict, List, Tuple, Optional
import math

from models.mission_models import MissionResult


Point3D = Tuple[float, float, float]


def compute_path_length(path_xyz: List[Point3D]) -> float:
    """
    计算三维路径总长度（米）。
    """
    if not path_xyz or len(path_xyz) < 2:
        return 0.0

    total_length_m = 0.0
    for i in range(len(path_xyz) - 1):
        x0, y0, z0 = path_xyz[i]
        x1, y1, z1 = path_xyz[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        total_length_m += math.sqrt(dx * dx + dy * dy + dz * dz)

    return total_length_m


def compute_horizontal_path_length(path_xyz: List[Point3D]) -> float:
    """
    计算路径在 XY 平面上的总长度（米）。
    """
    if not path_xyz or len(path_xyz) < 2:
        return 0.0

    total_length_m = 0.0
    for i in range(len(path_xyz) - 1):
        x0, y0, _ = path_xyz[i]
        x1, y1, _ = path_xyz[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        total_length_m += math.hypot(dx, dy)

    return total_length_m


def compute_altitude_range(path_xyz: List[Point3D]) -> Tuple[float, float]:
    """
    计算路径高度范围 (min_z, max_z)。
    """
    if not path_xyz:
        return 0.0, 0.0

    zs = [p[2] for p in path_xyz]
    return min(zs), max(zs)


def summarize_mission_result(mission_result: MissionResult) -> Dict[str, float | int | bool | str | None]:
    """
    从 MissionResult 中提取结构化指标摘要。
    """
    actual_path = mission_result.actual_flown_path_xyz
    no_wind_path = mission_result.initial_no_wind_path_xyz
    wind_path = mission_result.initial_wind_path_xyz

    actual_path_length_m = compute_path_length(actual_path)
    actual_horizontal_length_m = compute_horizontal_path_length(actual_path)

    no_wind_path_length_m = compute_path_length(no_wind_path)
    wind_path_length_m = compute_path_length(wind_path)

    min_z, max_z = compute_altitude_range(actual_path)

    summary = {
        "success": mission_result.success,
        "failure_reason": mission_result.failure_reason,
        "total_replans": mission_result.total_replans,
        "total_mission_time_s": mission_result.total_mission_time_s,
        "total_energy_used_j": mission_result.total_energy_used_j,
        "final_remaining_energy_j": mission_result.final_state.remaining_energy_j,
        "actual_path_points": len(actual_path),
        "actual_path_length_m": actual_path_length_m,
        "actual_horizontal_length_m": actual_horizontal_length_m,
        "initial_no_wind_path_length_m": no_wind_path_length_m,
        "initial_wind_path_length_m": wind_path_length_m,
        "executed_path_length_m": actual_path_length_m,
        "actual_min_altitude_m": min_z,
        "actual_max_altitude_m": max_z,
    }

    return summary


def mission_summary_to_row(summary: Dict[str, float | int | bool | str | None]) -> Dict[str, float | int | bool | str | None]:
    """
    将任务摘要整理成适合 CSV / DataFrame 的一行。
    当前直接返回同结构字典，后续可扩展字段顺序或重命名规则。
    """
    return dict(summary)


def format_summary_text(summary: Dict[str, float | int | bool | str | None]) -> str:
    """
    将任务摘要格式化为便于终端打印的文本。
    """
    lines = [
        "================ Mission Metrics ================",
        f"Success                  : {summary['success']}",
        f"Failure reason           : {summary['failure_reason']}",
        f"Total replans            : {summary['total_replans']}",
        f"Total mission time (s)   : {summary['total_mission_time_s']:.2f}",
        f"Total energy used (J)    : {summary['total_energy_used_j']:.2f}",
        f"Remaining energy (J)     : {summary['final_remaining_energy_j']:.2f}",
        f"Actual path points       : {summary['actual_path_points']}",
        f"Executed path length (m) : {summary['executed_path_length_m']:.2f}",
        f"Horizontal length (m)    : {summary['actual_horizontal_length_m']:.2f}",
        f"No-wind init length (m)  : {summary['initial_no_wind_path_length_m']:.2f}",
        f"Wind init length (m)     : {summary['initial_wind_path_length_m']:.2f}",
        f"Min altitude (m)         : {summary['actual_min_altitude_m']:.2f}",
        f"Max altitude (m)         : {summary['actual_max_altitude_m']:.2f}",
        "=================================================",
    ]
    return "\n".join(lines)