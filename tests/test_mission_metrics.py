from models.mission_models import SimulationState, MissionResult
from analysis.mission_metrics import (
    compute_path_length,
    compute_horizontal_path_length,
    summarize_mission_result,
    mission_summary_to_row,
    format_summary_text,
)


def build_dummy_mission_result() -> MissionResult:
    actual_path = [
        (0.0, 0.0, 100.0),
        (3.0, 4.0, 100.0),   # 长度 5
        (3.0, 4.0, 112.0),   # 长度 12
    ]

    no_wind_path = [
        (0.0, 0.0, 100.0),
        (10.0, 0.0, 100.0),
    ]

    wind_path = [
        (0.0, 0.0, 100.0),
        (6.0, 8.0, 100.0),
    ]

    final_state = SimulationState(
        current_time_s=60.0,
        position_xyz=(3.0, 4.0, 112.0),
        remaining_energy_j=850000.0,
        traveled_path_xyz=actual_path,
        replans_count=2,
        total_energy_used_j=150000.0,
        is_goal_reached=True,
        is_mission_failed=False,
        failure_reason=None,
    )

    return MissionResult(
        success=True,
        final_state=final_state,
        initial_no_wind_path_xyz=no_wind_path,
        initial_wind_path_xyz=wind_path,
        replanned_paths_xyz=[],
        actual_flown_path_xyz=actual_path,
        total_replans=2,
        total_mission_time_s=60.0,
        total_energy_used_j=150000.0,
        failure_reason=None,
    )


def test_compute_path_length():
    path = [
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 0.0),   # 5
        (3.0, 4.0, 12.0),  # 12
    ]
    assert compute_path_length(path) == 17.0


def test_compute_horizontal_path_length():
    path = [
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 10.0),  # 平面长度 5
        (6.0, 8.0, 20.0),  # 平面长度 5
    ]
    assert compute_horizontal_path_length(path) == 10.0


def test_summarize_mission_result():
    result = build_dummy_mission_result()
    summary = summarize_mission_result(result)

    assert summary["success"] is True
    assert summary["total_replans"] == 2
    assert summary["total_mission_time_s"] == 60.0
    assert summary["total_energy_used_j"] == 150000.0
    assert summary["final_remaining_energy_j"] == 850000.0
    assert summary["actual_path_points"] == 3
    assert summary["executed_path_length_m"] == 17.0
    assert summary["initial_no_wind_path_length_m"] == 10.0
    assert summary["initial_wind_path_length_m"] == 10.0
    assert summary["actual_min_altitude_m"] == 100.0
    assert summary["actual_max_altitude_m"] == 112.0


def test_mission_summary_to_row():
    result = build_dummy_mission_result()
    summary = summarize_mission_result(result)
    row = mission_summary_to_row(summary)

    assert isinstance(row, dict)
    assert row["success"] is True
    assert row["total_replans"] == 2


def test_format_summary_text():
    result = build_dummy_mission_result()
    summary = summarize_mission_result(result)
    text = format_summary_text(summary)

    assert "Mission Metrics" in text
    assert "Success" in text
    assert "Total replans" in text