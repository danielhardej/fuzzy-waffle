"""Focused tests for weather metrics."""

import math

import pandas as pd

from weather.metrics import (
    calc_gdd,
    frost_nights,
    heat_days,
    longest_dry_spell,
    monthly_rainfall,
    wet_days,
)


def test_calc_gdd_tracks_each_station_and_missing_days() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"]),
            "station": ["Alpha", "Alpha", "Beta", "Beta"],
            "max_temp_c": [20.0, 10.0, 5.0, None],
            "min_temp_c": [10.0, 2.0, -1.0, 1.0],
        }
    )

    result = calc_gdd(observations, base_c=5.0)

    alpha = result[result["station"] == "Alpha"]
    beta = result[result["station"] == "Beta"]
    assert alpha["gdd"].tolist() == [1.0, 10.0]
    assert alpha["cumulative_gdd"].tolist() == [1.0, 11.0]
    assert math.isnan(beta.iloc[0]["gdd"])
    assert beta["cumulative_gdd"].tolist() == [0.0, 0.0]


def test_frost_and_heat_thresholds() -> None:
    observations = pd.DataFrame(
        {
            "min_temp_c": [1.9, 2.0, None, -0.5],
            "max_temp_c": [34.9, 35.0, None, 41.0],
        }
    )

    assert frost_nights(observations, 2.0) == 2
    assert heat_days(observations, 35.0) == 2


def test_monthly_rainfall_groups_and_excludes_missing_values() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"]),
            "region": ["South West", "South West", "South West"],
            "rainfall_mm": [4.5, None, 2.0],
        }
    )

    result = monthly_rainfall(observations, "region")

    assert result["rainfall_mm"].tolist() == [4.5, 2.0]
    assert result["month"].dt.strftime("%Y-%m").tolist() == ["2024-01", "2024-02"]


def test_wet_days_use_inclusive_threshold_and_skip_missing() -> None:
    observations = pd.DataFrame({"rainfall_mm": [0.9, 1.0, None, 12.0]})

    assert wet_days(observations, 1.0) == 2


def test_longest_dry_spell_is_ended_by_wet_missing_and_absent_days() -> None:
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
    )
    observations = pd.DataFrame(
        {"date": dates, "rainfall_mm": [0.0, 0.5, 5.0, 0.0, None, 0.0]}
    )

    assert longest_dry_spell(observations, 1.0) == 2

    with_gap = observations.drop(index=[0, 2])
    assert longest_dry_spell(with_gap, 1.0) == 1


def test_longest_dry_spell_of_empty_frame_is_zero() -> None:
    empty = pd.DataFrame({"date": pd.to_datetime([]), "rainfall_mm": []})

    assert longest_dry_spell(empty, 1.0) == 0