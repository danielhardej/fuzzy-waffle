"""Focused tests for rainfall explorer data preparation."""

import pandas as pd

from weather.app_logic import (
    cumulative_rainfall,
    default_stations,
    rain_calendar,
    rainfall_comparison,
    rainfall_kpis,
    rainfall_matrix,
)


def _observations() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"])
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "station": ["Albany"] * 4 + ["Moora"] * 4,
            "region": ["Great Southern"] * 4 + ["Wheatbelt"] * 4,
            "rainfall_mm": [5.0, 0.0, 2.0, 1.0, 0.0, None, 0.0, None],
        }
    )


def test_default_stations_pick_one_per_region_alphabetically() -> None:
    data = pd.DataFrame(
        {
            "region": ["Wheatbelt", "Wheatbelt", "Kimberley"],
            "station": ["Northam", "Merredin", "Kununurra"],
        }
    )

    assert default_stations(data) == ["Kununurra", "Merredin"]


def test_rainfall_matrix_orders_wettest_first_and_keeps_missing_as_nan() -> None:
    matrix = rainfall_matrix(_observations(), "M")

    assert matrix.index.tolist() == ["Albany", "Moora"]
    assert matrix.loc["Albany"].tolist() == [7.0, 1.0]
    assert matrix.loc["Moora", pd.Timestamp("2024-01-01")] == 0.0
    assert pd.isna(matrix.loc["Moora", pd.Timestamp("2024-02-01")])


def test_rainfall_matrix_weeks_start_on_monday() -> None:
    matrix = rainfall_matrix(_observations(), "W")

    assert all(column.dayofweek == 0 for column in matrix.columns)


def test_cumulative_rainfall_stays_flat_across_missing_days() -> None:
    result = cumulative_rainfall(_observations())

    moora = result[result["station"] == "Moora"].set_index("date")["cumulative_mm"]
    assert moora[pd.Timestamp("2024-01-02")] == moora[pd.Timestamp("2024-01-01")]
    assert result["date"].nunique() == 32
    assert result[result["station"] == "Albany"]["cumulative_mm"].iloc[-1] == 8.0


def test_rainfall_comparison_reports_rates_and_coverage() -> None:
    result = rainfall_comparison(_observations(), wet_threshold_mm=1.0).set_index("Station")

    assert result.index.tolist() == ["Albany", "Moora"]
    assert result.loc["Albany", "Wet days"] == 3
    assert result.loc["Albany", "Wet-day rate (%)"] == 75.0
    assert result.loc["Moora", "Wet days"] == 0
    assert result.loc["Moora", "Longest dry spell (days)"] == 1
    assert result.loc["Moora", "Data coverage (%)"] == 2 / 32 * 100


def test_rain_calendar_marks_wet_dry_and_missing_days() -> None:
    grid = rain_calendar(_observations(), wet_threshold_mm=1.0)

    assert grid.loc["Albany", pd.Timestamp("2024-01-01")] == 1.0
    assert grid.loc["Albany", pd.Timestamp("2024-01-02")] == 0.0
    assert pd.isna(grid.loc["Moora", pd.Timestamp("2024-01-02")])
    assert pd.isna(grid.loc["Albany", pd.Timestamp("2024-01-15")])


def test_rainfall_kpis_summarise_selection() -> None:
    kpis = rainfall_kpis(_observations(), wet_threshold_mm=1.0)

    assert kpis.total_rainfall == 8.0
    assert kpis.wettest_station == "Albany"
    assert kpis.driest_station == "Moora"
    assert kpis.wet_days == 3
    assert (kpis.dry_spell_days, kpis.dry_spell_station) == (1, "Albany")


def test_rainfall_kpis_without_valid_rainfall_use_fallbacks() -> None:
    data = _observations().assign(rainfall_mm=float("nan"))

    kpis = rainfall_kpis(data, wet_threshold_mm=1.0)

    assert kpis.wettest_station == "Not available"
    assert kpis.wet_days == 0
    assert kpis.dry_spell_days == 0
