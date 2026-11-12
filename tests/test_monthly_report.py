"""Focused tests for monthly rainfall report calculations."""

import math

import pandas as pd
import pytest

from scripts.monthly_report import (
    calculate_monthly_report,
    latest_complete_month,
    render_markdown,
)


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2022-01-01",
                    "2022-01-02",
                    "2023-01-01",
                    "2023-01-01",
                    "2024-02-01",
                ]
            ),
            "station": [
                "Alpha",
                "Alpha",
                "Beta",
                "Beta",
                "Gamma",
                "Alpha",
                "Alpha",
                "Alpha",
                "Beta",
                "Alpha",
            ],
            "region": [
                "North",
                "North",
                "South",
                "South",
                "South",
                "North",
                "North",
                "North",
                "South",
                "North",
            ],
            "rainfall_mm": [10.0, 5.0, 2.0, 3.0, 5.0, 7.0, 3.0, 20.0, 2.0, 1.0],
        }
    )


def test_report_calculates_region_totals_and_all_extreme_ties() -> None:
    report = calculate_monthly_report(_observations(), pd.Period("2024-01", freq="M"))

    assert report.region_totals.set_index("region")["total_mm"].to_dict() == {
        "North": 15.0,
        "South": 10.0,
    }
    assert report.wettest_stations == ("Alpha",)
    assert report.driest_stations == ("Beta", "Gamma")


def test_report_compares_station_totals_with_other_year_average() -> None:
    report = calculate_monthly_report(_observations(), pd.Period("2024-01", freq="M"))
    comparisons = report.station_comparisons.set_index("station")

    assert comparisons.loc["Alpha", "historical_average_mm"] == 15.0
    assert comparisons.loc["Alpha", "difference_mm"] == 0.0
    assert comparisons.loc["Alpha", "difference_percent"] == 0.0
    assert comparisons.loc["Beta", "historical_average_mm"] == 2.0
    assert comparisons.loc["Beta", "difference_mm"] == 3.0
    assert comparisons.loc["Beta", "difference_percent"] == 150.0
    assert math.isnan(comparisons.loc["Gamma", "historical_average_mm"])


def test_zero_historical_average_has_mm_difference_but_no_percentage() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2024-01-01"]),
            "station": ["Alpha", "Alpha"],
            "region": ["North", "North"],
            "rainfall_mm": [0.0, 4.0],
        }
    )

    report = calculate_monthly_report(observations, pd.Period("2024-01", freq="M"))
    comparison = report.station_comparisons.iloc[0]

    assert comparison["difference_mm"] == 4.0
    assert math.isnan(comparison["difference_percent"])


def test_latest_complete_month_uses_the_data_boundary() -> None:
    partial_latest = pd.DataFrame(
        {"date": pd.to_datetime(["2023-12-31", "2024-01-15"])}
    )
    complete_latest = pd.DataFrame({"date": pd.to_datetime(["2024-12-31"])})

    assert latest_complete_month(partial_latest) == pd.Period("2023-12", freq="M")
    assert latest_complete_month(complete_latest) == pd.Period("2024-12", freq="M")


def test_report_raises_clear_error_for_empty_month() -> None:
    with pytest.raises(ValueError, match="No rainfall data is available for 2024-03"):
        calculate_monthly_report(_observations(), pd.Period("2024-03", freq="M"))


def test_markdown_contains_tables_comparisons_and_synthetic_note() -> None:
    report = calculate_monthly_report(_observations(), pd.Period("2024-01", freq="M"))

    markdown = render_markdown(report)

    assert "| Region | Total rainfall (mm) |" in markdown
    assert "**Driest:** Beta, Gamma (5.0 mm)" in markdown
    assert "| Beta | South | 5.0 | 2.0 | +3.0 | +150.0% |" in markdown
    assert "| Gamma | South | 5.0 | N/A | N/A | N/A |" in markdown
    assert "> Note: All data in this report is synthetic." in markdown
    assert markdown.endswith("\n")