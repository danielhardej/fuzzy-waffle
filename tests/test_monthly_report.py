"""Focused tests for monthly rainfall report calculations."""

import argparse
import math

import pandas as pd
import pytest

from scripts.monthly_report import (
    latest_complete_month,
    parse_month,
    prepare_report,
    render_report,
)


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-02-01",
                    "2022-02-02",
                    "2022-02-01",
                    "2023-02-01",
                    "2023-02-01",
                    "2024-02-01",
                    "2024-02-02",
                    "2024-02-01",
                    "2024-02-02",
                    "2024-02-01",
                    "2024-02-29",
                    "2024-03-30",
                ]
            ),
            "station_id": ["A", "A", "B", "A", "B", "A", "A", "B", "B", "C", "C", "A"],
            "station": [
                "Alpha",
                "Alpha",
                "Beta",
                "Alpha",
                "Beta",
                "Alpha",
                "Alpha",
                "Beta",
                "Beta",
                "Gamma",
                "Gamma",
                "Alpha",
            ],
            "region": [
                "North",
                "North",
                "South",
                "North",
                "South",
                "North",
                "North",
                "South",
                "South",
                "North",
                "North",
                "North",
            ],
            "rainfall_mm": [4.0, 6.0, 0.0, 20.0, 0.0, 8.0, 2.0, 3.0, 7.0, 4.0, None, 1.0],
        }
    )


def test_latest_complete_month_uses_month_end_present_in_data() -> None:
    assert latest_complete_month(_observations()) == pd.Period("2024-02", freq="M")


def test_prepare_report_calculates_region_totals_and_station_extremes() -> None:
    report = prepare_report(_observations(), pd.Period("2024-02", freq="M"))

    totals = report.region_totals.set_index("region")["rainfall_mm"]
    assert totals.to_dict() == {"North": 14.0, "South": 10.0}
    assert report.wettest_stations == ("Alpha", "Beta")
    assert report.wettest_total_mm == 10.0
    assert report.driest_stations == ("Gamma",)
    assert report.driest_total_mm == 4.0


def test_station_extremes_tolerate_floating_point_noise() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "station_id": ["A", "A", "B"],
            "station": ["Alpha", "Alpha", "Beta"],
            "region": ["North", "North", "South"],
            "rainfall_mm": [0.1, 0.2, 0.3],
        }
    )

    report = prepare_report(observations, pd.Period("2024-01", freq="M"))

    assert report.wettest_stations == ("Alpha", "Beta")
    assert report.driest_stations == ("Alpha", "Beta")


def test_prepare_report_compares_with_same_month_in_other_years() -> None:
    report = prepare_report(_observations(), pd.Period("2024-02", freq="M"))
    comparisons = report.station_comparisons.set_index("station")

    assert comparisons.index.tolist() == ["Alpha", "Beta", "Gamma"]
    assert comparisons.loc["Alpha", "other_year_average_mm"] == 15.0
    assert comparisons.loc["Alpha", "difference_mm"] == -5.0
    assert comparisons.loc["Alpha", "difference_pct"] == pytest.approx(-100 / 3)
    assert comparisons.loc["Beta", "other_year_average_mm"] == 0.0
    assert comparisons.loc["Beta", "difference_mm"] == 10.0
    assert math.isnan(comparisons.loc["Beta", "difference_pct"])
    assert math.isnan(comparisons.loc["Gamma", "other_year_average_mm"])
    assert math.isnan(comparisons.loc["Gamma", "difference_mm"])
    assert math.isnan(comparisons.loc["Gamma", "difference_pct"])


def test_prepare_report_rejects_a_month_without_observations() -> None:
    with pytest.raises(ValueError, match=r"^No observations found for 2025-01\.$"):
        prepare_report(_observations(), pd.Period("2025-01", freq="M"))


def test_all_missing_rainfall_produces_unavailable_report() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
            "station_id": ["A", "B"],
            "station": ["Alpha", "Beta"],
            "region": ["North", "South"],
            "rainfall_mm": [None, None],
        }
    )

    report = prepare_report(observations, pd.Period("2024-04", freq="M"))
    markdown = render_report(report)

    assert report.wettest_stations == ()
    assert report.driest_stations == ()
    assert "- **Wettest:** Not available" in markdown
    assert "- **Driest:** Not available" in markdown
    assert "| North | N/A |" in markdown
    assert "| Alpha | North | N/A | N/A | N/A | N/A |" in markdown


def test_markdown_contains_comparisons_and_synthetic_note() -> None:
    report = prepare_report(_observations(), pd.Period("2024-02", freq="M"))

    markdown = render_report(report)

    assert "# Monthly rainfall report: February 2024" in markdown
    assert "| Region | Total rainfall (mm) |" in markdown
    assert "**Wettest:** Alpha, Beta (10.0 mm)" in markdown
    assert "| Alpha | North | 10.0 | 15.0 | -5.0 | -33.3% |" in markdown
    assert "| Beta | South | 10.0 | 0.0 | +10.0 | N/A |" in markdown
    assert "| Gamma | North | 4.0 | N/A | N/A | N/A |" in markdown
    assert "> Note: All data in this report is synthetic." in markdown
    assert markdown.endswith("\n")


@pytest.mark.parametrize("value", ["2024", "2024-1", "2024-13", "January"])
def test_parse_month_rejects_invalid_formats(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="month must use YYYY-MM format"):
        parse_month(value)
