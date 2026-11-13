"""Write a monthly rainfall report from the cleaned daily weather data."""

import argparse
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class ReportData:
    """Calculated values needed to render one monthly report."""

    month: pd.Period
    region_totals: pd.DataFrame
    station_comparisons: pd.DataFrame
    wettest_stations: tuple[str, ...]
    wettest_total_mm: float | None
    driest_stations: tuple[str, ...]
    driest_total_mm: float | None


def parse_month(value: str) -> pd.Period:
    """Parse a strict YYYY-MM command-line value."""
    if not MONTH_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("month must use YYYY-MM format")
    return pd.Period(value, freq="M")


def latest_complete_month(df: pd.DataFrame) -> pd.Period:
    """Return the latest month whose final calendar day occurs in the data."""
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    month_end_dates = dates[dates.dt.is_month_end]
    if month_end_dates.empty:
        raise ValueError("No complete month found in the observations.")
    return month_end_dates.max().to_period("M")


def _station_totals(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["station_id", "station", "region"], as_index=False, dropna=False)[
            "rainfall_mm"
        ]
        .sum(min_count=1)
        .rename(columns={"rainfall_mm": "total_mm"})
        .sort_values(["station", "station_id"], kind="stable")
        .reset_index(drop=True)
    )


def _extreme_stations(
    totals: pd.DataFrame, mode: str
) -> tuple[tuple[str, ...], float | None]:
    valid = totals.dropna(subset=["total_mm"])
    if valid.empty:
        return (), None
    extreme = valid["total_mm"].max() if mode == "max" else valid["total_mm"].min()
    stations = tuple(
        sorted(
            {
                str(row.station)
                for row in valid.itertuples(index=False)
                if math.isclose(float(row.total_mm), float(extreme), abs_tol=1e-9)
            }
        )
    )
    return stations, float(extreme)


def _station_comparisons(
    df: pd.DataFrame, target_totals: pd.DataFrame, month: pd.Period
) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    historical_rows = df[(dates.dt.month == month.month) & (dates.dt.year != month.year)].copy()
    if historical_rows.empty:
        historical_averages = pd.DataFrame(columns=["station_id", "other_year_average_mm"])
    else:
        historical_rows["year"] = pd.to_datetime(historical_rows["date"]).dt.year
        yearly_totals = (
            historical_rows.groupby(["station_id", "year"], as_index=False)["rainfall_mm"]
            .sum(min_count=1)
            .rename(columns={"rainfall_mm": "year_total_mm"})
        )
        historical_averages = (
            yearly_totals.groupby("station_id", as_index=False)["year_total_mm"]
            .mean()
            .rename(columns={"year_total_mm": "other_year_average_mm"})
        )

    comparisons = target_totals.merge(historical_averages, on="station_id", how="left")
    valid_difference = comparisons["total_mm"].notna() & comparisons[
        "other_year_average_mm"
    ].notna()
    comparisons["difference_mm"] = (
        comparisons["total_mm"] - comparisons["other_year_average_mm"]
    ).where(valid_difference)
    valid_percentage = valid_difference & comparisons["other_year_average_mm"].ne(0)
    comparisons["difference_pct"] = (
        comparisons["difference_mm"] / comparisons["other_year_average_mm"] * 100
    ).where(valid_percentage)
    return comparisons


def prepare_report(df: pd.DataFrame, month: pd.Period) -> ReportData:
    """Calculate regional totals, station extremes and historical comparisons."""
    month = pd.Period(month, freq="M")
    dates = pd.to_datetime(df["date"])
    target_rows = df[dates.dt.to_period("M") == month].copy()
    if target_rows.empty:
        raise ValueError(f"No observations found for {month}.")

    from weather.app_logic import region_rainfall

    region_totals = region_rainfall(target_rows)[["region", "rainfall_mm"]].reset_index(
        drop=True
    )
    station_totals = _station_totals(target_rows)
    comparisons = _station_comparisons(df, station_totals, month)
    wettest_stations, wettest_total = _extreme_stations(station_totals, "max")
    driest_stations, driest_total = _extreme_stations(station_totals, "min")
    return ReportData(
        month=month,
        region_totals=region_totals,
        station_comparisons=comparisons,
        wettest_stations=wettest_stations,
        wettest_total_mm=wettest_total,
        driest_stations=driest_stations,
        driest_total_mm=driest_total,
    )


def _format_number(value: object) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.1f}"


def _format_difference(value: object, suffix: str = "") -> str:
    return "N/A" if pd.isna(value) else f"{float(value):+.1f}{suffix}"


def _format_extreme(stations: tuple[str, ...], total_mm: float | None) -> str:
    if not stations or total_mm is None:
        return "Not available"
    return f"{', '.join(stations)} ({total_mm:.1f} mm)"


def render_report(report: ReportData) -> str:
    """Render calculated report data as deterministic Markdown."""
    month_name = MONTH_NAMES[report.month.month - 1]
    lines = [
        f"# Monthly rainfall report: {month_name} {report.month.year}",
        "",
        "> Note: All data in this report is synthetic.",
        "",
        "## Total rainfall by region",
        "",
        "| Region | Total rainfall (mm) |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {row.region} | {_format_number(row.rainfall_mm)} |"
        for row in report.region_totals.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## Station extremes",
            "",
            f"- **Wettest:** {_format_extreme(report.wettest_stations, report.wettest_total_mm)}",
            f"- **Driest:** {_format_extreme(report.driest_stations, report.driest_total_mm)}",
            "",
            "## Comparison with the same month in other years",
            "",
            "| Station | Region | Total (mm) | Other-year average (mm) | Difference (mm) | Difference (%) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        "| "
        + " | ".join(
            [
                str(row.station),
                str(row.region),
                _format_number(row.total_mm),
                _format_number(row.other_year_average_mm),
                _format_difference(row.difference_mm),
                _format_difference(row.difference_pct, "%"),
            ]
        )
        + " |"
        for row in report.station_comparisons.itertuples(index=False)
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--month",
        type=parse_month,
        help="month to report in YYYY-MM format (default: latest complete month)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Load cleaned observations and write one monthly Markdown report."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from weather.data import load_daily

    observations = load_daily()
    try:
        month = args.month or latest_complete_month(observations)
        markdown = render_report(prepare_report(observations, month))
    except ValueError as error:
        parser.error(str(error))

    output = ROOT / "reports" / f"{month}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
