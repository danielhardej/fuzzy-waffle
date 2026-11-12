"""Write a deterministic monthly rainfall report in Markdown."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if __package__ is None:
    sys.path.insert(0, str(ROOT))

from weather.data import load_daily  # noqa: E402
from weather.metrics import monthly_rainfall  # noqa: E402

MONTH_NAMES = (
    "",
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
class MonthlyReport:
    """Calculated values needed to render one monthly report."""

    month: pd.Period
    region_totals: pd.DataFrame
    station_comparisons: pd.DataFrame
    wettest_stations: tuple[str, ...]
    driest_stations: tuple[str, ...]


def parse_month(value: str) -> pd.Period:
    """Parse a strict YYYY-MM command-line value."""
    try:
        month = pd.Period(value, freq="M")
    except ValueError as error:
        raise argparse.ArgumentTypeError("month must use YYYY-MM format") from error
    if str(month) != value:
        raise argparse.ArgumentTypeError("month must use YYYY-MM format")
    return month


def latest_complete_month(observations: pd.DataFrame) -> pd.Period:
    """Return the latest represented month whose final day is present in the date span."""
    dates = observations["date"].dropna()
    if dates.empty:
        raise ValueError("Cannot determine a complete month because the data has no dates.")

    maximum_date = dates.max().normalize()
    represented_months = sorted(dates.dt.to_period("M").unique(), reverse=True)
    for month in represented_months:
        if month.end_time.normalize() <= maximum_date:
            return month
    raise ValueError("The data does not contain a complete calendar month.")


def calculate_monthly_report(observations: pd.DataFrame, month: pd.Period) -> MonthlyReport:
    """Calculate regional totals, station extremes and historical comparisons."""
    observation_months = observations["date"].dt.to_period("M")
    selected = observations.loc[observation_months.eq(month)].copy()
    if selected.empty or selected["rainfall_mm"].notna().sum() == 0:
        raise ValueError(f"No rainfall data is available for {month}.")

    region_totals = (
        monthly_rainfall(selected, "region")[["region", "rainfall_mm"]]
        .rename(columns={"rainfall_mm": "total_mm"})
        .sort_values("region")
        .reset_index(drop=True)
    )

    selected_totals = selected.groupby("station")["rainfall_mm"].sum(min_count=1)
    valid_totals = selected_totals.dropna()
    wettest_total = valid_totals.max()
    driest_total = valid_totals.min()
    wettest_stations = tuple(sorted(valid_totals[valid_totals.eq(wettest_total)].index))
    driest_stations = tuple(sorted(valid_totals[valid_totals.eq(driest_total)].index))

    historical = observations.loc[
        observations["date"].dt.month.eq(month.month)
        & ~observations["date"].dt.year.eq(month.year)
    ].copy()
    historical["year"] = historical["date"].dt.year
    yearly_totals = (
        historical.groupby(["station", "year"])["rainfall_mm"].sum(min_count=1).dropna()
    )
    historical_averages = yearly_totals.groupby("station").mean()

    rows: list[dict[str, object]] = []
    for station, station_data in selected.groupby("station", sort=True):
        total = selected_totals.loc[station]
        average = historical_averages.get(station, float("nan"))
        difference = total - average if pd.notna(total) and pd.notna(average) else float("nan")
        difference_percent = (
            difference / average * 100
            if pd.notna(difference) and pd.notna(average) and average != 0
            else float("nan")
        )
        rows.append(
            {
                "station": str(station),
                "region": str(station_data["region"].iloc[0]),
                "total_mm": total,
                "historical_average_mm": average,
                "difference_mm": difference,
                "difference_percent": difference_percent,
            }
        )

    return MonthlyReport(
        month=month,
        region_totals=region_totals,
        station_comparisons=pd.DataFrame(rows),
        wettest_stations=wettest_stations,
        driest_stations=driest_stations,
    )


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_number(value: object) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.1f}"


def _format_difference(value: object, suffix: str = "") -> str:
    return "N/A" if pd.isna(value) else f"{float(value):+.1f}{suffix}"


def _extreme_summary(report: MonthlyReport, stations: tuple[str, ...]) -> str:
    total = report.station_comparisons.loc[
        report.station_comparisons["station"].eq(stations[0]), "total_mm"
    ].iloc[0]
    return f"{', '.join(_markdown_cell(station) for station in stations)} ({total:.1f} mm)"


def render_markdown(report: MonthlyReport) -> str:
    """Render calculated report values as stable Markdown."""
    lines = [
        f"# Monthly rainfall report: {MONTH_NAMES[report.month.month]} {report.month.year}",
        "",
        "> Note: All data in this report is synthetic.",
        "",
        "## Total rainfall by region",
        "",
        "| Region | Total rainfall (mm) |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_markdown_cell(row.region)} | {_format_number(row.total_mm)} |"
        for row in report.region_totals.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## Station extremes",
            "",
            f"- **Wettest:** {_extreme_summary(report, report.wettest_stations)}",
            f"- **Driest:** {_extreme_summary(report, report.driest_stations)}",
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
                _markdown_cell(row.station),
                _markdown_cell(row.region),
                _format_number(row.total_mm),
                _format_number(row.historical_average_mm),
                _format_difference(row.difference_mm),
                _format_difference(row.difference_percent, "%"),
            ]
        )
        + " |"
        for row in report.station_comparisons.itertuples(index=False)
    )
    return "\n".join(lines) + "\n"


def write_report(report: MonthlyReport, output_directory: Path = ROOT / "reports") -> Path:
    """Write one report and return its path."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{report.month}.md"
    output_path.write_text(render_markdown(report), encoding="utf-8")
    return output_path


def main() -> None:
    """Load observations and write the requested monthly report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--month",
        type=parse_month,
        help="report month in YYYY-MM format (default: latest complete month)",
    )
    args = parser.parse_args()
    try:
        observations = load_daily()
        month = args.month or latest_complete_month(observations)
        output_path = write_report(calculate_monthly_report(observations, month))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()