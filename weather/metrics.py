"""Weather metrics used by the app and tests."""

from collections import defaultdict

import pandas as pd


def calc_gdd(df: pd.DataFrame, base_c: float) -> pd.DataFrame:
    """Calculate daily and cumulative growing degree days by station."""
    result = df.copy().sort_values(["station", "date"])
    daily_values: list[float] = []
    cumulative_values: list[float] = []
    running_totals: defaultdict[str, float] = defaultdict(float)

    for _, row in result.iterrows():
        station = str(row["station"])
        if pd.isna(row["max_temp_c"]) or pd.isna(row["min_temp_c"]):
            daily_values.append(float("nan"))
        else:
            daily_gdd = max(0.0, (row["max_temp_c"] + row["min_temp_c"]) / 2 - base_c)
            daily_values.append(daily_gdd)
            running_totals[station] += daily_gdd
        cumulative_values.append(running_totals[station])

    result["gdd"] = daily_values
    result["cumulative_gdd"] = cumulative_values
    return result


def frost_nights(df: pd.DataFrame, threshold_c: float) -> int:
    """Count nights colder than the configured frost threshold."""
    return int(df["min_temp_c"].lt(threshold_c).sum())


def heat_days(df: pd.DataFrame, threshold_c: float) -> int:
    """Count days at or above the configured heat threshold."""
    return int(df["max_temp_c"].ge(threshold_c).sum())


def wet_days(df: pd.DataFrame, threshold_mm: float) -> int:
    """Count days with rainfall at or above the wet-day threshold."""
    return int(df["rainfall_mm"].ge(threshold_mm).sum())


def longest_dry_spell(df: pd.DataFrame, threshold_mm: float) -> int:
    """Return the longest run of consecutive dry days for one station.

    A missing reading or a gap in dates ends the run, so the result is a lower bound.
    """
    longest = current = 0
    previous_date = None
    for date, rainfall in df.sort_values("date")[["date", "rainfall_mm"]].itertuples(index=False):
        is_dry = pd.notna(rainfall) and rainfall < threshold_mm
        contiguous = previous_date is not None and date - previous_date == pd.Timedelta(days=1)
        if is_dry:
            current = current + 1 if contiguous else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_date = date
    return longest


def monthly_rainfall(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Aggregate valid rainfall observations by month and one category."""
    monthly = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
    return (
        monthly.groupby(["month", group_by], as_index=False)["rainfall_mm"]
        .sum(min_count=1)
        .sort_values(["month", group_by])
    )