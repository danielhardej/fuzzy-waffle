"""Data preparation for the Streamlit pages."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from weather.metrics import frost_nights, heat_days, longest_dry_spell, monthly_rainfall, wet_days

DEFAULT_STATION_LIMIT = 6


@dataclass(frozen=True)
class OverviewKpis:
    """Headline values for the overview page."""

    total_rainfall: float
    wettest_station: str
    hottest_day: str
    frost_nights: int


@dataclass(frozen=True)
class RainfallKpis:
    """Headline values for the rainfall explorer page."""

    total_rainfall: float
    wettest_station: str
    driest_station: str
    wet_days: int
    dry_spell_days: int
    dry_spell_station: str


def filter_observations(
    df: pd.DataFrame, regions: list[str], start_date: date, end_date: date
) -> pd.DataFrame:
    """Filter observations by selected regions and inclusive dates."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df[df["region"].isin(regions) & df["date"].between(start, end)].copy()


def overview_kpis(df: pd.DataFrame, frost_threshold: float) -> OverviewKpis:
    """Calculate the four overview headline values."""
    station_rain = df.groupby("station")["rainfall_mm"].sum(min_count=1).dropna()
    wettest = station_rain.idxmax() if not station_rain.empty else "Not available"
    valid_heat = df.dropna(subset=["max_temp_c"])
    if valid_heat.empty:
        hottest = "Not available"
    else:
        row = valid_heat.loc[valid_heat["max_temp_c"].idxmax()]
        hottest = f"{row['station']} · {row['max_temp_c']:.1f} °C"
    return OverviewKpis(
        total_rainfall=float(df["rainfall_mm"].sum()),
        wettest_station=str(wettest),
        hottest_day=hottest,
        frost_nights=frost_nights(df, frost_threshold),
    )


def region_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly rainfall totals by region."""
    return monthly_rainfall(df, "region")


def daily_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily average maximum and minimum temperatures."""
    return (
        df.groupby("date", as_index=False)[["max_temp_c", "min_temp_c"]]
        .mean()
        .sort_values("date")
    )


def map_totals(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Join selected-period station rainfall totals to map coordinates."""
    totals = df.groupby("station_id", as_index=False)["rainfall_mm"].sum(min_count=1)
    return stations.merge(totals, on="station_id", how="inner")


def station_monthly_summary(
    df: pd.DataFrame, frost_threshold: float, heat_threshold: float
) -> pd.DataFrame:
    """Build a monthly station summary with rain and temperature events."""
    rows: list[dict[str, object]] = []
    for month, month_data in df.groupby(df["date"].dt.to_period("M")):
        rows.append(
            {
                "Month": month.to_timestamp(),
                "Rainfall (mm)": month_data["rainfall_mm"].sum(min_count=1),
                "Mean max (°C)": month_data["max_temp_c"].mean(),
                "Mean min (°C)": month_data["min_temp_c"].mean(),
                "Frost nights": frost_nights(month_data, frost_threshold),
                "Heat days": heat_days(month_data, heat_threshold),
            }
        )
    return pd.DataFrame(rows)


def default_stations(df: pd.DataFrame) -> list[str]:
    """Pick the first station alphabetically in each region, up to a small limit."""
    pairs = df[["region", "station"]].drop_duplicates().sort_values(["region", "station"])
    return pairs.drop_duplicates("region")["station"].head(DEFAULT_STATION_LIMIT).tolist()


def filter_stations(df: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    """Keep observations for the selected station names."""
    return df[df["station"].isin(stations)].copy()


def rainfall_kpis(df: pd.DataFrame, wet_threshold_mm: float) -> RainfallKpis:
    """Calculate the rainfall explorer headline values."""
    totals = df.groupby("station")["rainfall_mm"].sum(min_count=1).dropna()
    spells = {
        str(station): longest_dry_spell(group, wet_threshold_mm)
        for station, group in df.groupby("station")
    }
    spell_station = max(spells, key=lambda name: spells[name]) if spells else None
    return RainfallKpis(
        total_rainfall=float(df["rainfall_mm"].sum()),
        wettest_station=str(totals.idxmax()) if not totals.empty else "Not available",
        driest_station=str(totals.idxmin()) if not totals.empty else "Not available",
        wet_days=wet_days(df, wet_threshold_mm),
        dry_spell_days=spells[spell_station] if spell_station else 0,
        dry_spell_station=spell_station or "Not available",
    )


def rainfall_matrix(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Return rainfall by station (rows) and week or month (columns), wettest station first.

    Periods with no valid observations stay NaN rather than becoming zero.
    """
    period = df["date"].dt.to_period("W-SUN" if freq == "W" else "M").dt.start_time
    matrix = (
        df.assign(period=period)
        .groupby(["station", "period"])["rainfall_mm"]
        .sum(min_count=1)
        .unstack("period")
    )
    order = matrix.sum(axis=1).sort_values(ascending=False, kind="stable").index
    return matrix.loc[order]


def cumulative_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Return running rainfall totals per station across the full date span.

    Missing readings add nothing, so the total stays flat across them.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "station", "cumulative_mm"])
    span = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    frames = []
    for station, group in df.groupby("station"):
        daily = group.set_index("date")["rainfall_mm"].reindex(span).fillna(0)
        frames.append(
            pd.DataFrame(
                {"date": span, "station": station, "cumulative_mm": daily.cumsum().to_numpy()}
            )
        )
    return pd.concat(frames, ignore_index=True)


def rainfall_comparison(df: pd.DataFrame, wet_threshold_mm: float) -> pd.DataFrame:
    """Summarise rainfall per station, including how much of the period has data."""
    if df.empty:
        return pd.DataFrame()
    days_in_range = (df["date"].max() - df["date"].min()).days + 1
    rows: list[dict[str, object]] = []
    for station, group in df.groupby("station"):
        valid = int(group["rainfall_mm"].notna().sum())
        wet = wet_days(group, wet_threshold_mm)
        rows.append(
            {
                "Station": station,
                "Region": group["region"].iloc[0],
                "Total (mm)": group["rainfall_mm"].sum(min_count=1),
                "Wet days": wet,
                "Wet-day rate (%)": wet / valid * 100 if valid else float("nan"),
                "Heaviest day (mm)": group["rainfall_mm"].max(),
                "Longest dry spell (days)": longest_dry_spell(group, wet_threshold_mm),
                "Data coverage (%)": valid / days_in_range * 100,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Total (mm)", ascending=False, na_position="last", kind="stable")
        .reset_index(drop=True)
    )


def rain_calendar(df: pd.DataFrame, wet_threshold_mm: float) -> pd.DataFrame:
    """Return a station-by-date grid: 1 for wet days, 0 for dry days, NaN when missing."""
    rainfall = (
        df.groupby(["station", "date"])["rainfall_mm"].sum(min_count=1).unstack("date").sort_index()
    )
    if rainfall.empty:
        return rainfall
    span = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    rainfall = rainfall.reindex(columns=span)
    return rainfall.ge(wet_threshold_mm).astype(float).where(rainfall.notna())