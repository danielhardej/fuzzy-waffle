"""Rainfall explorer page for WA Weather Tracker."""

import streamlit as st

from weather.app_logic import (
    cumulative_rainfall,
    default_stations,
    filter_observations,
    filter_stations,
    rain_calendar,
    rainfall_comparison,
    rainfall_kpis,
    rainfall_matrix,
)
from weather.charts import cumulative_rainfall_chart, rain_calendar_chart, rainfall_heatmap_chart
from weather.config import load_config
from weather.data import load_daily
from weather.ui import configure_page, footer, synthetic_notice

configure_page("Rainfall explorer · WA Weather Tracker")
config = load_config()
wet_mm = config.thresholds.wet_day_mm

st.title("Rainfall explorer")
st.write("Compare rainfall patterns across stations and through the year.")
synthetic_notice()

try:
    daily = load_daily()
except FileNotFoundError:
    st.error("Weather data is not available yet. Run `python scripts/generate_data.py` first.")
    st.stop()

all_regions = sorted(daily["region"].unique())
with st.sidebar:
    st.header("Filters")
    selected_regions = st.multiselect("Region", all_regions, default=all_regions)
    region_daily = daily[daily["region"].isin(selected_regions)]
    station_options = sorted(region_daily["station"].unique())
    selected_stations = st.multiselect(
        "Stations",
        station_options,
        default=default_stations(region_daily) if station_options else [],
    )
    selected_dates = st.date_input(
        "Date range",
        value=(daily["date"].min().date(), daily["date"].max().date()),
        min_value=daily["date"].min().date(),
        max_value=daily["date"].max().date(),
        format="DD/MM/YYYY",
    )
    granularity = st.radio("Heatmap period", ["Monthly", "Weekly"], horizontal=True)

if len(selected_dates) != 2 or not selected_stations:
    st.info("Choose at least one station and a complete date range to explore rainfall.")
    footer()
    st.stop()

filtered = filter_stations(
    filter_observations(daily, selected_regions, selected_dates[0], selected_dates[1]),
    selected_stations,
)
if filtered.empty:
    st.info("No observations match those filters. Try a wider date range or other stations.")
    footer()
    st.stop()

kpis = rainfall_kpis(filtered, wet_mm)
columns = st.columns(4)
columns[0].metric("Total rainfall", f"{kpis.total_rainfall:,.0f} mm")
columns[1].metric("Wettest station", kpis.wettest_station)
columns[2].metric("Wet days", f"{kpis.wet_days:,}", help=f"Days with at least {wet_mm:g} mm")
columns[3].metric(
    "Longest dry spell",
    f"{kpis.dry_spell_days} days",
    help="Consecutive days below the wet-day threshold; missing readings end a spell.",
)
columns[3].caption(kpis.dry_spell_station)

missing_days = int(filtered["rainfall_mm"].isna().sum())
if missing_days:
    st.caption(
        f"{missing_days} station-days have no rainfall reading. They appear as gaps and "
        "are not counted as dry."
    )

st.subheader(f"{granularity} rainfall by station")
frequency = "W" if granularity == "Weekly" else "M"
st.plotly_chart(rainfall_heatmap_chart(rainfall_matrix(filtered, frequency)), use_container_width=True)

st.subheader("Cumulative rainfall")
st.plotly_chart(cumulative_rainfall_chart(cumulative_rainfall(filtered)), use_container_width=True)

st.subheader("Wet and dry days")
st.plotly_chart(rain_calendar_chart(rain_calendar(filtered, wet_mm)), use_container_width=True)

st.subheader("Station comparison")
st.dataframe(
    rainfall_comparison(filtered, wet_mm),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Total (mm)": st.column_config.NumberColumn(format="%.1f mm"),
        "Wet-day rate (%)": st.column_config.NumberColumn(format="%.0f%%"),
        "Heaviest day (mm)": st.column_config.NumberColumn(format="%.1f mm"),
        "Data coverage (%)": st.column_config.NumberColumn(format="%.0f%%"),
    },
)

footer()
