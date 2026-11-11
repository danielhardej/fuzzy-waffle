"""Plotly figure builders with a shared visual language."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "rain": "#58A6FF",
    "max_temp": "#F85149",
    "min_temp": "#79C0FF",
    "heat": "#D29922",
    "frost": "#BC8CFF",
    "gdd": "#3FB950",
    "accent": "#8250DF",
    "text": "#E6EDF3",
    "grid": "#30363D",
}


def _style(figure: go.Figure, *, height: int = 410) -> go.Figure:
    """Apply the shared transparent chart layout."""
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"], "size": 15},
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], linecolor=COLORS["grid"])
    figure.update_yaxes(gridcolor=COLORS["grid"], linecolor=COLORS["grid"])
    return figure


def monthly_rainfall_chart(df: pd.DataFrame) -> go.Figure:
    """Build grouped monthly rainfall bars by region."""
    figure = go.Figure()
    for region in df["region"].drop_duplicates():
        region_data = df[df["region"] == region]
        figure.add_bar(x=region_data["month"], y=region_data["rainfall_mm"], name=region)
    figure.update_layout(barmode="group", colorway=[COLORS["rain"], COLORS["accent"], "#D29922"])
    figure.update_yaxes(title="Rainfall (mm)")
    return _style(figure)


def temperature_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Build average daily maximum and minimum temperature lines."""
    figure = go.Figure()
    figure.add_scatter(
        x=df["date"], y=df["max_temp_c"], name="Average max", line={"color": COLORS["max_temp"]}
    )
    figure.add_scatter(
        x=df["date"], y=df["min_temp_c"], name="Average min", line={"color": COLORS["min_temp"]}
    )
    figure.update_yaxes(title="Temperature (°C)")
    return _style(figure)


def rainfall_heatmap_chart(matrix: pd.DataFrame) -> go.Figure:
    """Build a station-by-period rainfall heatmap; periods without data stay blank."""
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            colorscale=[[0, "#161B22"], [1, COLORS["rain"]]],
            colorbar={"title": "mm"},
            hoverongaps=False,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y}</b><br>%{x|%d %b %Y}<br>%{z:.1f} mm<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed")
    _style(figure, height=max(260, 70 + 44 * len(matrix.index)))
    figure.update_layout(hovermode="closest")
    return figure


def cumulative_rainfall_chart(df: pd.DataFrame) -> go.Figure:
    """Build one cumulative rainfall line per station."""
    figure = go.Figure()
    for station in df["station"].drop_duplicates():
        station_data = df[df["station"] == station]
        figure.add_scatter(
            x=station_data["date"], y=station_data["cumulative_mm"], name=station, mode="lines"
        )
    figure.update_layout(colorway=[COLORS["rain"], COLORS["accent"], COLORS["heat"], COLORS["gdd"]])
    figure.update_yaxes(title="Cumulative rainfall (mm)")
    return _style(figure)


def rain_calendar_chart(grid: pd.DataFrame) -> go.Figure:
    """Build a station-by-day strip showing wet, dry and missing days."""
    states = {0: ("Missing", COLORS["heat"]), 1: ("Dry", "#21262D"), 2: ("Wet", COLORS["rain"])}
    codes = grid.fillna(-1).to_numpy() + 1
    labels = np.vectorize(lambda code: states[int(code)][0])(codes)
    colorscale = [
        [0, states[0][1]], [1 / 3, states[0][1]],
        [1 / 3, states[1][1]], [2 / 3, states[1][1]],
        [2 / 3, states[2][1]], [1, states[2][1]],
    ]
    figure = go.Figure(
        go.Heatmap(
            z=codes,
            x=grid.columns,
            y=grid.index,
            zmin=0,
            zmax=2,
            colorscale=colorscale,
            showscale=False,
            customdata=labels,
            hovertemplate="<b>%{y}</b><br>%{x|%d %b %Y}<br>%{customdata}<extra></extra>",
        )
    )
    for label, color in states.values():
        figure.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=label,
            marker={"symbol": "square", "size": 10, "color": color},
        )
    figure.update_yaxes(autorange="reversed")
    _style(figure, height=max(200, 90 + 36 * len(grid.index)))
    figure.update_layout(hovermode="closest")
    return figure


def station_map_chart(df: pd.DataFrame) -> go.Figure:
    """Build a token-free station map sized by total rainfall."""
    sizes = df["rainfall_mm"].fillna(0).clip(lower=0)
    max_size = max(float(sizes.max()), 1.0)
    figure = go.Figure(
        go.Scattermap(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            text=df["station"],
            customdata=df[["region", "rainfall_mm"]],
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]}<br>"
                "%{customdata[1]:.1f} mm<extra></extra>"
            ),
            marker={
                "color": COLORS["rain"],
                "opacity": 0.8,
                "size": 10 + 32 * sizes / max_size,
            },
        )
    )
    figure.update_layout(
        map={"style": "open-street-map", "center": {"lat": -26.5, "lon": 119.5}, "zoom": 3.2},
        showlegend=False,
        hovermode="closest",
    )
    return _style(figure, height=500)


def station_weather_chart(df: pd.DataFrame, frost_c: float, heat_c: float) -> go.Figure:
    """Build rainfall, temperature, heat-day, and frost-night traces."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(
        x=df["date"], y=df["rainfall_mm"], name="Rainfall", marker_color=COLORS["rain"]
    )
    figure.add_scatter(
        x=df["date"],
        y=df["max_temp_c"],
        name="Max temperature",
        line={"color": COLORS["max_temp"], "width": 2},
        secondary_y=True,
    )
    figure.add_scatter(
        x=df["date"],
        y=df["min_temp_c"],
        name="Min temperature",
        line={"color": COLORS["min_temp"], "width": 2},
        secondary_y=True,
    )
    heat_days = df.loc[df["max_temp_c"].notna() & df["max_temp_c"].ge(heat_c)]
    figure.add_scatter(
        x=heat_days["date"],
        y=heat_days["max_temp_c"],
        name="Heat day",
        mode="markers",
        marker={"color": COLORS["heat"], "symbol": "triangle-up", "size": 10},
        hovertemplate="Date: %{x|%d %b %Y}<br>Temperature: %{y:.1f} °C<br>Event: Heat day<extra></extra>",
        secondary_y=True,
    )
    frost_nights = df.loc[df["min_temp_c"].notna() & df["min_temp_c"].lt(frost_c)]
    figure.add_scatter(
        x=frost_nights["date"],
        y=frost_nights["min_temp_c"],
        name="Frost night",
        mode="markers",
        marker={"color": COLORS["frost"], "symbol": "triangle-down", "size": 10},
        hovertemplate="Date: %{x|%d %b %Y}<br>Temperature: %{y:.1f} °C<br>Event: Frost night<extra></extra>",
        secondary_y=True,
    )
    figure.update_yaxes(title="Rainfall (mm)", secondary_y=False)
    figure.update_yaxes(title="Temperature (°C)", secondary_y=True)
    return _style(figure, height=450)


def gdd_chart(df: pd.DataFrame) -> go.Figure:
    """Build a cumulative growing degree day chart."""
    figure = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_gdd"],
            name="Cumulative GDD",
            fill="tozeroy",
            line={"color": COLORS["gdd"], "width": 3},
            fillcolor="rgba(63,185,80,0.16)",
        )
    )
    figure.update_yaxes(title="Growing degree days (°C days)")
    return _style(figure)