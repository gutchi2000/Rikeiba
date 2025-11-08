"""
Utilities for enhanced visualization and narration features in the Rikeiba app.

This module provides helper functions to add advanced features such as:

* 3D race simulation with animated progress for each horse on a circular track
* Radar and heatmap visualisations for multiple performance metrics
* Simple AI-like race commentary generated from predicted ranking indicators
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def simulate_3d_race(
    df: pd.DataFrame,
    distance_m: float,
    *,
    slope: float = 0.0,
    n_frames: int = 20,
) -> go.Figure:
    """Create a simple 3D race animation based on predicted goal times.

    Parameters
    ----------
    df:
        DataFrame containing ``'馬名'`` and ``'PredTime_s'`` columns.
    distance_m:
        Total course distance in metres.
    slope:
        Ratio of total elevation gain to course length (m/m). Default ``0.0``.
    n_frames:
        Number of animation frames. Higher values generate smoother motion but
        larger figures.

    Returns
    -------
    plotly.graph_objects.Figure
        A 3D animation figure that can be displayed with Streamlit.
    """

    if 'PredTime_s' not in df.columns:
        raise ValueError("DataFrame に 'PredTime_s' 列が必要です")

    names = df['馬名'].astype(str).tolist()
    times = pd.to_numeric(df['PredTime_s'], errors='coerce').to_numpy(float)

    max_time = float(np.nanmax(times)) if len(times) else 1.0
    radius = distance_m / (2.0 * np.pi) / 6.0
    elev_total = distance_m * slope

    time_steps = np.linspace(0.0, max_time, int(max(n_frames, 2)))
    frames: list[go.Frame] = []

    for t in time_steps:
        xs, ys, zs = [], [], []
        for tm in times:
            progress = 0.0
            if np.isfinite(tm) and tm > 0:
                progress = min(t / tm, 1.0)
            angle = 2.0 * np.pi * progress
            xs.append(np.cos(angle) * radius)
            ys.append(np.sin(angle) * radius)
            zs.append(elev_total * progress)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode='markers',
                        marker=dict(size=5),
                        text=names,
                    )
                ],
                name=f"{t:.2f}",
            )
        )

    init_x = frames[0].data[0].x
    init_y = frames[0].data[0].y
    init_z = frames[0].data[0].z
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=init_x,
                y=init_y,
                z=init_z,
                mode='markers',
                marker=dict(size=5),
                text=names,
            )
        ],
        frames=frames,
    )
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Height'),
        margin=dict(l=20, r=20, t=40, b=20),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[
                            None,
                            {
                                'frame': {'duration': 300, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 0},
                            },
                        ],
                    )
                ],
            )
        ],
    )
    return fig


def plot_radar_and_heatmap(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> tuple[go.Figure, go.Figure]:
    """Generate a radar chart and heatmap for the provided metrics."""

    if metrics is None:
        metrics = ['RecencyZ', 'StabZ', 'SectionZ', 'PhysicsZ', 'SpecFitZ']

    for col in metrics:
        if col not in df.columns:
            raise ValueError(f"DataFrame に指標列 {col} が見つかりません")
    if '馬名' not in df.columns:
        raise ValueError("DataFrame に '馬名' 列が必要です")

    radar_fig = go.Figure()
    for _, row in df.iterrows():
        values = [float(row[col]) if np.isfinite(row[col]) else 0.0 for col in metrics]
        radar_fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                name=str(row['馬名']),
            )
        )
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    heat_df = df[['馬名'] + metrics].copy()
    heat_df.set_index('馬名', inplace=True)
    heatmap_fig = px.imshow(
        heat_df[metrics].astype(float),
        labels=dict(x='指標', y='馬名', color='値'),
        x=metrics,
        y=heat_df.index,
        color_continuous_scale='RdBu_r',
        aspect='auto',
    )
    heatmap_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return radar_fig, heatmap_fig


def generate_ai_commentary(df: pd.DataFrame, *, top_n: int = 3) -> str:
    """Create a simple race commentary based on ranking indicators."""

    key_col = 'AR100' if 'AR100' in df.columns else (
        '勝率%_PL' if '勝率%_PL' in df.columns else None
    )
    if key_col is None:
        return "レース予測データが不足しているため実況を生成できません。"

    df_sorted = df.sort_values(key_col, ascending=False).reset_index(drop=True)
    comments: list[str] = []

    if df_sorted.empty:
        return "出走馬が見つかりません。"

    lead = df_sorted.at[0, '馬名']
    comments.append(f"スタートしました！ {lead} が好スタートを切っています。")

    if len(df_sorted) > 1:
        second = df_sorted.at[1, '馬名']
        comments.append(f"続いて {second} が追走。")

    comments.append("各馬第3コーナーを通過、展開はどう動くか？")

    top_horses = df_sorted.head(top_n)
    names = '、'.join([str(n) for n in top_horses['馬名']])
    comments.append(f"最後の直線！ {names} が上位を争っています！")

    finish = df_sorted.at[0, '馬名']
    comments.append(f"ゴールイン！ 勝ったのは {finish} です！")

    return '\n'.join(comments)
