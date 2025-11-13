"""
enhanced_features.py
====================

このモジュールは、Rikeiba アプリに拡張機能を追加するためのヘルパ関数を提供します。

主な機能:

1. 3D コース＆レース・シミュレーション
   - PredTime_s などの予測タイムから各馬の進捗率を計算し、円形トラック上で移動する 3D アニメーションを作成します。
   - 高低差があれば slope を与えることで Z 軸方向にも変化を付けられます。

2. ヒートマップとレーダーチャートによる特徴可視化
   - 複数の指標（RecencyZ, StabZ, SectionZ, PhysicsZ, SpecFitZ など）を馬ごとに比較するレーダーチャートと、
     全馬×指標のヒートマップを生成します。

3. AI 実況コメント生成
   - AR100 などの指標に基づき、レース展開風のコメントを簡易生成します。

このファイルをプロジェクトに追加し、keiba_web_app.py からインポートして利用してください。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def simulate_3d_race(df: pd.DataFrame, distance_m: float, *, slope: float = 0.0, n_frames: int = 20) -> go.Figure:
    """予測タイムとトラック長から 3D レースアニメーションを生成します。

    Parameters
    ----------
    df : DataFrame
        '馬名' と 'PredTime_s' 列を含む DataFrame。PredTime_s は予測ゴールタイム（秒）。
    distance_m : float
        レース距離 [m]。
    slope : float, optional
        トラック全長に対する高低差の割合 (m/m)。デフォルト 0.0。例えば 0.005 なら全長で 0.5% の上り。
    n_frames : int, optional
        アニメーションフレーム数。

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Streamlit で表示可能な 3D アニメーション。
    """
    # 必要な列確認
    if 'PredTime_s' not in df.columns:
        raise ValueError("DataFrame に 'PredTime_s' 列が必要です")
    names = df['馬名'].astype(str).tolist()
    times = pd.to_numeric(df['PredTime_s'], errors='coerce').to_numpy(float)
    # 最大タイムとトラックの円半径を算出（単位調整用に縮尺を適用）
    max_time = float(np.nanmax(times)) if len(times) else 1.0
    # 見栄えのため、半径を距離の 1/6 程度に縮小
    radius = distance_m / (2.0 * np.pi) / 6.0
    # Z 軸勾配を距離長×slope で定義
    elev_total = distance_m * slope
    # 各フレーム時刻
    time_steps = np.linspace(0.0, max_time, n_frames)
    frames = []
    # 初期座標計算
    for t in time_steps:
        xs, ys, zs = [], [], []
        for i, tm in enumerate(times):
            # 進捗率（各馬の経過時間/ゴールタイム）
            progress = 0.0
            if np.isfinite(tm) and tm > 0:
                progress = min(t / tm, 1.0)
            angle = 2.0 * np.pi * progress
            xs.append(np.cos(angle) * radius)
            ys.append(np.sin(angle) * radius)
            zs.append(elev_total * progress)
        frames.append(go.Frame(
            data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                              marker=dict(size=5), text=names)],
            name=f"{t:.2f}"
        ))
    # 初期表示は最初のフレーム
    init_x = frames[0].data[0].x
    init_y = frames[0].data[0].y
    init_z = frames[0].data[0].z
    fig = go.Figure(data=[go.Scatter3d(x=init_x, y=init_y, z=init_z,
                                       mode='markers', marker=dict(size=5), text=names)],
                    frames=frames)
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
                        args=[None, {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}]
                    )
                ]
            )
        ]
    )
    return fig


def _ensure_specfitz(df: pd.DataFrame) -> pd.DataFrame:
    """
    SpecFitZ 列を堅牢に生成するヘルパ関数。

    - 既存の SpecFitZ が利用可能ならそれを使用し、Z スコア化して 0-100 スケールにする。
    - 見つからない / 全欠損の場合は TurnPrefPts, PacePts, DistTurnZ から平均を取りフォールバック。
    - 分散が極小の場合は微小なノイズを加えて可視化に耐える値に調整。

    Parameters
    ----------
    df : DataFrame
        SpecFitZ またはフォールバックに利用できる列を含む DataFrame。

    Returns
    -------
    df_out : DataFrame
        SpecFitZ 列を追加した DataFrame。
    """
    d = df.copy()
    src_col = None
    # 既に SpecFitZ が存在して有効値がある場合はそれを使用
    if 'SpecFitZ' in d.columns and pd.to_numeric(d['SpecFitZ'], errors='coerce').notna().any():
        src_col = 'SpecFitZ'
    else:
        # 代替候補
        for alt in ['SpecFit', 'SpecFitCoeff', 'SpecFit_score']:
            if alt in d.columns and pd.to_numeric(d[alt], errors='coerce').notna().any():
                src_col = alt
                break
    if src_col:
        s = pd.to_numeric(d[src_col], errors='coerce')
    else:
        # フォールバック: TurnPrefPts, PacePts, DistTurnZ のうち存在するものを平均
        parts = []
        for alt in ['TurnPrefPts', 'PacePts', 'DistTurnZ']:
            if alt in d.columns:
                parts.append(pd.to_numeric(d[alt], errors='coerce'))
        if parts:
            s = sum(parts) / len(parts)
        else:
            # 何もない場合はゼロ列
            d['SpecFitZ'] = 0.0
            return d
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        d['SpecFitZ'] = 0.0
        return d
    # 分散が小さい場合は微小ジッターを加える
    std = float(np.nanstd(s))
    if std < 1e-9:
        # 行番号でわずかな差分を付与
        jitter = pd.Series(np.linspace(-1e-4, 1e-4, len(s)), index=s.index)
        s = s.fillna(s.mean()) + jitter
    # Z スコア化し 0-100 に写像
    m = float(np.nanmean(s))
    std = float(np.nanstd(s)) or 1.0
    z = (s - m) / std
    z = z.clip(-3.0, 3.0)
    d['SpecFitZ'] = (z + 3.0) * (100.0 / 6.0)
    return d


def plot_radar_and_heatmap(df: pd.DataFrame, metrics: list[str] | None = None) -> tuple[go.Figure, go.Figure]:
    """
    指定した指標に基づくレーダーチャートとヒートマップを生成する。

    この関数では以下の安全処理を行います。

    * 指定された指標列を数値化し、各列を 0–100 のスケールに正規化します。
    * 列の値に分散がない場合（全頭同じ値、ほぼ同値）、レーダーには 50 点として描きます。
      ヒートマップにはそのまま表示しますが警告を表示します。
    * SpecFitZ 列は存在しない場合や分散が極小の場合に堅牢な生成を試みます（
      `_ensure_specfitz` を内部的に呼び出します）。

    Parameters
    ----------
    df : DataFrame
        各馬の指標列を含む DataFrame。必須列に '馬名' がある。
    metrics : list[str], optional
        可視化する列名リスト。指定しない場合は RecencyZ, StabZ, SectionZ, PhysicsZ, SpecFitZ を用いる。

    Returns
    -------
    radar_fig : plotly.graph_objects.Figure
        レーダーチャート。全馬をプロット。
    heatmap_fig : plotly.graph_objects.Figure
        馬×指標のヒートマップ。
    """
    if metrics is None:
        metrics = ['RecencyZ', 'StabZ', 'SectionZ', 'PhysicsZ', 'SpecFitZ']
    # 'SpecFitZ' を補完
    df = _ensure_specfitz(df)
    # 馬名存在確認
    if '馬名' not in df.columns:
        raise ValueError("DataFrame に '馬名' 列が必要です")
    # 使用できるメトリクス
    valid_metrics = []
    constant_metrics = []
    missing_metrics = []
    norm_df = pd.DataFrame(index=df.index)
    for col in metrics:
        if col not in df.columns:
            missing_metrics.append(col)
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        # 全部欠損は除外
        if vals.notna().sum() == 0:
            missing_metrics.append(col)
            continue
        # 分散計算
        mn, mx = float(vals.min()), float(vals.max())
        rng = mx - mn
        if not np.isfinite(rng) or rng < 1e-9:
            constant_metrics.append(col)
            norm_df[col] = 50.0
            valid_metrics.append(col)
        else:
            # 0–100 に線形正規化
            norm_df[col] = (vals - mn) / rng * 100.0
            valid_metrics.append(col)
    # 警告表示
    warnings = []
    if missing_metrics:
        warnings.append("欠損のため省略: " + ', '.join(missing_metrics))
    if constant_metrics:
        warnings.append("全頭同一値のためレーダーでは50点として描画: " + ', '.join(constant_metrics))
    if warnings:
        try:
            st.caption(' / '.join(warnings))
        except Exception:
            pass
    # レーダー作成
    radar_fig = go.Figure()
    # 指標リスト
    theta = [col for col in valid_metrics]
    if theta:
        for _, row in norm_df.iterrows():
            values = [float(row[col]) if np.isfinite(row[col]) else 0.0 for col in theta]
            # 閉路にする
            radar_fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=theta + [theta[0]],
                name=str(df.loc[row.name, '馬名'])
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
    else:
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=False)),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
    # ヒートマップ用 (0–100スケール) のみ使用
    heat_df = norm_df[valid_metrics].copy()
    heat_df.index = df['馬名']
    if valid_metrics:
        heatmap_fig = px.imshow(
            heat_df.astype(float),
            labels=dict(x="指標", y="馬名", color="値"),
            x=valid_metrics,
            y=heat_df.index,
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        heatmap_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    else:
        heatmap_fig = go.Figure()
    return radar_fig, heatmap_fig


def generate_ai_commentary(df: pd.DataFrame, *, top_n: int = 3) -> str:
    """予測順位に基づき簡易実況コメントを生成する。

    Parameters
    ----------
    df : DataFrame
        少なくとも '馬名', 'AR100' または '勝率%_PL' を含む DataFrame。
    top_n : int, optional
        結果コメントに含める上位馬数。

    Returns
    -------
    commentary : str
        複数行の実況コメント。
    """
    # 順位付け
    key_col = 'AR100' if 'AR100' in df.columns else ('勝率%_PL' if '勝率%_PL' in df.columns else None)
    if key_col is None:
        return "レース予測データが不足しているため実況を生成できません。"
    df_sorted = df.sort_values(key_col, ascending=False).reset_index(drop=True)
    comments = []
    if not df_sorted.empty:
        lead = df_sorted.at[0, '馬名']
        comments.append(f"スタートしました！ {lead} が好スタートを切っています。")
        if len(df_sorted) > 1:
            second = df_sorted.at[1, '馬名']
            comments.append(f"続いて {second} が追走。")
        # 中盤の盛り上げ
        comments.append("各馬第3コーナーを通過、展開はどう動くか？")
        # 終盤の実況
        top_horses = df_sorted.head(top_n)
        names = '、'.join([str(n) for n in top_horses['馬名']])
        comments.append(f"最後の直線！ {names} が上位を争っています！")
        # フィニッシュ
        finish = df_sorted.at[0, '馬名']
        comments.append(f"ゴールイン！ 勝ったのは {finish} です！")
    else:
        comments.append("出走馬が見つかりません。")
    return '\n'.join(comments)
