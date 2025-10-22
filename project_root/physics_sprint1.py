# -*- coding: utf-8 -*-
"""
physics_sprint1.py
Sprint 1 の物理スコア（コーナー v^2/r ・スタート加速 ・終盤勾配）を DataFrame に付与する。

公開関数:
- add_phys_s1_features(df, *, group_cols=("race_id",), band_col=None, verbose=False) -> pd.DataFrame
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

# あなたの既存パッケージ
from course_geometry import get_course_geom  # 幾何を取得

# =========================
# 設定（初期値・テーブル）
# =========================

# スパイラルカーブ低減係数（0.0=負荷ゼロ, 1.0=低減なし）
SPIRAL_COEFF: Dict[str, float] = {
    # 右回り
    "札幌": 1.0,
    "函館": 0.7,
    "福島": 0.7,
    "中山": 0.9,
    "中京": 0.7,
    "京都": 0.7,
    "阪神": 0.7,
    "小倉": 0.7,
    # 左回り
    "東京": 0.9,
    "新潟": 0.9,
}

# 代表コーナー半径 r0 [m]（“場×レイアウト”の初期近似）
RADIUS_R0: Dict[Tuple[str, str], float] = {
    ("札幌", "内回り"): 105.0,
    ("函館", "内回り"): 105.0,
    ("福島", "内回り"): 100.0,
    ("中山", "内回り"): 115.0,
    ("中山", "外回り"): 120.0,
    ("東京", "外回り"): 145.0,
    ("新潟", "直線"):   1e9,   # 直千はコーナー無し相当
    ("新潟", "内回り"): 120.0,
    ("新潟", "外回り"): 140.0,
    ("中京", "外回り"): 130.0,
    ("京都", "内回り"): 125.0,
    ("京都", "外回り"): 140.0,
    ("阪神", "内回り"): 120.0,
    ("阪神", "外回り"): 140.0,
    ("小倉", "内回り"): 105.0,
}

# 合成スコアの重み（0-1で正規化後）
WEIGHTS = dict(corner=0.45, start=0.35, finish=0.20)

# スタート加速コストの係数
START_ALPHA = 1.0     # 巡航速度^2 の係数
START_BETA  = 0.5     # 出遅れ秒ペナルティの係数

# 1角まで距離が欠損のときの暫定値（安全側）
FIRST_TURN_DIST_DEFAULT = 800.0  # [m]

# 終盤上り区間の代表長さ（不明時）
FINISH_UPHILL_LEN_DEFAULT = 200.0  # [m]

# =========================
# 内部ユーティリティ
# =========================

def _avg_track_width(geom) -> float:
    """幅員が取れないときは 25m を返す。"""
    wmin, wmax = geom.track_width_min_m, geom.track_width_max_m
    if wmin and wmax:
        return (wmin + wmax) / 2.0
    return 25.0

def _band_delta_r(band: Optional[str], width_avg: float) -> float:
    """帯域（内/中/外）による半径補正。"""
    if not band:
        return 0.0
    b = str(band)
    if b in ("内", "内側", "inside", "inner"):
        return -0.5 * width_avg
    if b in ("外", "外側", "outside", "outer"):
        return +0.5 * width_avg
    return 0.0  # 中

def _course_spiral_coeff(course_id: str) -> float:
    return SPIRAL_COEFF.get(course_id, 0.85)

def _represent_radius(course_id: str, layout: str) -> float:
    return RADIUS_R0.get((course_id, layout), 120.0)

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return float(a) / float(b)
    except Exception:
        return default

def _minmax01(x: pd.Series) -> pd.Series:
    xmin, xmax = x.min(), x.max()
    if pd.isna(xmin) or pd.isna(xmax) or xmax - xmin <= 1e-12:
        # すべて同値なら 0.5 を返して「中立」にする
        return pd.Series(np.full(len(x), 0.5), index=x.index, dtype=float)
    return (x - xmin) / (xmax - xmin)

def _pick_speed_mps(row: pd.Series, distance_m: float) -> float:
    """
    速度[m/s]の代表値：
    1) avg_speed_mps
    2) distance_m / final_time_sec
    3) first3f_sec / last3f_sec から 1F平均
    どれも無ければ 16.7 m/s。
    """
    if 'avg_speed_mps' in row and pd.notna(row['avg_speed_mps']):
        return float(row['avg_speed_mps'])
    for k in ('final_time_sec', 'race_time_sec', 'time_sec'):
        if k in row and pd.notna(row[k]) and row[k] > 0:
            return _safe_div(distance_m, row[k], 16.7)
    for k in ('first3f_sec', 'last3f_sec'):
        if k in row and pd.notna(row[k]) and row[k] > 0:
            one_f_sec = float(row[k]) / 3.0
            return _safe_div(200.0, one_f_sec, 16.7)
    return 16.7

def _first_turn_weight(d_first: Optional[float]) -> float:
    """
    1角まで距離の短さをウェイト化（短いほど重い）。
    d=200m -> 1.0, d=800m -> 0.0 の線形をクリップ。
    """
    d = d_first if (d_first is not None and d_first > 0) else FIRST_TURN_DIST_DEFAULT
    x = (d - 200.0) / (800.0 - 200.0)
    x = max(0.0, min(1.0, x))
    return 1.0 - x

# =================================================
# 帯域を作るユーティリティ（import時には実行しない）
# =================================================
def band_from_waku(gate_no: pd.Series, field_size: pd.Series) -> pd.Series:
    """枠番と頭数から '内/中/外' を機械的に割り当てる。"""
    g = pd.to_numeric(gate_no, errors="coerce")
    f = pd.to_numeric(field_size, errors="coerce").clip(lower=2)
    q = (g - 1.0) / (f - 1.0)  # 0=最内, 1=大外
    bins = [-1e9, 1/3, 2/3, 1e9]
    lab  = ["内", "中", "外"]
    out = pd.cut(q, bins=bins, labels=lab)
    return out.astype(str)

# =========================
# メイン計算（1行）
# =========================

def _compute_corner_load_row(row: pd.Series) -> float:
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', 'A')
    band      = row.get('band', None)
    distance  = float(row.get('distance_m', np.nan))
    if not isinstance(course_id, str) or surface != '芝' or not math.isfinite(distance):
        return 0.0

    try:
        geom = get_course_geom(course_id, surface, int(distance), layout, rail)
    except Exception:
        return 0.0

    r0 = _represent_radius(course_id, layout)
    width_avg = _avg_track_width(geom)
    r = max(5.0, r0 + _band_delta_r(band, width_avg))   # 帯域で半径を動かす
    s = _course_spiral_coeff(course_id)

    nt = row.get('num_turns', None)
    if pd.isna(nt):
        nt = geom.num_turns or 2
    nt = int(nt)

    theta_total = (math.pi / 2.0) * nt
    v = _pick_speed_mps(row, distance)

    # 半径が効く形： s * v^2 * θ / r
    load = s * (v ** 2) * theta_total / max(r, 1e-6)
    return float(max(0.0, load))

def _compute_start_cost_row(row: pd.Series) -> float:
    distance  = float(row.get('distance_m', np.nan))
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', 'A')
    if not isinstance(course_id, str) or surface != '芝' or not math.isfinite(distance):
        return 0.0

    v_c = _pick_speed_mps(row, distance)

    try:
        geom = get_course_geom(course_id, surface, int(distance), layout, rail)
        d_first = geom.start_to_first_turn_m
    except Exception:
        d_first = None
    w_first = _first_turn_weight(d_first)

    break_loss = float(row.get('break_loss_sec', 0.0) or 0.0)

    # 外枠ほど不利（短1角ほど強く）
    lane_pen = 0.0
    gate = row.get('gate_no', None)        # 枠番(1..頭数)
    field = row.get('field_size', None)    # 頭数
    if pd.notna(gate) and pd.notna(field) and float(field) > 1:
        q = (float(gate) - 1.0) / (float(field) - 1.0)  # 0=最内,1=大外
        lane_pen = 0.25 * w_first * q

    return float(START_ALPHA * (v_c ** 2) * w_first + START_BETA * break_loss + lane_pen)

def _compute_finish_grade_row(row: pd.Series) -> float:
    """終盤勾配コスト: (grade_pct/100) * L_fin"""
    distance  = float(row.get('distance_m', np.nan))
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', 'A')

    if not isinstance(course_id, str) or surface != '芝' or not math.isfinite(distance):
        return 0.0

    try:
        geom = get_course_geom(course_id, surface, int(distance), layout, rail)
        grade_pct = geom.finish_grade_pct
    except Exception:
        grade_pct = None

    if not grade_pct:
        return 0.0

    L_fin = float(row.get('finish_uphill_len_m', FINISH_UPHILL_LEN_DEFAULT))
    return float((grade_pct / 100.0) * L_fin)

# =========================
# 公開: DataFrame に付与
# =========================

def add_phys_s1_features(
    df: pd.DataFrame,
    *,
    group_cols: Tuple[str, ...] = ("race_id",),
    band_col: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    入力DFに以下の列を追加:
      - phys_corner_load_raw, phys_corner_load
      - phys_start_cost_raw,  phys_start_cost
      - phys_finish_grade_raw, phys_finish_grade
      - phys_s1_score（合成 0-1）
    """
    df = df.copy()

    # 必須列チェック
    required = ("course_id", "surface", "distance_m", "layout", "rail_state")
    for k in required:
        if k not in df.columns:
            raise KeyError(f"required column missing: {k}")

    # band（内/中/外）
    if band_col and band_col in df.columns:
        df["band"] = df[band_col]
    else:
        if {"gate_no", "field_size"}.issubset(df.columns):
            df["band"] = band_from_waku(df["gate_no"], df["field_size"])
        else:
            df["band"] = None

    # 原スコア計算
    if verbose:
        print("[phys_s1] computing raw loads...")

    df["phys_corner_load_raw"]  = df.apply(_compute_corner_load_row, axis=1)
    df["phys_start_cost_raw"]   = df.apply(_compute_start_cost_row,  axis=1)
    df["phys_finish_grade_raw"] = df.apply(_compute_finish_grade_row, axis=1)

    # グループ内 min-max 正規化 → 合成
    if verbose:
        print("[phys_s1] min-max normalization within groups:", group_cols)

    def _norm_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["phys_corner_load"]  = _minmax01(g["phys_corner_load_raw"])
        g["phys_start_cost"]   = _minmax01(g["phys_start_cost_raw"])
        g["phys_finish_grade"] = _minmax01(g["phys_finish_grade_raw"])
        w = WEIGHTS
        g["phys_s1_score"] = (
            w["corner"] * g["phys_corner_load"] +
            w["start"]  * g["phys_start_cost"] +
            w["finish"] * g["phys_finish_grade"]
        )
        return g

    if len(group_cols) == 0:
        df = _norm_group(df)
    else:
        df = df.groupby(list(group_cols), dropna=False, group_keys=False).apply(_norm_group)

    return df


# =========================
# 使い方（参考）
# =========================
"""
from course_geometry import register_all_turf
from physics_sprint1 import add_phys_s1_features

register_all_turf()

# 例：馬ごとに1行持つ DF（同一レース）
df_in = pd.DataFrame({
    "race_id": "R1",
    "horse":  ["A","B","C","D"],
    "course_id": ["京都"]*4,
    "surface": ["芝"]*4,
    "distance_m": [1800]*4,
    "layout": ["外回り"]*4,
    "rail_state": ["A"]*4,
    "gate_no": [1,4,8,12],
    "field_size": [12]*4,
})

df_out = add_phys_s1_features(df_in, group_cols=("race_id",), band_col=None, verbose=False)
"""
