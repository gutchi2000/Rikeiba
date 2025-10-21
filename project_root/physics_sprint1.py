# -*- coding: utf-8 -*-
"""
physics_sprint1.py
Sprint 1 の物理スコア（コーナー v^2/r ・スタート加速 ・終盤勾配）を DataFrame に付与する。

前提:
- course_geometry パッケージ（あなたの環境のまま）を使用
- DF はレース単位で、最低限 'course_id','surface','distance_m','layout','rail_state' があればOK
- 速度は列があれば優先的に使い、無ければレース平均速度で代用

公開関数:
- add_phys_s1_features(df, *, group_cols=("race_id",), band_col=None, verbose=False) -> pd.DataFrame

使用例は本ファイル末尾の docstring を参照。
"""
from __future__ import annotations
import math
from typing import Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# あなたの既存パッケージ
from course_geometry import (
    get_course_geom,  # 幾何を取得
)

# =========================
# 設定（初期値・テーブル）
# =========================

# スパイラルカーブ低減係数（0.0=負荷ゼロ, 1.0=低減なし）
# ※テキストに基づく初期値。後で自由に調整/上書きしてOK。
SPIRAL_COEFF: Dict[str, float] = {
    # 右回り
    "札幌": 1.0,    # 採用なし
    "函館": 0.7,    # 3-4角スパイラル
    "福島": 0.7,    # スパイラル
    "中山": 0.9,    # 明記なし→軽減弱め
    "中京": 0.7,    # スパイラル
    "京都": 0.7,    # スパイラル
    "阪神": 0.7,    # スパイラル
    "小倉": 0.7,    # スパイラル
    # 左回り
    "東京": 0.9,    # 明記なし→軽減弱め
    "新潟": 0.9,    # 直線コース別扱い、外回りは軽減弱め
}

# 代表コーナー半径 r0 [m]（“場×レイアウト”の初期近似）
# ※おおまかに：小回り≒100, 中間≒120, 大回り≒140。後で残差で校正を。
RADIUS_R0: Dict[Tuple[str, str], float] = {
    ("札幌", "内回り"): 105.0,
    ("函館", "内回り"): 105.0,
    ("福島", "内回り"): 100.0,
    ("中山", "内回り"): 115.0,
    ("中山", "外回り"): 120.0,
    ("東京", "外回り"): 145.0,
    ("新潟", "直線"):   1e9,    # 直千はコーナー無し相当
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
    """
    帯域（内/中/外）による半径補正。
    - 内: -width_avg/2
    - 中:  0
    - 外: +width_avg/2
    None/その他: 0
    """
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
    _min = x.min()
    _max = x.max()
    if pd.isna(_min) or pd.isna(_max) or _max - _min <= 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - _min) / (_max - _min)

def _pick_speed_mps(row: pd.Series, distance_m: float) -> float:
    """
    速度[m/s]の代表値を柔軟に作る。
    優先順位:
      1) row['avg_speed_mps']
      2) distance_m / row['final_time_sec']
      3) (前半3F or 後半3F) から 1F平均速度
    どれも無ければ 16.7 m/s (=1000mを60s) を控えめに返す。
    """
    # 1) そのまま列
    if 'avg_speed_mps' in row and pd.notna(row['avg_speed_mps']):
        return float(row['avg_speed_mps'])
    # 2) 総合タイムから
    for k in ('final_time_sec', 'race_time_sec', 'time_sec'):
        if k in row and pd.notna(row[k]) and row[k] > 0:
            return _safe_div(distance_m, row[k], 16.7)
    # 3) 3F系
    for k in ('first3f_sec', 'last3f_sec'):
        if k in row and pd.notna(row[k]) and row[k] > 0:
            one_f_sec = float(row[k]) / 3.0
            return _safe_div(200.0, one_f_sec, 16.7)  # 1F=200m想定
    # fallback
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

# =========================
# メイン計算（1行）
# =========================

def _compute_corner_load_row(row: pd.Series) -> float:
    """
    コーナー負荷（無次元） ~ s * v * theta を近似で算出。
    - s: スパイラル低減
    - v: 代表速度
    - theta: 1周あたりの曲がり角 ≈ π を 1周のコーナー数で割った角を、通過回数分加算
    """
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', 'A')
    band      = row.get('band', None)  # 内/中/外 のどれか（あれば）
    distance  = float(row.get('distance_m', np.nan))

    if not isinstance(course_id, str) or surface != '芝' or not math.isfinite(distance):
        return 0.0

    # 幾何（幅員・直線長等）
    try:
        geom = get_course_geom(course_id, surface, int(distance), layout, rail)
    except Exception:
        return 0.0

    # 半径とスパイラル
    r0 = _represent_radius(course_id, layout)
    width_avg = _avg_track_width(geom)
    r = max(5.0, r0 + _band_delta_r(band, width_avg))
    s = _course_spiral_coeff(course_id)

    # 角度：1周の合計を π（180°）として、1コーナーあたり π/2
    #   → 内回り/外回り共通の簡易近似。通過回数 = num_turns（周回は4）
    num_turns = int(row.get('num_turns', geom.num_turns or 2))
    theta_per_turn = math.pi / 2.0
    theta_total = theta_per_turn * num_turns

    # 速度
    v = _pick_speed_mps(row, distance)

    # 指標
    load = s * v * theta_total * (1.0 / max(r, 1e-6)) * r  # v*theta（上の導出では r が約分する）
    # ↑理屈上 v*theta で十分だが、あとで係数校正しやすいよう load を v*theta に等価化
    return float(max(0.0, load))


def _compute_start_cost_row(row: pd.Series) -> float:
    """
    スタート加速コスト: alpha * v_c^2 * weight(first_turn) + beta * break_loss_sec
    """
    distance  = float(row.get('distance_m', np.nan))
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', 'A')

    if not isinstance(course_id, str) or surface != '芝' or not math.isfinite(distance):
        return 0.0

    # 巡航速度
    v_c = _pick_speed_mps(row, distance)

    # 1角まで距離
    try:
        geom = get_course_geom(course_id, surface, int(distance), layout, rail)
        d_first = geom.start_to_first_turn_m
    except Exception:
        d_first = None
    w_first = _first_turn_weight(d_first)

    # 出遅れ（任意列）
    break_loss = float(row.get('break_loss_sec', 0.0) or 0.0)

    return float(START_ALPHA * (v_c ** 2) * w_first + START_BETA * break_loss)


def _compute_finish_grade_row(row: pd.Series) -> float:
    """
    終盤勾配コスト: gamma * (grade_pct/100) * L_fin
    """
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
    Sprint 1 の物理特徴を df に付与する。

    Parameters
    ----------
    df : pd.DataFrame
        入力DF。最低限の列:
          - course_id, surface, distance_m, layout, rail_state
        あると良い列:
          - final_time_sec / race_time_sec / time_sec （平均速度の推定に使用）
          - first3f_sec / last3f_sec （速度の代替）
          - break_loss_sec（出遅れ秒）
          - finish_uphill_len_m（分かる場は200m以外を指定可能）
          - num_turns（周回レースなどで4にしておくと精度↑）
        あれば使う列:
          - band_col（'内'/'中'/'外' 等の帯域）

    group_cols : tuple
        正規化（min-max）を行うグループキー。通常は ("race_id",)。
        まとめて同一レース内で 0-1 スケールされます。

    band_col : str or None
        帯域の列名。列がなければ None でOK（半径補正は 0 扱い）。

    Returns
    -------
    df_out : pd.DataFrame
        以下の列が追加されます:
          - phys_corner_load_raw, phys_corner_load
          - phys_start_cost_raw,  phys_start_cost
          - phys_finish_grade_raw, phys_finish_grade
          - phys_s1_score  （合成スコア: 0-1）
    """
    df = df.copy()

    # インプットの正規化（列名）
    required = ("course_id", "surface", "distance_m", "layout", "rail_state")
    for k in required:
        if k not in df.columns:
            raise KeyError(f"required column missing: {k}")

    if band_col and band_col in df.columns:
        df["band"] = df[band_col]
    else:
        df["band"] = None

    # 原スコア計算
    if verbose:
        print("[phys_s1] computing raw loads...")

    df["phys_corner_load_raw"] = df.apply(_compute_corner_load_row, axis=1)
    df["phys_start_cost_raw"]  = df.apply(_compute_start_cost_row, axis=1)
    df["phys_finish_grade_raw"]= df.apply(_compute_finish_grade_row, axis=1)

    # グループ内 min-max 正規化
    if verbose:
        print("[phys_s1] min-max normalization within groups:", group_cols)

    def _norm_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["phys_corner_load"]  = _minmax01(g["phys_corner_load_raw"])
        g["phys_start_cost"]   = _minmax01(g["phys_start_cost_raw"])
        g["phys_finish_grade"] = _minmax01(g["phys_finish_grade_raw"])
        # 合成
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
# 使い方（例）
# =========================
"""
# --- 例: keiba_web_app.py での使い方 ---

from course_geometry import register_all_turf
from physics_sprint1 import add_phys_s1_features

register_all_turf()

# races_df はあなたの既存のレース行 DataFrame
# 必須列: course_id,surface,distance_m,layout,rail_state
# あると良い: final_time_sec / first3f_sec / break_loss_sec / num_turns / (帯域列)
races_df = add_phys_s1_features(
    races_df,
    group_cols=("race_id",),   # レース内で正規化
    band_col="band",           # あれば '内'/'中'/'外' 等
    verbose=False,
)

# 追加される列:
#   phys_corner_load_raw, phys_corner_load
#   phys_start_cost_raw,  phys_start_cost
#   phys_finish_grade_raw,phys_finish_grade
#   phys_s1_score
"""
