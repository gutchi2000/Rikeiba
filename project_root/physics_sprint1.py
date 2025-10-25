# -*- coding: utf-8 -*-
"""
physics_sprint1.py
Sprint 1 の物理スコア（コーナー v^2/r ・スタート加速 ・終盤勾配）。
公開関数:
- add_phys_s1_features(df, *, group_cols=("race_id",), band_col=None, verbose=False) -> pd.DataFrame

主な仕様:
- 最終スコアは「raw合成S_raw → Z正規化 → CDF(0–1) = phys_s1_score」。
- 帯域は連続化：gate_no/field_size → band_q∈[0,1] を半径補正に直接使用。
  * 外部から band_q を与えればそれを優先。離散 band（内/中/外）があれば後方互換で使用。
- 行ごとの final_time_sec を使うため、馬ごとに CornerLoad が連続値化。
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict
from functools import lru_cache

import numpy as np
import pandas as pd

# 既存パッケージ
from course_geometry import get_course_geom  # 幾何を取得

# =========================
# 設定
# =========================

# スパイラルカーブ低減係数（0.0=負荷ゼロ, 1.0=低減なし）
SPIRAL_COEFF: Dict[str, float] = {
    "札幌": 1.0, "函館": 0.7, "福島": 0.7, "中山": 0.9, "中京": 0.7, "京都": 0.7, "阪神": 0.7, "小倉": 0.7,
    "東京": 0.9, "新潟": 0.9,
}

# 代表コーナー半径 r0 [m]
RADIUS_R0: Dict[Tuple[str, str], float] = {
    ("札幌", "内回り"): 105.0, ("函館", "内回り"): 105.0, ("福島", "内回り"): 100.0,
    ("中山", "内回り"): 115.0, ("中山", "外回り"): 120.0, ("東京", "外回り"): 145.0,
    ("新潟", "直線"): 1e9, ("新潟", "内回り"): 120.0, ("新潟", "外回り"): 140.0,
    ("中京", "外回り"): 130.0, ("京都", "内回り"): 125.0, ("京都", "外回り"): 140.0,
    ("阪神", "内回り"): 120.0, ("阪神", "外回り"): 140.0, ("小倉", "内回り"): 105.0,
}

# 表示用 0–1 合成の重み（互換用）
WEIGHTS_SHOW = dict(corner=0.45, start=0.35, finish=0.20)

# ★ raw合成の重み
W_CORNER = 1.0
W_START  = 1.0
W_FINISH = 1.0

# スケール
K_CORNER = 1000.0    # (num_turns / distance_m) をkm相当に
K_START  = 200.0     # 1 / start_to_first_turn_m のスケール
# finish_raw = |finish_grade_pct| [%] をそのまま

# Z基準フォールバック
Z_N_MIN_STRICT = 30
Z_N_MIN_LOOSE  = 5
EPS = 1e-6

# 既定
FIRST_TURN_DIST_DEFAULT = 800.0   # [m]
FINISH_UPHILL_LEN_DEFAULT = 200.0 # [m]

# 連続帯域の半径補正ゲイン（幅員×この係数×(q-0.5)）
BAND_RADIUS_GAIN = 1.0

# レイアウト候補
LAYOUT_OPTS: Dict[str, Tuple[str, ...]] = {
    "札幌": ("内回り",), "函館": ("内回り",), "福島": ("内回り",),
    "新潟": ("外回り","内回り","直線"), "東京": ("外回り",),
    "中山": ("内回り","外回り"), "中京": ("外回り",),
    "京都": ("外回り","内回り"), "阪神": ("外回り","内回り"), "小倉": ("内回り",),
}

# =========================
# 内部ユーティリティ
# =========================

def _avg_track_width(geom) -> float:
    wmin = getattr(geom, "track_width_min_m", None)
    wmax = getattr(geom, "track_width_max_m", None)
    if wmin and wmax:
        return (wmin + wmax) / 2.0
    return 25.0

def _band_delta_r_from_q(q: Optional[float], width_avg: float) -> float:
    """連続帯域 q∈[0,1] → 半径補正。内0.0 / 中央0.5 / 外1.0"""
    if q is None or not math.isfinite(q):
        return 0.0
    qq = float(np.clip(q, 0.0, 1.0))
    return BAND_RADIUS_GAIN * (qq - 0.5) * width_avg

def _band_delta_r_from_label(band: Optional[str], width_avg: float) -> float:
    """離散ラベル（内/中/外）→ 半径補正（後方互換）"""
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
        return pd.Series(np.full(len(x), 0.5), index=x.index, dtype=float)
    return (x - xmin) / (xmax - xmin)

def _pick_speed_mps(row: pd.Series, distance_m: float) -> float:
    """行ごとの速度推定。final_time_sec を優先（馬ごとの予測タイム可）"""
    if 'avg_speed_mps' in row and pd.notna(row['avg_speed_mps']):
        return float(row['avg_speed_mps'])
    for k in ('final_time_sec', 'race_time_sec', 'time_sec'):
        v = row.get(k, None)
        if v and float(v) > 0:
            return _safe_div(distance_m, float(v), 16.7)
    for k in ('first3f_sec', 'last3f_sec'):
        v = row.get(k, None)
        if v and float(v) > 0:
            one_f_sec = float(v) / 3.0
            return _safe_div(200.0, one_f_sec, 16.7)
    return 16.7

def _first_turn_weight(d_first: Optional[float]) -> float:
    d = d_first if (d_first is not None and d_first > 0) else FIRST_TURN_DIST_DEFAULT
    x = (d - 200.0) / (800.0 - 200.0)
    x = max(0.0, min(1.0, x))
    return 1.0 - x

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))

def _estimate_S_range(weights=(W_CORNER, W_START, W_FINISH)):
    # 常識レンジ（安全側）
    wC, wS, wF = weights
    corner_min = (2 / 3600.0) * K_CORNER
    corner_max = (8 / 1000.0) * K_CORNER
    start_min  = (1.0 / 1000.0) * K_START
    start_max  = (1.0 / 100.0)  * K_START
    finish_min = 0.0
    finish_max = 4.0
    S_min = wC*corner_min + wS*start_min + wF*finish_min
    S_max = wC*corner_max + wS*start_max + wF*finish_max
    return S_min, S_max

# =================================================
# 幾何のフォールバック解決
# =================================================

@lru_cache(maxsize=4096)
def _resolve_geom(course_id: str, surface: str, distance_m: int, layout: str, rail: str):
    surf = "芝" if str(surface).lower().startswith(("芝","turf")) else "ダ"
    dist = int(distance_m)
    try:
        g = get_course_geom(course_id, surf, dist, layout, rail)
        if g is not None:
            return g, layout, rail
    except Exception:
        pass
    for r2 in ("A","B","C","D",""):
        if r2 == rail:
            continue
        try:
            g = get_course_geom(course_id, surf, dist, layout, r2)
            if g is not None:
                return g, layout, r2
        except Exception:
            continue
    for lay2 in LAYOUT_OPTS.get(course_id, ("内回り","外回り","直線")):
        if lay2 == layout:
            continue
        for r2 in ("A","B","C","D",""):
            try:
                g = get_course_geom(course_id, surf, dist, lay2, r2)
                if g is not None:
                    return g, lay2, r2
            except Exception:
                continue
    try:
        g = get_course_geom(course_id, surf, dist, layout, "")
        if g is not None:
            return g, layout, ""
    except Exception:
        pass
    return None, None, None

# =================================================
# 帯域（連続 & 離散）
# =================================================
def band_q_from_waku(gate_no: pd.Series, field_size: pd.Series) -> pd.Series:
    """連続帯域 q∈[0,1]。0=最内, 1=大外"""
    g = pd.to_numeric(gate_no, errors="coerce")
    f = pd.to_numeric(field_size, errors="coerce").clip(lower=2)
    q = (g - 1.0) / (f - 1.0)
    return q.clip(lower=0.0, upper=1.0)

def band_from_waku(gate_no: pd.Series, field_size: pd.Series) -> pd.Series:
    """後方互換の離散ラベル（内/中/外）"""
    q = band_q_from_waku(gate_no, field_size)
    bins = [-1e9, 1/3, 2/3, 1e9]
    lab  = ["内", "中", "外"]
    out = pd.cut(q, bins=bins, labels=lab)
    return out.astype(str)

# =========================
# raw 計算（1行）
# =========================

def _corner_load_raw_row(row: pd.Series) -> float:
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', '')
    distance  = float(row.get('distance_m', np.nan))
    if not isinstance(course_id, str) or str(surface) != '芝' or not math.isfinite(distance):
        return 0.0
    geom, lay_used, rail_used = _resolve_geom(course_id, surface, int(distance), layout, rail)
    if geom is None:
        return 0.0

    r0 = _represent_radius(course_id, lay_used or layout)
    width_avg = _avg_track_width(geom)

    # 連続帯域優先 → 無ければ離散ラベル
    band_q = row.get('band_q', None)
    if pd.notna(band_q):
        delta_r = _band_delta_r_from_q(float(band_q), width_avg)
    else:
        delta_r = _band_delta_r_from_label(row.get('band', None), width_avg)

    r = max(5.0, r0 + delta_r)
    s = _course_spiral_coeff(course_id)

    nt = row.get('num_turns', None)
    if pd.isna(nt):
        nt = getattr(geom, "num_turns", 2) or 2
    nt = int(nt)

    theta_total = (math.pi / 2.0) * nt
    v = _pick_speed_mps(row, distance)

    # v^2/r × 角度総量（比例）
    basic = max(0.0, s * (v ** 2) * theta_total / max(r, 1e-6))
    # 粗レンジ（回数/距離）も少量ブレンド
    approx = (nt / max(distance, 1.0)) * K_CORNER
    return 0.5 * basic + 0.5 * approx

def _start_cost_raw_row(row: pd.Series) -> float:
    distance  = float(row.get('distance_m', np.nan))
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', '')
    if not isinstance(course_id, str) or str(surface) != '芝' or not math.isfinite(distance):
        return 0.0

    v_c = _pick_speed_mps(row, distance)

    geom, lay_used, rail_used = _resolve_geom(course_id, surface, int(distance), layout, rail)
    d_first = getattr(geom, "start_to_first_turn_m", None) if geom else None
    w_first = _first_turn_weight(d_first)

    # 出遅れ秒・枠ロスの寄与（列が無ければ0）
    break_loss = float(row.get('break_loss_sec', 0.0) or 0.0)
    lane_pen = 0.0
    gate = row.get('gate_no', None); field = row.get('field_size', None)
    if pd.notna(gate) and pd.notna(field) and float(field) > 1:
        q = (float(gate) - 1.0) / (float(field) - 1.0)  # 0=最内,1=大外
        q = float(np.clip(q, 0.0, 1.0))
        lane_pen = 0.25 * w_first * q

    # 1角が短いほど難 → 1/d ベース + 速度寄与
    base = (1.0 / max(d_first if d_first else FIRST_TURN_DIST_DEFAULT, 1.0)) * K_START
    base += 0.02 * (v_c ** 2) * w_first
    return float(base + 0.5 * break_loss + lane_pen)

def _finish_grade_raw_row(row: pd.Series) -> float:
    distance  = float(row.get('distance_m', np.nan))
    course_id = row.get('course_id')
    surface   = row.get('surface', '芝')
    layout    = row.get('layout', '内回り')
    rail      = row.get('rail_state', '')
    if not isinstance(course_id, str) or str(surface) != '芝' or not math.isfinite(distance):
        return 0.0
    geom, _, _ = _resolve_geom(course_id, surface, int(distance), layout, rail)
    grade_pct = getattr(geom, "finish_grade_pct", None) if geom else None
    if not grade_pct:
        return 0.0
    # 今は % の絶対値を採用（上り/下りの強さとして）
    L_fin = float(row.get('finish_uphill_len_m', FINISH_UPHILL_LEN_DEFAULT))
    _ = L_fin  # 将来の距離スケーリング用に確保（現状は強度そのものを使う）
    return float(abs(grade_pct))

# =========================
# 公開: DataFrame 付与
# =========================

def add_phys_s1_features(
    df: pd.DataFrame,
    *,
    group_cols: Tuple[str, ...] = ("race_id",),
    band_col: Optional[str] = None,  # 連続なら band_q 列名/Series、離散なら band 列名を想定
    verbose: bool = False,
) -> pd.DataFrame:
    """
    追加する列:
      - phys_corner_load_raw, phys_start_cost_raw, phys_finish_grade_raw
      - phys_corner_load, phys_start_cost, phys_finish_grade  （表示用 0–1）
      - phys_s1_score_raw（raw合成）
      - phys_s1_score_z（Z）, phys_s1_score_t（偏差値）
      - phys_s1_score（最終0–1 = CDF(Z)）
    入力の帯域優先順:
      1) df['band_q'] があれば連続帯域として使用（0..1）
      2) band_col が数値列ならそれを band_q として使用
      3) band_col が文字列列なら離散ラベルとして使用
      4) gate_no × field_size から band_q & band を自動生成
    """
    df = df.copy()

    # 必須列
    required = ("course_id", "surface", "distance_m", "layout", "rail_state")
    for k in required:
        if k not in df.columns:
            raise KeyError(f"required column missing: {k}")

    # 帯域の解決
    # 既に band_q があるなら最優先
    if "band_q" not in df.columns:
        if band_col and band_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[band_col]):
                df["band_q"] = pd.to_numeric(df[band_col], errors="coerce").clip(0.0, 1.0)
            else:
                df["band"] = df[band_col]
        else:
            # gate/field から自動で両方生成（安全網）
            if {"gate_no", "field_size"}.issubset(df.columns):
                df["band_q"] = band_q_from_waku(df["gate_no"], df["field_size"])
                df["band"]   = band_from_waku(df["gate_no"], df["field_size"])
            else:
                df["band_q"] = np.nan
                df["band"]   = None

    # raw を計算
    if verbose: print("[phys_s1] computing RAW...")
    df["phys_corner_load_raw"]  = df.apply(_corner_load_raw_row,  axis=1)
    df["phys_start_cost_raw"]   = df.apply(_start_cost_raw_row,   axis=1)
    df["phys_finish_grade_raw"] = df.apply(_finish_grade_raw_row, axis=1)

    # 表示用の各0–1（従来互換）
    if verbose: print("[phys_s1] 0–1 normalize (show only)...")
    def _norm_group_show(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["phys_corner_load"]  = _minmax01(g["phys_corner_load_raw"])
        g["phys_start_cost"]   = _minmax01(g["phys_start_cost_raw"])
        g["phys_finish_grade"] = _minmax01(g["phys_finish_grade_raw"])
        w = WEIGHTS_SHOW
        g["phys_s1_score_show01"] = (
            w["corner"] * g["phys_corner_load"] +
            w["start"]  * g["phys_start_cost"] +
            w["finish"] * g["phys_finish_grade"]
        )
        return g

    if len(group_cols) == 0:
        df = _norm_group_show(df)
    else:
        df = df.groupby(list(group_cols), dropna=False, group_keys=False).apply(_norm_group_show)

    # ★ raw合成（本線）
    df["phys_s1_score_raw"] = (
        W_CORNER * df["phys_corner_load_raw"] +
        W_START  * df["phys_start_cost_raw"]  +
        W_FINISH * df["phys_finish_grade_raw"]
    )

    # === Z 正規化（フォールバック込み） ===
    if verbose: print("[phys_s1] Z-normalize with fallback...")

    # 基準集合キー（場×面×layout×距離帯400m）
    bins = list(range(800, 4200, 400))
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    df["_dist_band"] = pd.cut(pd.to_numeric(df["distance_m"], errors="coerce").astype("Int64"),
                              bins=bins, labels=labels, right=False)

    def _stats_by(keys, col):
        g = df.groupby(keys, dropna=False)[col]
        mu = g.transform("mean")
        sd = g.transform("std").fillna(0.0)
        n  = g.transform("count")
        return mu, sd, n

    keys_strict = ["course_id","surface","layout","_dist_band"]
    mu_s, sd_s, n_s = _stats_by(keys_strict, "phys_s1_score_raw")

    use_loose = (n_s < Z_N_MIN_STRICT)
    keys_loose = ["course_id","surface","layout"]
    mu_l, sd_l, n_l = _stats_by(keys_loose, "phys_s1_score_raw")
    mu = mu_s.where(~use_loose, mu_l)
    sd = sd_s.where(~use_loose, sd_l)
    n  = n_s.where(~use_loose, n_l)

    mu = mu.fillna(df["phys_s1_score_raw"].mean())
    sd = sd.fillna(df["phys_s1_score_raw"].std() or 0.0)
    n  = n.fillna(len(df))

    S_min, S_max = _estimate_S_range()
    mu_fb = (S_min + S_max) / 2.0
    sd_fb = max((S_max - S_min) / 6.0, 1e-3)

    need_fb = (n < Z_N_MIN_LOOSE) | (sd < EPS)
    Z_norm  = (df["phys_s1_score_raw"] - mu) / (sd + EPS)
    Z_fb    = (df["phys_s1_score_raw"] - mu_fb) / sd_fb
    Z_final = np.where(need_fb, Z_fb, Z_norm)

    df["phys_s1_score_z"] = Z_final
    df["phys_s1_score_t"] = 50.0 + 10.0 * df["phys_s1_score_z"]
    df["phys_s1_score"]   = pd.Series(Z_final, index=df.index).map(_norm_cdf)  # 最終: 0–1（CDF）

    # 後片付け
    df = df.drop(columns=["_dist_band"], errors="ignore")

    if verbose:
        print("[phys_s1] done.")
    return df


# =========================
# 使い方（参考）
# =========================
"""
from course_geometry import register_all_turf
from physics_sprint1 import add_phys_s1_features

register_all_turf()

df_in = pd.DataFrame({
    "race_id": "R1",
    "horse":  ["A","B","C","D"],
    "course_id": ["京都"]*4,
    "surface": ["芝"]*4,
    "distance_m": [3000]*4,
    "layout": ["外回り"]*4,
    "rail_state": ["A"]*4,
    "gate_no": [1,4,8,12],
    "field_size": [12]*4,
    "final_time_sec": [178.5, 179.2, 180.1, 181.0],  # 馬ごと速度OK
})
df_out = add_phys_s1_features(df_in, group_cols=("race_id",), band_col=None, verbose=True)
"""
