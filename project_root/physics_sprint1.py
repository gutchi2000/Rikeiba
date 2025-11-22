# -*- coding: utf-8 -*-
"""
physics_sprint1.py (revised physics-heavy version)

Sprint 1 の物理スコア（コーナー v^2/r ・スタート加速 ・終盤勾配）。

公開関数:
- add_phys_s1_features(df, *, group_cols=("race_id",), band_col=None, verbose=False) -> pd.DataFrame

思想:
- できるだけ「馬にかかる力 / 仕事 (J/kg 相当)」をベースに raw 値を作り、
  正規化・偏差値化はこのモジュール内で行う（既存 API 互換）。
- 3 本柱:
    1) Corner: 横方向加速度 a_c = v^2 / r による「横 G」の二乗 × コーナー滞在時間
       → corner_dose_g2s [g^2 * s] を計算してスケーリング
    2) Start: スタート〜1角までに必要な加速 a ≒ v^2 / (2 d_first) と、
       出遅れ秒 / 外枠ロスをエネルギーコスト的に加算
    3) Finish: 終盤の勾配による位置エネルギー変化 ΔE/m ≒ g * slope * L_fin [J/kg]

- 最終スコアは
    「raw 合成 S_raw → Z 正規化 → CDF(0–1) = phys_s1_score」
  で従来版と同じ形を維持する。
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from course_geometry import get_course_geom  # コース幾何

# =========================
# 物理定数・係数
# =========================

# 重力加速度 [m/s^2]
G0: float = 9.80665

# スパイラルカーブ低減係数（0.0=負荷ゼロ, 1.0=低減なし）
SPIRAL_COEFF: Dict[str, float] = {
    "札幌": 1.0, "函館": 0.8, "福島": 0.75, "中山": 0.9, "中京": 0.8,
    "京都": 0.8, "阪神": 0.8, "小倉": 0.8, "東京": 0.95, "新潟": 0.95,
}

# 代表コーナー半径 r0 [m]
RADIUS_R0: Dict[Tuple[str, str], float] = {
    ("札幌", "内回り"): 105.0, ("函館", "内回り"): 105.0, ("福島", "内回り"): 100.0,
    ("中山", "内回り"): 115.0, ("中山", "外回り"): 120.0, ("東京", "外回り"): 145.0,
    ("新潟", "直線"): 1e9, ("新潟", "内回り"): 120.0, ("新潟", "外回り"): 140.0,
    ("中京", "外回り"): 130.0, ("京都", "内回り"): 125.0, ("京都", "外回り"): 140.0,
    ("阪神", "内回り"): 120.0, ("阪神", "外回り"): 140.0, ("小倉", "内回り"): 105.0,
}

# show 用 0–1 合成の重み（互換用）
WEIGHTS_SHOW = dict(corner=0.45, start=0.35, finish=0.20)

# raw 合成の重み（ここをチューニング対象にしても OK）
W_CORNER = 1.0   # corner_dose
W_START  = 1.0   # start_cost
W_FINISH = 1.0   # finish_cost

# スケール係数（単位をそろえつつ数十〜数百レンジに収める）
# Corner: g^2*s オーダー → ×100 で 100 前後
K_CORNER_DOSE = 100.0
# Start: 0.5 v^2 [J/kg] オーダー → ×1.0 で 100 台
K_START_ENERGY = 1.0
# Finish: g * slope * L_fin [J/kg] オーダー → ×2.0 で 10〜50
K_FINISH_ENERGY = 2.0

# 出遅れ秒ペナルティ & 枠番スタートペナルティ
# 「一定のパワーをロス時間だけ余計に出す」というイメージで線形に加算
K_BREAK_ENERGY_PER_SEC = 30.0   # [J/kg / sec] 相当の係数
K_LANE_EXTRA_PER_Q     = 40.0   # 大外 (q=1) で ~40 J/kg くらいを想定

# Z基準フォールバック
Z_N_MIN_STRICT = 30
Z_N_MIN_LOOSE  = 5
EPS = 1e-6

# 幾何フォールバック
FIRST_TURN_DIST_DEFAULT = 800.0      # [m]
FINISH_UPHILL_LEN_DEFAULT = 200.0    # [m]

# 連続帯域の半径補正ゲイン（幅員×この係数×(q-0.5)）
BAND_RADIUS_GAIN = 1.0

# コーナー速度の平均速度に対する係数
CORNER_V_RATIO_DEFAULT = 0.94

# レイアウト候補
LAYOUT_OPTS: Dict[str, Tuple[str, ...]] = {
    "札幌": ("内回り",), "函館": ("内回り",), "福島": ("内回り",),
    "新潟": ("外回り", "内回り", "直線"), "東京": ("外回り",),
    "中山": ("内回り", "外回り"), "中京": ("外回り",),
    "京都": ("外回り", "内回り"), "阪神": ("外回り", "内回り"), "小倉": ("内回り",),
}


# =========================
# 内部ユーティリティ
# =========================

def _avg_track_width(geom) -> float:
    wmin = getattr(geom, "track_width_min_m", None)
    wmax = getattr(geom, "track_width_max_m", None)
    if wmin is not None and wmax is not None:
        return float(wmin + wmax) / 2.0
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
    return SPIRAL_COEFF.get(course_id, 0.9)


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
    """
    行ごとの平均速度 [m/s] を推定。
    final_time_sec を優先（馬ごとの予測タイムでも OK）。
    """
    if "avg_speed_mps" in row and pd.notna(row["avg_speed_mps"]):
        return float(row["avg_speed_mps"])

    for k in ("final_time_sec", "race_time_sec", "time_sec"):
        v = row.get(k, None)
        if v and float(v) > 0:
            return _safe_div(distance_m, float(v), 16.7)

    for k in ("first3f_sec", "last3f_sec"):
        v = row.get(k, None)
        if v and float(v) > 0:
            one_f_sec = float(v) / 3.0
            return _safe_div(200.0, one_f_sec, 16.7)

    # fallback: 60 km/h
    return 16.7


def _first_turn_weight(d_first: Optional[float]) -> float:
    """
    1角までの距離に応じた「スタート重要度」重み。
    200m → 1.0 （すぐコーナーで難しい）
    800m → 0.0 （充分余裕）
    線形で補間。
    """
    d = d_first if (d_first is not None and d_first > 0) else FIRST_TURN_DIST_DEFAULT
    x = (d - 200.0) / (800.0 - 200.0)
    x = max(0.0, min(1.0, x))
    return 1.0 - x


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _estimate_S_range(weights=(W_CORNER, W_START, W_FINISH)):
    """
    raw 合成値の概ねのレンジを安全側に見積もる。
    Corner dose ≈ [0, 2], Start ≈ [50, 250], Finish ≈ [0, 60] 程度を想定。
    """
    wC, wS, wF = weights
    corner_min, corner_max = 0.0, 200.0
    start_min,  start_max  = 50.0, 250.0
    finish_min, finish_max = 0.0, 80.0
    S_min = wC * corner_min + wS * start_min + wF * finish_min
    S_max = wC * corner_max + wS * start_max  + wF * finish_max
    return S_min, S_max


# =================================================
# 幾何のフォールバック解決
# =================================================

@lru_cache(maxsize=4096)
def _resolve_geom(course_id: str, surface: str, distance_m: int, layout: str, rail: str):
    """
    course_geometry 側で幾何情報を引いて、見つからない場合は layout / rail を
    緩めながらフォールバックする。
    """
    surf = "芝" if str(surface).lower().startswith(("芝", "turf")) else "ダ"
    dist = int(distance_m)

    try:
        g = get_course_geom(course_id, surf, dist, layout, rail)
        if g is not None:
            return g, layout, rail
    except Exception:
        pass

    # rail を変えて探す
    for r2 in ("A", "B", "C", "D", ""):
        if r2 == rail:
            continue
        try:
            g = get_course_geom(course_id, surf, dist, layout, r2)
            if g is not None:
                return g, layout, r2
        except Exception:
            continue

    # layout を変えて探す
    for lay2 in LAYOUT_OPTS.get(course_id, ("内回り", "外回り", "直線")):
        if lay2 == layout:
            continue
        for r2 in ("A", "B", "C", "D", ""):
            try:
                g = get_course_geom(course_id, surf, dist, lay2, r2)
                if g is not None:
                    return g, lay2, r2
            except Exception:
                continue

    # rail 指定なしでもう一度
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
    """
    Corner: 横方向加速度（横 G）の「線量」っぽい指標を計算。

    - コーナー有効半径 r_eff を gate/band から補正
    - コーナー滞在時間 t_curve ≒ 弧長 / v_corner
    - 横 G: a_c / g = v_corner^2 / (r_eff * g)
    - corner_dose_g2s = (横G)^2 * t_curve * スパイラル補正
    """
    course_id = row.get("course_id")
    surface   = row.get("surface", "芝")
    layout    = row.get("layout", "内回り")
    rail      = row.get("rail_state", "")
    distance  = float(row.get("distance_m", np.nan))

    if not isinstance(course_id, str) or str(surface) != "芝" or not math.isfinite(distance):
        return 0.0

    geom, lay_used, rail_used = _resolve_geom(course_id, surface, int(distance), layout, rail)
    if geom is None:
        return 0.0

    layout_used = lay_used or layout

    width_avg = _avg_track_width(geom)

    # 基本半径
    r_base = getattr(geom, "corner_radius_m", None)
    if r_base is None or not math.isfinite(r_base):
        r_base = _represent_radius(course_id, layout_used)

    # 帯域による半径補正
    band_q = row.get("band_q", None)
    if pd.notna(band_q):
        delta_r = _band_delta_r_from_q(float(band_q), width_avg)
    else:
        delta_r = _band_delta_r_from_label(row.get("band", None), width_avg)

    r_eff = max(5.0, float(r_base) + float(delta_r))

    # コーナーの総旋回角（ラジアン）
    nt = row.get("num_turns", None)
    if pd.isna(nt):
        nt = getattr(geom, "num_turns", 2) or 2
    nt = int(nt)
    theta_total = (math.pi / 2.0) * nt  # 90°コーナーが nt 回あるイメージ

    # 曲線弧長・滞在時間
    v_mean = _pick_speed_mps(row, distance)
    v_corner = v_mean * getattr(geom, "corner_speed_ratio", CORNER_V_RATIO_DEFAULT)
    arc_len = r_eff * theta_total
    t_curve = _safe_div(arc_len, max(v_corner, 1e-3), default=0.0)  # [s]

    # 横 G
    g_force = (v_corner ** 2) / max(r_eff * G0, 1e-6)  # a_c / g
    # 安全クリップ（1.5 g くらいまで）
    g_force = max(0.0, min(g_force, 1.5))

    # 総距離に対する「コーナー区間比」
    frac_curve = _safe_div(arc_len, distance, default=0.0)
    frac_curve = max(0.0, min(frac_curve, 0.8))

    s_spiral = _course_spiral_coeff(course_id)

    # 横G^2 × コーナー滞在時間 × 区間比 × スパイラル補正
    dose_g2s = (g_force ** 2) * t_curve * frac_curve * s_spiral

    corner_raw = K_CORNER_DOSE * dose_g2s
    return float(corner_raw)


def _start_cost_raw_row(row: pd.Series) -> float:
    """
    Start: スタート〜1角の加速負荷と「ロス」をエネルギーっぽく評価。

    1) 0→v_mean までの速度立ち上げに必要な運動エネルギー 0.5 * v^2 [J/kg]
    2) 1角までの距離が短いほど「急加速」が要求されるので、
       a_req ≒ v^2 / (2 * d_first) を使って重み付け
    3) 出遅れ秒は「そのぶん余分に全力で走る時間」とみなして線形ペナルティ
    4) 外枠は初動で余分な距離を走るイメージで q∈[0,1] をエネルギー加算
    """
    distance  = float(row.get("distance_m", np.nan))
    course_id = row.get("course_id")
    surface   = row.get("surface", "芝")
    layout    = row.get("layout", "内回り")
    rail      = row.get("rail_state", "")

    if not isinstance(course_id, str) or str(surface) != "芝" or not math.isfinite(distance):
        return 0.0

    v_mean = _pick_speed_mps(row, distance)

    geom, lay_used, rail_used = _resolve_geom(course_id, surface, int(distance), layout, rail)
    d_first = getattr(geom, "start_to_first_turn_m", None) if geom else None
    if d_first is None or not math.isfinite(d_first) or d_first <= 0:
        d_first = FIRST_TURN_DIST_DEFAULT
    d_first = float(d_first)

    w_first = _first_turn_weight(d_first)

    # 0→v の運動エネルギー（質量で割った値） [J/kg]
    ke_spec = 0.5 * (v_mean ** 2)

    # 必要加速度 a_req ≒ v^2 / (2 d_first) [m/s^2]
    a_req = (v_mean ** 2) / max(2.0 * d_first, 1.0)
    a_req_g = a_req / G0

    # 「急加速度」による負荷を係数として乗せる
    base = K_START_ENERGY * ke_spec * (1.0 + 0.3 * a_req_g * w_first)

    # 出遅れ秒による追加コスト
    break_loss = float(row.get("break_loss_sec", 0.0) or 0.0)
    break_loss = max(0.0, break_loss)
    break_pen = K_BREAK_ENERGY_PER_SEC * break_loss

    # 外枠ロスによる追加コスト
    lane_pen = 0.0
    gate = row.get("gate_no", None)
    field = row.get("field_size", None)
    if pd.notna(gate) and pd.notna(field) and float(field) > 1:
        q = (float(gate) - 1.0) / (float(field) - 1.0)  # 0=最内,1=大外
        q = float(np.clip(q, 0.0, 1.0))
        lane_pen = K_LANE_EXTRA_PER_Q * q * w_first

    start_raw = base + break_pen + lane_pen
    return float(start_raw)


def _finish_grade_raw_row(row: pd.Series) -> float:
    """
    Finish: 終盤勾配による位置エネルギー変化を J/kg ベースで評価。

    ΔE/m ≒ g * slope * L_fin
      - slope: 勾配[%] / 100 （上り/下りとも絶対値）
      - L_fin: 終盤の勾配区間長 [m]
    下り坂も「脚への負荷」としてみなすため絶対値を使用。
    """
    distance  = float(row.get("distance_m", np.nan))
    course_id = row.get("course_id")
    surface   = row.get("surface", "芝")
    layout    = row.get("layout", "内回り")
    rail      = row.get("rail_state", "")

    if not isinstance(course_id, str) or str(surface) != "芝" or not math.isfinite(distance):
        return 0.0

    geom, _, _ = _resolve_geom(course_id, surface, int(distance), layout, rail)
    grade_pct = getattr(geom, "finish_grade_pct", None) if geom else None
    if grade_pct is None or not math.isfinite(grade_pct) or float(grade_pct) == 0.0:
        return 0.0

    grade_pct = float(grade_pct)
    slope = abs(grade_pct) / 100.0  # 無次元

    L_fin = float(row.get("finish_uphill_len_m", FINISH_UPHILL_LEN_DEFAULT))
    if not math.isfinite(L_fin) or L_fin <= 0:
        L_fin = FINISH_UPHILL_LEN_DEFAULT
    L_fin = min(L_fin, distance)

    # 勾配区間比
    frac = _safe_div(L_fin, distance, default=0.0)
    frac = max(0.0, min(frac, 1.0))

    # ΔE/m ≒ g * slope * L_fin × 区間比 （位置エネルギーのオーダー）
    energy_per_kg = G0 * slope * L_fin * frac  # [J/kg]

    finish_raw = K_FINISH_ENERGY * energy_per_kg
    return float(finish_raw)


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
    追加する列（従来 API 互換）:
      - phys_corner_load_raw, phys_start_cost_raw, phys_finish_grade_raw
      - phys_corner_load, phys_start_cost, phys_finish_grade  （表示用 0–1）
      - phys_s1_score_raw（raw 合成）
      - phys_s1_score_z（Z）, phys_s1_score_t（偏差値）
      - phys_s1_score（最終 0–1 = CDF(Z)）

    入力の帯域優先順:
      1) df["band_q"] があれば連続帯域として使用（0..1）
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
    if "band_q" not in df.columns:
        if band_col and band_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[band_col]):
                df["band_q"] = pd.to_numeric(df[band_col], errors="coerce").clip(0.0, 1.0)
            else:
                df["band"] = df[band_col]
        else:
            # gate/field から自動で両方生成
            if {"gate_no", "field_size"}.issubset(df.columns):
                df["band_q"] = band_q_from_waku(df["gate_no"], df["field_size"])
                df["band"]   = band_from_waku(df["gate_no"], df["field_size"])
            else:
                df["band_q"] = np.nan
                df["band"]   = None

    # raw を計算
    if verbose:
        print("[phys_s1] computing RAW (physics-based)...")
    df["phys_corner_load_raw"]  = df.apply(_corner_load_raw_row,  axis=1)
    df["phys_start_cost_raw"]   = df.apply(_start_cost_raw_row,   axis=1)
    df["phys_finish_grade_raw"] = df.apply(_finish_grade_raw_row, axis=1)

    # 表示用の各 0–1（従来互換・レース単位 Min–Max）
    if verbose:
        print("[phys_s1] 0–1 normalize (show only)...")

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

    # ★ raw 合成（本線）
    df["phys_s1_score_raw"] = (
        W_CORNER * df["phys_corner_load_raw"] +
        W_START  * df["phys_start_cost_raw"]  +
        W_FINISH * df["phys_finish_grade_raw"]
    )

    # === Z 正規化（フォールバック込み） ===
    if verbose:
        print("[phys_s1] Z-normalize with fallback...")

    # 基準集合キー（場×面×layout×距離帯 400m）
    bins = list(range(800, 4200, 400))
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["_dist_band"] = pd.cut(
        pd.to_numeric(df["distance_m"], errors="coerce").astype("Int64"),
        bins=bins, labels=labels, right=False
    )

    def _stats_by(keys, col):
        g = df.groupby(keys, dropna=False)[col]
        mu = g.transform("mean")
        sd = g.transform("std").fillna(0.0)
        n  = g.transform("count")
        return mu, sd, n

    keys_strict = ["course_id", "surface", "layout", "_dist_band"]
    mu_s, sd_s, n_s = _stats_by(keys_strict, "phys_s1_score_raw")

    use_loose = (n_s < Z_N_MIN_STRICT)
    keys_loose = ["course_id", "surface", "layout"]
    mu_l, sd_l, n_l = _stats_by(keys_loose, "phys_s1_score_raw")
    mu = mu_s.where(~use_loose, mu_l)
    sd = sd_s.where(~use_loose, sd_l)
    n  = n_s.where(~use_loose, n_l)

    # 全体平均・分散
    global_mu = df["phys_s1_score_raw"].mean()
    global_sd = df["phys_s1_score_raw"].std() or 0.0
    mu = mu.fillna(global_mu)
    sd = sd.fillna(global_sd)
    n  = n.fillna(len(df))

    # さらにデータが極端に少ない時の理論レンジフォールバック
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
