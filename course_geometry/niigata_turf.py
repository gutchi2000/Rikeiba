# -*- coding: utf-8 -*-
# course_geometry/niigata_turf.py
# 新潟競馬場（芝）— 直線/内回り/外回り、A/B対応 完成版
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "左"
TYPICAL_LANE_BIAS = "直千は外ラチ優勢が出やすい。外回りは直線長く差し届く開催も。"

# ── 仮柵（A=オフセット0, B=+4m相当） 一周距離・幅員・直線長（JRA表）
# 直線長は内回り=358.7m / 外回り=658.7m（A/B共通）
RAILS_INNER = [  # (rail_state, offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0, 1623.0, 25, 25, 358.7),
    ("B", 4, 1648.1, 21, 21, 358.7),
]
RAILS_OUTER = [
    ("A", 0, 2223.0, 25, 25, 658.7),
    ("B", 4, 2248.1, 21, 21, 658.7),
]

# ── 直線1000m（コーナー無し）
def _register_straight():
    _add(CourseGeometry(
        course_id="新潟", surface="芝", layout="直線", direction=DIRECTION,
        distance_m=1000,
        straight_length_m=1000.0, start_to_first_turn_m=None, num_turns=0,
        elevation_gain_last600_m=0.0, finish_grade_pct=0.0,  # 直千はフラット扱い（後で更新可）
        rail_state="A", rail_offset_m=0,
        typical_lane_bias="外ラチ優位が出やすい（開催依存）",
        notes="直線1000m。帯域=枠寄りで推定",
    ))

# ── 内回り：JRA発走距離（表より）
DIST_INNER = [
    (1200, "内回り：3角まで長め、直線は358.7m"),
    (1400, "内回り：テン速くなりやすい"),
    (2000, "内回り：2角ポケット→向正面長い"),
    (2200, "内回り：最初の直線長め"),
    (2400, "内回り"),
]

# ── 外回り：JRA発走距離（表より）
DIST_OUTER = [
    (1400, "外回り：直線658.7mで末脚問われる"),
    (1600, "外回り：緩やかに3角へ→直線長い"),
    (1800, "外回り"),
    (2000, "外回り：ニイガタ記念など"),
    (3000, "外回り：周回（num_turns=4）"),
    (3200, "外回り：周回（num_turns=4）"),
]

def _register_inner():
    for dist, note in DIST_INNER:
        for rs, off, lap, wmin, wmax, straight in RAILS_INNER:
            _add(CourseGeometry(
                course_id="新潟", surface="芝", layout="内回り", direction=DIRECTION,
                distance_m=dist,
                straight_length_m=straight,
                start_to_first_turn_m=None,           # 公表数値見つかり次第更新
                num_turns=2,
                elevation_gain_last600_m=None, finish_grade_pct=None,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))

def _register_outer():
    for dist, note in DIST_OUTER:
        turns = 4 if dist in (3000, 3200) else 2
        for rs, off, lap, wmin, wmax, straight in RAILS_OUTER:
            _add(CourseGeometry(
                course_id="新潟", surface="芝", layout="外回り", direction=DIRECTION,
                distance_m=dist,
                straight_length_m=straight,
                start_to_first_turn_m=None,           # 公表数値見つかり次第更新
                num_turns=turns,
                elevation_gain_last600_m=None, finish_grade_pct=None,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))

def register():
    _register_straight()
    _register_inner()
    _register_outer()
