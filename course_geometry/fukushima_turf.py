# -*- coding: utf-8 -*-
# course_geometry/fukushima_turf.py
# 福島競馬場（芝）— 右回り・A/B/C対応 完成版
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "右"
TYPICAL_LANE_BIAS = (
    "小回り＋スパイラル。直線292〜300mで立ち回り重視。"
    "開催が進むと外差しが決まりやすい。"
)

# ── 仮柵ごとの一周距離・幅員・直線長（JRA表）
RAILS = [  # (rail_state, rail_offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0, 1600.0, 25, 27, 292.0),
    ("B", 3, 1614.1, 22.5, 25.0, 297.5),
    ("C", 6, 1628.1, 20, 23, 299.7),
]
# 高低差メモ：芝 1.9m
# ゴール前の上り：残170→50mで +1.2m ≒ 勾配 ~1.0% を代表値に採用
FINISH_GRADE_PCT = 1.0

# ── 距離別：スタート〜最初コーナー（緑パネル数値）
DISTANCES = [
    # (distance_m, start_to_first_turn_m, num_turns, notes)
    (1000, 211.7, 2, "向正面2角過ぎ→3角まで約212m（短）"),
    (1200, 411.7, 2, "向正面2角奥ポケット→3角まで約412m（長め）"),
    (1700, 205.3, 2, "スタンド前半ば→1角まで約205m（短）"),
    (1800, 305.3, 2, "スタンド前4角寄り→1角まで約305m"),
    (2000, 505.3, 2, "スタンド前4角奥ポケット→1角まで約505m（長い）"),
    (2600, 262.5, 2, "向正面2角過ぎ→最初コーナーまで約263m"),
]

def register():
    for dist, first, turns, note in DISTANCES:
        for rs, off, lap, wmin, wmax, straight in RAILS:
            _add(CourseGeometry(
                course_id="福島", surface="芝", layout="内回り",  # 福島は単一レイアウト扱い
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,      # 必要なら高低図から追記可
                finish_grade_pct=FINISH_GRADE_PCT,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
