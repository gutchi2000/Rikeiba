# -*- coding: utf-8 -*-
# course_geometry/sapporo_turf.py
# 札幌競馬場（芝）— 右回り・A/B/C対応 完成版
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "右"
TYPICAL_LANE_BIAS = (
    "ほぼ平坦・円形に近い小回り。直線短く立ち回り＆先行有利になりやすい。"
    "捲りが決まりやすい開催も。オール洋芝で時計はやや掛かりがち。"
)

# ── 仮柵別 一周距離・幅員・直線長（JRA表）
RAILS = [  # (rail_state, rail_offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0, 1640.9, 25, 27, 266.1),
    ("B", 3, 1650.4, 22, 25.5, 267.6),
    ("C", 6, 1659.8, 19, 25, 269.1),
]
# 高低差メモ：0.7m（平坦）。ゴール前の代表勾配は特記なし → finish_grade_pct=None のまま。

# ── 距離別：スタート〜最初コーナーまで（解説パネル数値）
DISTANCES = [
    # (distance_m, start_to_first_turn_m, num_turns, notes)
    (1000, 205.5, 2, "向正面半ば→3角まで約206m（短）"),
    (1200, 405.5, 2, "向正面2角奥P→3角まで約406m（長め）"),
    (1500, 170.0, 2, "1角寄りP→2角まで約170m（短）"),
    (1800, 185.1, 2, "スタンド前やや4角寄り→1角まで約185m（短）"),
    (2000, 325.3, 2, "スタンド前4角奥P→1角まで約325m"),
    (2600, 164.6, 2, "向正面半ば→最初のコーナーまで約165m（短）"),
]

def register():
    for dist, first, turns, note in DISTANCES:
        for rs, off, lap, wmin, wmax, straight in RAILS:
            _add(CourseGeometry(
                course_id="札幌", surface="芝", layout="内回り",  # 札幌は単一レイアウト扱い
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,  # 平坦のため未設定（必要なら後で追記）
                finish_grade_pct=None,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
