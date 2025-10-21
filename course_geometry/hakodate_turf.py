# -*- coding: utf-8 -*-
# course_geometry/hakodate_turf.py
# 函館競馬場（芝）— 右回り・A/B/C対応 完成版
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "右"
TYPICAL_LANE_BIAS = (
    "直線最短クラス（262m前後）で先行/内の利が出やすい。"
    "3～4角はスパイラル。起伏は高低差3.5mでタフ。洋芝で後半ほど時計掛かりやすい。"
)

# ── 仮柵別 一周距離・幅員・直線長（JRA表）
RAILS = [  # (rail_state, rail_offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0, 1626.6, 29,     29,    262.1),
    ("B", 3, 1651.8, 25,     25,    262.1),
    ("C", 6, 1675.8, 21,     22,    264.5),
]
# 高低差メモ：3.5m（芝）。finish_grade_pct は代表値不明のため未設定。

# ── 距離別：スタート〜最初コーナー（画像パネル数値より）
DISTANCES = [
    # (distance_m, start_to_first_turn_m, num_turns, notes)
    (1000, 289.1, 2, "向正面半ば→3角まで約289m（短）"),
    (1200, 489.1, 2, "向正面2角奥ポケット→3角まで約489m（長め・上り発走）"),
    (1700, None,  2, "向正面寄り発走（数値未確認）"),
    (1800, 275.8, 2, "スタンド前4角寄り→1角まで約276m（短）"),
    (2000, 475.8, 2, "スタンド前4角奥ポケット→1角まで約476m（長め）"),
    (2600, 262.5, 2, "向正面半ば→最初のコーナーまで約263m（上り2回踏む設定）"),
]

def register():
    for dist, first, turns, note in DISTANCES:
        for rs, off, lap, wmin, wmax, straight in RAILS:
            _add(CourseGeometry(
                course_id="函館", surface="芝", layout="内回り",  # 函館は単一レイアウト扱い
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,
                finish_grade_pct=None,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
