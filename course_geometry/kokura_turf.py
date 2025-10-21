# -*- coding: utf-8 -*-
# course_geometry/kokura_turf.py
# 小倉競馬場（芝）— 右回り・A/B/C対応 完成版
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "右"
TYPICAL_LANE_BIAS = (
    "平坦＋小回り。2角まで緩やかに上り→向正面と3～4角で下り。"
    "直線は293mで先行優勢になりやすいが、中長距離はまくりが決まることも。"
)

# ── 仮柵別 一周距離・幅員・直線長（JRA表）
RAILS = [  # (rail_state, rail_offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0, 1615.1, 30, 30, 293.0),
    ("B", 3, 1633.9, 27, 27, 293.0),
    ("C", 6, 1652.8, 24, 24, 293.0),
]
# 高低差：3.0m。直線は平坦 → finish_grade_pct=0.0 を採用。

# ── 距離別：スタート〜最初コーナー（緑パネル数値）
DISTANCES = [
    # (distance_m, start_to_first_turn_m, num_turns, notes)
    (1000, 279.0, 2, "向正面半ば→3角まで約279m（短）"),
    (1200, 479.0, 2, "向正面2角奥ポケット→3角まで約479m（長め）"),
    (1700, 171.5, 2, "スタンド前半ば→1角まで約171.5m（短）"),
    (1800, 271.5, 2, "スタンド前半ば→1角まで約271.5m"),
    (2000, 505.3, 2, "スタンド前4角奥ポケット→1角まで約505.3m（長い）"),
    (2600, 243.5, 2, "向正面半ば→最初のコーナーまで約243.5m"),
]

def register():
    for dist, first, turns, note in DISTANCES:
        for rs, off, lap, wmin, wmax, straight in RAILS:
            _add(CourseGeometry(
                course_id="小倉", surface="芝", layout="内回り",  # 小倉は単一レイアウト扱い
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,   # 必要なら高低図から追記
                finish_grade_pct=0.0,            # 直線は平坦
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
