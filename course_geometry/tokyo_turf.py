# -*- coding: utf-8 -*-
# course_geometry/tokyo_turf.py
# 東京競馬場（芝コース）— 外回り・A〜D対応／スタート～最初コーナー距離入り
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

# 共通
DIRECTION = "左"
TYPICAL_LANE_BIAS = "長い直線で瞬発力要求。開催により内外の振れ（春～初夏フラット寄り）。"

# 仮柵：一周距離・幅員・直線長（JRA公表値）
RAILS = [  # (rail_state, offset(m), lap_len(m), width_min, width_max, straight_len(m))
    ("A", 0,  2083.1, 31, 41, 525.9),
    ("B", 3,  2101.9, 28, 38, 525.9),
    ("C", 6,  2120.8, 25, 35, 525.9),
    ("D", 9,  2139.6, 22, 32, 525.9),
]

# 距離ごとの「スタート～最初のコーナー」距離（画像の数値）
# 注）“3コーナーまで”と表記のあるものは、最初に到達するコーナーが3角という意味なので
#     その距離を start_to_first_turn_m にそのまま入れている。
DISTANCES = [
    # (distance_m, start_to_first_turn_m, num_turns, notes)
    (1400, 366.3, 2, "向こう正面半ばスタート→最初は3角まで約366m"),
    (1600, 566.3, 2, "直線入口付近スタート→3角まで約566m（マイル）"),
    (1800, 187.8, 2, "向こう正面2角奥ポケット→コーナーまで約188m"),
    (2000, 102.7, 2, "1角奥ポケット→2角まで約103m"),
    (2200, 193.5, 2, "正面スタンド前→1角まで約194m"),
    (2300, 293.5, 2, "正面スタンド前→1角まで約294m"),
    (2400, 393.5, 2, "正面スタンド前→1角まで約394m（ダービー/オークス系距離）"),
    (2500, 493.5, 2, "正面スタンド前坂下→1角まで約494m"),
    (3200,  94.0, 4, "向こう正面半ば→最初のコーナーまで約94m（周回）"),
]

def register():
    for dist, first, turns, note in DISTANCES:
        for rs, off, lap, wmin, wmax, straight in RAILS:
            _add(CourseGeometry(
                course_id="東京", surface="芝", layout="外回り",
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,   # 高低図から追記可
                finish_grade_pct=None,            # 同上
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
