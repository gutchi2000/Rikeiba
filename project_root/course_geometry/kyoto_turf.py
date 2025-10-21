# -*- coding: utf-8 -*-
# course_geometry/kyoto_turf.py
# 京都競馬場（芝のみ）— A/B/C/D・内回り/外回りの幾何データ
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

# 方向は右回り
DIRECTION = "右"

# 仮柵別の一周距離・幅員・直線長（m）
# 画像の「芝コースデータ」および断面図より
RAILS_INNER = [  # (rail_state, rail_offset_m, lap_length_m, width_min, width_max, straight_len_m)
    ("A", 0, 1782.8, 27, 38, 328.4),
    ("B", 3, 1802.2, 24, 35, 323.4),
    ("C", 6, 1821.1, 21, 32, 323.4),
    ("D", 9, 1839.9, 18, 29, 323.4),
]
RAILS_OUTER = [
    ("A", 0, 1894.3, 24, 38, 403.7),
    ("B", 3, 1913.6, 21, 35, 398.7),
    ("C", 6, 1932.4, 18, 32, 398.7),
    ("D", 9, 1951.3, 15, 29, 398.7),
]

# 備考：高低差（メモ） 内回り=3.1m / 外回り=4.3m
# finish_grade_pct / elevation_gain_last600_m は不明のため後追いで更新可能

# 内回りの実施距離（JRA発走距離）
# --- 内回りコース ---
DIST_INNER = [
    (1200, 316.2, 2, "向正面半ば発走（1角まで約316m）"),
    (1400, 516.2, 2, "2角奥ポケット発走（3角まで約516m）"),
    (1600, 516.8, 2, "2角奥ポケット発走（3角まで約516m）"),
    (2000, 308.3, 2, "スタンド前発走（1角まで約308m）"),
    (2200, 397.7, 2, "スタンド前発走（1角まで約398m）"),
]

# --- 外回りコース ---
DIST_OUTER = [
    (1400, 511.7, 2, "2角出口付近発走（3角まで約512m）"),
    (1600, 711.7, 2, "直線坂下発走（3角まで約712m）"),
    (1800, 911.7, 2, "直線奥ポケット発走（3角まで約912m）"),
    (2400, 597.7, 2, "4角奥ポケット発走（1角まで約598m）"),
    (3000, 217.4, 4, "向正面中ほど発走（1角まで約217m・周回）"),
    (3200, 412.7, 4, "向正面半ば発走（1角まで約413m・周回）"),
]


# 典型バイアスのメモ（任意記述）
TYPICAL_LANE_BIAS = "開催週により内有利が出やすい。長距離は持久寄り。"

def register():
    # 内回り
    for dist, first, turns, note in DIST_INNER:
        for rs, off, lap, wmin, wmax, straight_len in RAILS_INNER:
            _add(CourseGeometry(
                course_id="京都", surface="芝", layout="内回り",
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight_len,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,    # 後で高低図から追記可
                finish_grade_pct=None,            # 同上
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))

    # 外回り
    for dist, first, turns, note in DIST_OUTER:
        for rs, off, lap, wmin, wmax, straight_len in RAILS_OUTER:
            _add(CourseGeometry(
                course_id="京都", surface="芝", layout="外回り",
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight_len,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,
                finish_grade_pct=None,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
