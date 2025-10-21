# -*- coding: utf-8 -*-
# course_geometry/nakayama_turf.py
# 中山競馬場（芝）— 内/外 × A/B/C、距離別の1角距離を反映
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "右"
TYPICAL_LANE_BIAS = "高低差5.3m、ゴール前急坂。先行・内有利が出やすい開催多め。"

# ── 仮柵別 一周距離・幅員・直線長（JRA表より）
# 直線は内外とも 310m
RAILS_INNER = [  # (rail_state, rail_offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0, 1667.1, 20, 32, 310.0),
    ("B", 3, 1686.0, 17, 29, 310.0),
    ("C", 6, 1704.8, 14, 26, 310.0),
]
RAILS_OUTER = [
    ("A", 0, 1839.7, 24, 32, 310.0),
    ("B", 3, 1858.5, 21, 29, 310.0),
    ("C", 6, 1877.3, 18, 26, 310.0),
]

# 高低差メモ：芝 5.3m（参考）。finish_grade_pct / last600 は後で追記可。

# ── 内回り距離（画像テキストで1角距離が読めたものは数値確定）
DIST_INNER = [
    # (distance_m, start_to_first_turn_m, num_turns, note)
    (1800, 204.9, 2, "スタンド前・急坂手前スタート→1角まで約205m"),
    (2000, 404.9, 2, "4角直線入口スタート→1角まで約405m"),
    (2200, None, 2, "2000mに近い位置からの発走。内回り使用（数値保留）"),
    (3200, None, 4, "内回り周回（数値未確認）"),
    (3600, 337.7, 4, "スタンド前発走→1角まで約338m（内回り2周）"),
]

# ── 外回り距離
DIST_OUTER = [
    (1200, 275.1, 2, "向正面・外回り2角過ぎ→3角まで約275m"),
    (1600, 239.8, 2, "1角奥ポケット→2角まで約240m（外回り）"),
    (2000, None, 2, "外回り使用の2000m（数値保留。通常は内2000が主流）"),
    (2500, 192.0, 2, "向正面3角手前スタート→3角まで約192m"),
    (2600, None, 2, "外回り（数値保留）"),
    (3200, None, 4, "外回り周回（数値保留）"),
    (4000, None, 4, "外回り周回（数値保留）"),
]

def register():
    # 内回り
    for dist, first, turns, note in DIST_INNER:
        for rs, off, lap, wmin, wmax, straight in RAILS_INNER:
            _add(CourseGeometry(
                course_id="中山", surface="芝", layout="内回り",
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None, finish_grade_pct=None,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS, notes=note,
            ))

    # 外回り
    for dist, first, turns, note in DIST_OUTER:
        for rs, off, lap, wmin, wmax, straight in RAILS_OUTER:
            _add(CourseGeometry(
                course_id="中山", surface="芝", layout="外回り",
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None, finish_grade_pct=None,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS, notes=note,
            ))
