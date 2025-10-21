# -*- coding: utf-8 -*-
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

# 仮柵A=0, B=+3, C=+6, D=+9（JRA数値）
RAILS = [
    ("A", 0,   2083.1, 31, 41),
    ("B", 3,   2101.9, 28, 38),
    ("C", 6,   2120.8, 25, 35),
    ("D", 9,   2139.6, 22, 32),
]

COMMON = dict(
    course_id="東京", surface="芝", direction="左",
    straight_length_m=525.9,
    elevation_gain_last600_m=None,   # 高低図から後で追記OK
    finish_grade_pct=None,
    typical_lane_bias="季節で内外の振れ。春～初夏はフラット寄り",
)

DISTANCES = [
    # layout, distance_m, start_to_first_turn_m, num_turns, notes
    ("外回り", 1600, 540, 2, "長直線×瞬発（アルテミス/安田）"),
    ("外回り", 1400, None, 2, "要入力"),
    ("外回り", 1800, None, 2, "要入力"),
    ("外回り", 2000, None, 2, "要入力"),
    ("外回り", 2300, None, 2, "要入力"),
    ("外回り", 2400, None, 2, "ダービー/オークス系"),
    ("外回り", 2500, None, 2, "要入力"),
    ("外回り", 2600, None, 2, "要入力"),
]

def register():
    for layout, dist, first, turns, note in DISTANCES:
        for rs, off, lap, wmin, wmax in RAILS:
            _add(CourseGeometry(
                layout=layout, distance_m=dist, start_to_first_turn_m=first, num_turns=turns,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                notes=note, **COMMON
            ))
