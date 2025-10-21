# -*- coding: utf-8 -*-
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

COMMON = dict(
    course_id="京都", surface="芝", direction="右",
    straight_length_m=404.0,
    elevation_gain_last600_m=None,  # 後で追記OK
    finish_grade_pct=None,
    typical_lane_bias="長距離は持久寄り。週により内有利が出やすい",
)

DISTANCES = [
    ("外回り", 3000, 300, 4, "菊花賞系。配分/心肺要求↑"),
    ("外回り", 1800, 700, 2, "カシオペアS系。瞬発×位置バランス"),
]

def register():
    for layout, dist, first, turns, note in DISTANCES:
        for rs, off in [("A",0),("B",3),("C",6),("D",9)]:
            _add(CourseGeometry(
                layout=layout, distance_m=dist, start_to_first_turn_m=first, num_turns=turns,
                rail_state=rs, rail_offset_m=off,
                notes=note, **COMMON
            ))
