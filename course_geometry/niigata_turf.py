# -*- coding: utf-8 -*-
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

COMMON = dict(
    course_id="新潟", surface="芝", direction="左",
    straight_length_m=1000.0,
    elevation_gain_last600_m=0.0, finish_grade_pct=0.0,
    track_width_min_m=30, track_width_max_m=30,
    typical_lane_bias="外ラチ優位が出やすい（開催次第）",
)

DISTANCES = [
    ("直線", 1000, None, 0, "直千。帯域=枠寄り"),
]

def register():
    for layout, dist, first, turns, note in DISTANCES:
        _add(CourseGeometry(
            layout=layout, distance_m=dist, start_to_first_turn_m=first, num_turns=turns,
            rail_state="A", rail_offset_m=0,
            notes=note, **COMMON
        ))
