# -*- coding: utf-8 -*-
"""中京競馬場のダートコース定義。"""
from __future__ import annotations

from ..base_types import CourseGeometry
from ..registry import _add

DIRECTION = "左"
LAP_LEN = 1530.0
STRAIGHT = 410.7
WIDTH = 25.0
ELEV600 = 2.5        # 仮：高低差2.5m
FINISH_PCT = (ELEV600 / 600.0) * 100.0

DISTANCES = [
    (1200, 350.0, 2, ""),
    (1400, 350.0, 2, ""),
    (1800, 350.0, 2, ""),
    (1900, 350.0, 2, ""),
]


def register() -> None:
    for dist, first, turns, note in DISTANCES:
        _add(
            CourseGeometry(
                course_id="中京",
                surface="ダ",
                layout="内回り",
                distance_m=int(dist),
                direction=DIRECTION,
                straight_length_m=STRAIGHT,
                start_to_first_turn_m=first,
                num_turns=int(turns),
                elevation_gain_last600_m=ELEV600,
                finish_grade_pct=FINISH_PCT,
                rail_state="",
                rail_offset_m=0.0,
                lap_length_m=LAP_LEN,
                track_width_min_m=WIDTH,
                track_width_max_m=WIDTH,
                typical_lane_bias="直線入り口が上り坂で差し有利",
                notes=note,
            )
        )


__all__ = ["register"]
