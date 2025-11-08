# -*- coding: utf-8 -*-
"""京都競馬場のダートコース定義。"""
from __future__ import annotations

from ..base_types import CourseGeometry
from ..registry import _add

DIRECTION = "右"
LAP_LEN = 1607.6
STRAIGHT = 329.1
WIDTH = 25.0
ELEV600 = 3.0         # 仮：高低差3.0m
FINISH_PCT = (ELEV600 / 600.0) * 100.0

DISTANCES = [
    (1200, 300.0, 2, ""),
    (1400, 300.0, 2, ""),
    (1800, 300.0, 2, ""),
    (1900, 300.0, 2, ""),
]


def register() -> None:
    for dist, first, turns, note in DISTANCES:
        _add(
            CourseGeometry(
                course_id="京都",
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
                typical_lane_bias="起伏あり、パワー必要",
                notes=note,
            )
        )


__all__ = ["register"]
