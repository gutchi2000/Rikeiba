# -*- coding: utf-8 -*-
"""中山競馬場のダートコース定義。"""
from __future__ import annotations

from ..base_types import CourseGeometry
from ..registry import _add

DIRECTION = "右"
LAP_LEN = 1493.0
STRAIGHT = 308.0
WIDTH = 20.0
ELEV600 = 3.5        # 仮：高低差3.5m
FINISH_PCT = (ELEV600 / 600.0) * 100.0

DISTANCES = [
    (1200, 250.0, 2, ""),
    (1700, 250.0, 2, ""),
    (1800, 250.0, 2, ""),
    (2400, 250.0, 2, ""),
]


def register() -> None:
    for dist, first, turns, note in DISTANCES:
        _add(
            CourseGeometry(
                course_id="中山",
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
                typical_lane_bias="急坂の短い直線で逃げ先行有利",
                notes=note,
            )
        )


__all__ = ["register"]
