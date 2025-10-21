# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CourseKey:
    course_id: str          # "東京" など
    surface: str            # 常に "芝"
    layout: str             # "外回り" / "内回り" / "直線"
    distance_m: int         # レース距離
    rail_state: str = "A"   # "A","B","C","D"

@dataclass
class CourseGeometry:
    course_id: str
    surface: str            # 常に "芝"
    layout: str
    distance_m: int
    direction: str                  # "右" or "左"
    straight_length_m: Optional[float] = None
    start_to_first_turn_m: Optional[float] = None
    num_turns: int = 0
    elevation_gain_last600_m: Optional[float] = None
    finish_grade_pct: Optional[float] = None
    rail_state: str = "A"
    rail_offset_m: Optional[float] = None
    lap_length_m: Optional[float] = None
    track_width_min_m: Optional[float] = None
    track_width_max_m: Optional[float] = None
    typical_lane_bias: Optional[str] = None
    notes: Optional[str] = None
