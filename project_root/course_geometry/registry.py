# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional, Tuple
from importlib import import_module
import pkgutil

from .base_types import CourseKey, CourseGeometry

COURSES: Dict[CourseKey, CourseGeometry] = {}

def _add(g: CourseGeometry):
    key = CourseKey(g.course_id, g.surface, g.layout, g.distance_m, g.rail_state)
    COURSES[key] = g

def get_course_geom(course_id: str, surface: str, distance_m: int,
                    layout: Optional[str] = None, rail_state: str = "A") -> CourseGeometry:
    cand: Dict[CourseKey, CourseGeometry] = {}
    for k, v in COURSES.items():
        if k.course_id != course_id or k.surface != surface:
            continue
        if k.distance_m != distance_m:
            continue
        if layout and k.layout != layout:
            continue
        if k.rail_state != rail_state:
            continue
        cand[k] = v
    if cand:
        return next(iter(cand.values()))
    for k, v in COURSES.items():
        if k.course_id == course_id and k.surface == surface and k.distance_m == distance_m:
            if (not layout) or (k.layout == layout):
                return v
    for k, v in COURSES.items():
        if k.course_id == course_id and k.surface == surface:
            if abs(k.distance_m - distance_m) <= 100 and ((not layout) or (k.layout == layout)):
                return v
    raise KeyError(f"Course geometry not found: {course_id}/{surface}/{distance_m}/{layout or '*'} ({rail_state})")

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def estimate_tci(g: CourseGeometry) -> float:
    straight_n = _clip01(((g.straight_length_m or 0) - 300.0) / (600.0 - 300.0))
    firstturn_cost = 1.0 - _clip01(((g.start_to_first_turn_m or 800.0) - 200.0) / (800.0 - 200.0))
    width_avg = None
    if g.track_width_min_m and g.track_width_max_m:
        width_avg = (g.track_width_min_m + g.track_width_max_m) / 2.0
    width_cost = 1.0 - _clip01(((width_avg or 30.0) - 22.0) / (41.0 - 22.0))
    grade_n = _clip01((g.finish_grade_pct or 0.0) / 2.5)
    turns_n = _clip01((g.num_turns or 0) / 4.0)
    tci01 = (0.30*straight_n + 0.25*firstturn_cost + 0.15*width_cost + 0.20*grade_n + 0.10*turns_n)
    return round(100.0 * tci01, 1)

def gate_influence_coeff(g: CourseGeometry, headcount: int) -> float:
    base = 1.0 - _clip01(((g.start_to_first_turn_m or 800.0) - 200.0) / (800.0 - 200.0))
    density = _clip01((headcount - 8) / (18 - 8))
    return _clip01(0.6*base + 0.4*density)

def band_split(headcount: int) -> Tuple[range, range, range]:
    if headcount <= 12:
        inner = range(1, 1+max(3, headcount//4))
        outer = range(headcount - max(2, headcount//6) + 1, headcount + 1)
        middle = range(inner.stop, outer.start)
    else:
        inner = range(1, 5)
        outer = range(headcount-4, headcount+1)
        middle = range(inner.stop, outer.start)
    return inner, middle, outer

def register_all_turf():
    """package内の *_turf.py を自動検出して register() を呼ぶ"""
    import course_geometry as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.endswith("_turf"):
            mod = import_module(f"{pkg.__name__}.{m.name}")
            if hasattr(mod, "register"):
                mod.register()
