# project_root/course_geometry/registry.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
from .base_types import CourseGeometry

# key: (course_id, surface, distance_m, layout, rail_state)
_REGISTRY: Dict[Tuple[str, str, int, str, str], CourseGeometry] = {}

def _key(course_id: str, surface: str, distance_m: int, layout: str, rail_state: str):
    return (str(course_id), str(surface), int(distance_m), str(layout), str(rail_state))

def _add(geom: CourseGeometry) -> None:
    """各 *_turf.py の register() から呼ばれて、幾何を登録する内部API。"""
    _REGISTRY[_key(geom.course_id, geom.surface, geom.distance_m, geom.layout, geom.rail_state)] = geom

def get_course_geom(course_id: str, surface: str, distance_m: int, layout: str, rail_state: str) -> Optional[CourseGeometry]:
    """アプリ側/物理側が幾何を取得するときに使う外部API。"""
    return _REGISTRY.get(_key(course_id, surface, int(distance_m), layout, rail_state))
