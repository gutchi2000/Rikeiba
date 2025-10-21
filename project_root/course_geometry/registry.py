# -*- coding: utf-8 -*-
# course_geometry/registry.py
from __future__ import annotations
import importlib
import pkgutil
from dataclasses import asdict
from typing import Dict, Tuple, Any, List

# CourseGeometry は既存の base_types から使う前提
from .base_types import CourseGeometry

# 内部レジストリ: キー = (course_id, surface, layout, rail_state, distance_m)
_REG: Dict[Tuple[str, str, str, str, int], Dict[str, Any]] = {}
_REGISTERED = False

def _add(cg: CourseGeometry) -> None:
    """*_turf.py から呼ばれる想定の登録関数（互換用）"""
    key = (
        cg.course_id, cg.surface, cg.layout, cg.rail_state, int(cg.distance_m)
    )
    _REG[key] = asdict(cg)

def _auto_import_all_turf_modules() -> None:
    """course_geometry パッケージ内の *_turf.py を全て import し、各モジュールの register() を呼ぶ。"""
    pkg_name = __name__.rsplit(".", 1)[0]  # "course_geometry"
    pkg = importlib.import_module(pkg_name)
    for m in pkgutil.iter_modules(pkg.__path__):
        name = m.name
        if not name.endswith("_turf"):
            continue
        mod = importlib.import_module(f"{pkg_name}.{name}")
        # 既存スタイル: 引数なし register() が _add を内部 import して呼ぶ
        if hasattr(mod, "register") and callable(mod.register):
            mod.register()

def register_all_turf(force: bool = False) -> None:
    """全コース登録（idempotent）"""
    global _REGISTERED
    if _REGISTERED and not force:
        return
    _REG.clear()
    _auto_import_all_turf_modules()
    if not _REG:
        raise RuntimeError("No turf geometry registered. *_turf.py の register() が実行されませんでした。")
    _REGISTERED = True

def _nearest_same_layout(key_like: Tuple[str, str, str, str, int]) -> Tuple[str, str, str, str, int] | None:
    """距離が一致しない時の同 venue/surface/layout/rail の最近傍距離を返す"""
    cand = [k for k in _REG.keys() if k[:4] == key_like[:4]]
    if not cand:
        return None
    target_d = key_like[4]
    return min(cand, key=lambda k: abs(k[4] - target_d))

def get_course_geom(course_id: str, surface: str, distance_m: int, layout: str, rail_state: str) -> Dict[str, Any]:
    """幾何を取得（距離が無ければ最近傍距離でフォールバック）"""
    if not _REGISTERED:
        register_all_turf()
    key = (course_id, surface, layout, rail_state, int(distance_m))
    if key in _REG:
        return _REG[key]
    near = _nearest_same_layout(key)
    if near is None:
        raise KeyError(f"geometry not found: {key}")
    return _REG[near]

# ==== 物理S1側が使う軽ユーティリティ（ダミーでもOK） ====

def estimate_tci(geom: Dict[str, Any]) -> float:
    # コーナー難度の概算（メタから推定。無ければ 1.0）
    wmin = float(geom.get("track_width_min_m", 26.0))
    firstR = float(geom.get("first_turn_R_m", 120.0))
    base = float(geom.get("tci", 1.0))
    # 幅が狭いほど +、R が小さいほど +（=難しい）
    return base * (26.0 / max(20.0, wmin)) * (120.0 / max(80.0, firstR))

def gate_influence_coeff(geom: Dict[str, Any], headcount: int) -> float:
    w = float(geom.get("track_width_min_m", 26.0))
    firstR = float(geom.get("first_turn_R_m", 120.0))
    n = max(8, int(headcount))
    k = 0.20 + 0.35*(n/18.0) + 0.25*(26.0/w) + 0.20*(120.0/firstR)
    return float(max(0.0, min(1.0, k)))

def band_split(headcount: int):
    n = int(max(1, headcount))
    a = n // 3
    b = n // 3
    c = n - a - b
    return a, b, c
