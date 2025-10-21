# project_root/course_geometry/registry.py
from __future__ import annotations
import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

# ============ 内部レジストリ ============
# キー: (course_id, surface, layout, rail_state, distance_m)
_REG: Dict[Tuple[str, str, str, str, int], Dict[str, Any]] = {}
_REGISTERED = False

@dataclass(frozen=True)
class Geom:
    course_id: str     # 例: "東京"
    surface: str       # "芝" or "ダ"
    layout: str        # "内回り" / "外回り" / "直線"
    rail_state: str    # "A"|"B"|"C"|"D"
    distance_m: int
    # 任意で使う情報（S1やTciの計算に利用）
    meta: Dict[str, Any]

def _add(course_id: str, surface: str, layout: str, rail_state: str, distance_m: int, meta: Dict[str, Any]):
    key = (course_id, surface, layout, rail_state, int(distance_m))
    _REG[key] = dict(
        course_id=course_id,
        surface=surface,
        layout=layout,
        rail_state=rail_state,
        distance_m=int(distance_m),
        **(meta or {}),
    )

def _auto_import_all_turf_modules():
    """course_geometry パッケージ内の *_turf.py を全部 import して、各モジュールの register(reg) を叩く"""
    pkg_name = __name__.rsplit(".", 1)[0]  # "course_geometry"
    pkg = importlib.import_module(pkg_name)
    for m in pkgutil.iter_modules([Path(pkg.__file__).parent.as_posix()]):
        name = m.name
        if not name.endswith("_turf"):
            continue
        mod = importlib.import_module(f"{pkg_name}.{name}")
        # 各 *_turf.py は register(registry_add_func) を持つ前提
        if hasattr(mod, "register") and callable(mod.register):
            mod.register(_add)

def register_all_turf(force: bool = False) -> None:
    """各競馬場モジュールを読み込み、幾何レジストリを構築"""
    global _REGISTERED
    if _REGISTERED and not force:
        return
    _REG.clear()
    _auto_import_all_turf_modules()
    if not _REG:
        raise RuntimeError("No turf geometry registered. *_turf.py の register() が呼ばれていません。")
    _REGISTERED = True

def get_course_geom(course_id: str, surface: str, distance_m: int, layout: str, rail_state: str) -> Dict[str, Any]:
    if not _REGISTERED:
        register_all_turf()
    key = (course_id, surface, layout, rail_state, int(distance_m))
    if key in _REG:
        return _REG[key]
    # 距離がグリッド化されている場合の簡易最近傍フォールバック（同 venue/surface/layout/rail の中で距離最小差）
    cands = [k for k in _REG.keys() if k[:4] == key[:4]]
    if not cands:
        raise KeyError(f"geometry not found: {key}")
    nearest = min(cands, key=lambda k: abs(k[4] - key[4]))
    return _REG[nearest]

# ===== ここからは S1 が使う軽量ユーティリティ =====
def estimate_tci(geom: Dict[str, Any]) -> float:
    """Turn/Camber/Incline の総合っぽい係数（暫定のダミーでもOK）。なければ meta から推定。"""
    # 例: コーナー角度や勾配が meta で提供されるならそこから
    return float(geom.get("tci", geom.get("corner_gain", 1.0)))

def gate_influence_coeff(geom: Dict[str, Any], headcount: int) -> float:
    """枠順影響の強さ（0..1）"""
    # コース幅やコーナー半径からヒューリスティック
    w = float(geom.get("track_width_m", 26.0))
    r = float(geom.get("first_turn_R_m", 120.0))
    n = max(8, int(headcount))
    k = min(1.0, max(0.0, 0.25 + 0.35*(n/18.0) + 0.20*(26.0/w) + 0.20*(120.0/r)))
    return k

def band_split(headcount: int):
    """内・中・外の概算本数（例: 18頭 → 6/6/6）"""
    n = int(max(1, headcount))
    a = n // 3
    b = n // 3
    c = n - a - b
    return a, b, c
