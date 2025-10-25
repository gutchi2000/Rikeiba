# project_root/course_geometry/__init__.py
from __future__ import annotations
import importlib, pkgutil

from .registry import get_course_geom  # re-export

# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import pkgutil
import pathlib
from typing import Iterable, Set

# 外から利用されるAPIをエクスポート
try:
    from .registry import register as _register, clear_registry  # noqa: F401
    from .api import get_course_geom  # あなたの実装に合わせて
except Exception:
    # registry や api の実装が違う場合はここを調整
    pass

def _iter_turf_modules(pkg_name: str, search_dirs: Iterable[pathlib.Path]):
    """
    指定ディレクトリ群直下の *_turf.py を列挙し、(import_name, is_subdir) を返す。
    - pkg_name: このパッケージ名（例: 'course_geometry'）
    - search_dirs: 走査する物理ディレクトリ（ルート直下, turf/）
    """
    for base_dir in search_dirs:
        is_subdir = (base_dir.name == "turf")
        for m in pkgutil.iter_modules([str(base_dir)]):
            name = m.name
            if name.endswith("_turf"):
                if is_subdir:
                    import_name = f"{pkg_name}.turf.{name}"
                else:
                    import_name = f"{pkg_name}.{name}"
                yield import_name

def register_all_turf(*, clear: bool = False, verbose: bool = False) -> None:
    """
    ルート直下と turf/ サブディレクトリの両方から *_turf.py を読み込み、
    各モジュールの register() を呼ぶ。重複は自動で回避。
    - clear=True で開始前にレジストリ初期化
    """
    pkg = pathlib.Path(__file__).parent
    root_dir = pkg
    turf_dir = pkg / "turf"

    if clear:
        try:
            from .registry import clear_registry
            clear_registry()
            if verbose:
                print("[course_geometry] registry cleared.")
        except Exception:
            if verbose:
                print("[course_geometry] clear_registry() not available, skipped.")

    mod_names = list(_iter_turf_modules(__name__, [root_dir] + ([turf_dir] if turf_dir.exists() else [])))

    # 同一 import 名の重複は除去
    seen: Set[str] = set()
    loaded = 0

    for import_name in mod_names:
        if import_name in seen:
            continue
        seen.add(import_name)
        try:
            mod = importlib.import_module(import_name)
            if hasattr(mod, "register"):
                mod.register()
                loaded += 1
                if verbose:
                    print(f"[course_geometry] registered: {import_name}")
            else:
                if verbose:
                    print(f"[course_geometry] skip (no register()): {import_name}")
        except Exception as e:
            if verbose:
                print(f"[course_geometry] failed to load {import_name}: {e}")

    if verbose:
        print(f"[course_geometry] total registered modules: {loaded}")
