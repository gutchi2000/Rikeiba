# project_root/course_geometry/__init__.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import pkgutil
import pathlib
from typing import Iterable, Set

# --- 外部に見せるAPI（存在する方を優先して再エクスポート） ---
# get_course_geom は registry にある想定。無ければ api から。
try:
    from .registry import get_course_geom  # type: ignore[attr-defined]
except Exception:
    from .api import get_course_geom  # type: ignore[no-redef]

# クリア関数（あれば使う）
try:
    from .registry import clear_registry  # type: ignore[attr-defined]
except Exception:
    clear_registry = None  # 無くても動くように

__all__ = [
    "register_all_turf",
    "register_all_dirt",
    "get_course_geom",
    "clear_registry",
]

# --- 内部: *_turf モジュールを列挙 ---
def _iter_turf_modules(pkg_name: str, search_dirs: Iterable[pathlib.Path]):
    """
    指定ディレクトリ群直下の *_turf.py を列挙し、import 名を返す。
    - pkg_name: このパッケージ名（例: 'course_geometry'）
    - search_dirs: 走査する物理ディレクトリ（ルート直下, turf/）
    """
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        is_subdir = (base_dir.name == "turf")
        for m in pkgutil.iter_modules([str(base_dir)]):
            name = m.name
            if name.endswith("_turf"):
                import_name = f"{pkg_name}.turf.{name}" if is_subdir else f"{pkg_name}.{name}"
                yield import_name

# --- 内部: *_dirt モジュールを列挙 ---
def _iter_dirt_modules(pkg_name: str, search_dirs: Iterable[pathlib.Path]):
    """
    指定ディレクトリ群直下の *_dirt.py を列挙し、import 名を返す。
    """
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        is_subdir = base_dir.name == "dirt"
        for m in pkgutil.iter_modules([str(base_dir)]):
            name = m.name
            if name.endswith("_dirt"):
                import_name = (
                    f"{pkg_name}.dirt.{name}" if is_subdir else f"{pkg_name}.{name}"
                )
                yield import_name

# --- 公開: 全 turf の register() を呼ぶ ---
def register_all_turf(*, clear: bool = False, verbose: bool = False) -> None:
    """
    ルート直下と turf/ サブディレクトリの両方から *_turf.py を import し、
    各モジュールの register() を呼び出す。
    - clear=True なら開始前にレジストリ初期化（clear_registry がある場合）
    """
    pkg_dir = pathlib.Path(__file__).parent
    root_dir = pkg_dir
    turf_dir = pkg_dir / "turf"

    if clear and callable(clear_registry):
        try:
            clear_registry()  # type: ignore[misc]
            if verbose:
                print("[course_geometry] registry cleared.")
        except Exception as e:
            if verbose:
                print(f"[course_geometry] clear_registry() failed: {e}")

    mod_names = list(_iter_turf_modules(__name__, [root_dir, turf_dir]))
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
            elif verbose:
                print(f"[course_geometry] skip (no register()): {import_name}")
        except Exception as e:
            if verbose:
                print(f"[course_geometry] failed to load {import_name}: {e}")

    if verbose:
        print(f"[course_geometry] total registered modules: {loaded}")


# --- 公開: 全 dirt の register() を呼ぶ ---
def register_all_dirt(*, clear: bool = False, verbose: bool = False) -> None:
    """
    ルート直下と dirt/ サブディレクトリの両方から *_dirt.py を import し、各モジュールの
    register() を呼び出す。clear=True なら開始前に clear_registry() を呼び出す。
    """
    pkg_dir = pathlib.Path(__file__).parent
    root_dir = pkg_dir
    dirt_dir = pkg_dir / "dirt"

    if clear and callable(clear_registry):
        try:
            clear_registry()  # type: ignore[misc]
            if verbose:
                print("[course_geometry] registry cleared.")
        except Exception as e:
            if verbose:
                print(f"[course_geometry] clear_registry() failed: {e}")

    mod_names = list(_iter_dirt_modules(__name__, [root_dir, dirt_dir]))
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
                    print(f"[course_geometry] registered dirt: {import_name}")
            elif verbose:
                print(f"[course_geometry] skip dirt (no register()): {import_name}")
        except Exception as e:
            if verbose:
                print(f"[course_geometry] failed to load dirt {import_name}: {e}")

    if verbose:
        print(f"[course_geometry] total registered dirt modules: {loaded}")
