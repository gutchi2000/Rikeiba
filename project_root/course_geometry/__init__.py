# project_root/course_geometry/__init__.py
from __future__ import annotations
import importlib, pkgutil

from .registry import get_course_geom  # re-export

def register_all_turf() -> None:
    """
    このパッケージ直下の '*_turf.py' を走査し、各モジュールの register() を呼ぶ。
    """
    pkg = __name__
    for m in pkgutil.iter_modules(__path__):
        if m.name.endswith("_turf"):
            mod = importlib.import_module(f"{pkg}.{m.name}")
            if hasattr(mod, "register"):
                mod.register()
