# -*- coding: utf-8 -*-
"""
mc_and_spectrum.py

数学系の実装は ``math_utils.py`` へ移動しました。

- ``run_mc_prob``: 競馬スコアから勝率/連対率/複勝率を推定する高速モンテカルロ
- ``spectrum_fit``: 区間ペースなど等間隔系列からスペクトル適合係数を計算
- ``specfit_to_z``: スペクトル適合係数を Z 値に変換
"""

from __future__ import annotations

from .math_utils import run_mc_prob, specfit_to_z, spectrum_fit

__all__ = [
    "run_mc_prob",
    "spectrum_fit",
    "specfit_to_z",
]

