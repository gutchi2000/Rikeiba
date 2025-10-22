# -*- coding: utf-8 -*-
"""
course_geometry/kyoto_turf.py
京都競馬場（芝）: 内回り/外回り × A/B/C/D 柵の幾何定義
- 対応距離（代表）
  内回り: 1200, 1400, 1600, 2000, 2200
  外回り: 1400, 1600, 1800, 2400, 3000, 3200
"""
from __future__ import annotations

from .base_types import CourseGeometry
from .registry import _add

# 方向
DIRECTION = "右"

# ---- 仮柵ごとの一周距離・幅員・直線長（m） ----
# (rail_state, rail_offset_m, lap_length_m, width_min, width_max, straight_len_m)
RAILS_INNER = [
    ("A", 0.0, 1782.8, 27.0, 38.0, 328.4),
    ("B", 3.0, 1802.2, 24.0, 35.0, 323.4),
    ("C", 6.0, 1821.1, 21.0, 32.0, 323.4),
    ("D", 9.0, 1839.9, 18.0, 29.0, 323.4),
]
RAILS_OUTER = [
    ("A", 0.0, 1894.3, 24.0, 38.0, 403.7),
    ("B", 3.0, 1913.6, 21.0, 35.0, 398.7),
    ("C", 6.0, 1932.4, 18.0, 32.0, 398.7),
    ("D", 9.0, 1951.3, 15.0, 29.0, 398.7),
]

# ---- 実施距離テーブル ----
# (distance_m, start_to_first_turn_m, num_turns, note)
DIST_INNER = [
    (1200, 316.2, 2, "向正面半ば発走（1角まで約316m）"),
    (1400, 516.2, 2, "2角奥ポケット発走（3角まで約516m）"),
    (1600, 516.8, 2, "2角奥ポケット発走（3角まで約517m）"),
    (2000, 308.3, 2, "スタンド前発走（1角まで約308m）"),
    (2200, 397.7, 2, "スタンド前発走（1角まで約398m）"),
]

DIST_OUTER = [
    (1400, 511.7, 2, "2角出口付近発走（3角まで約512m）"),
    (1600, 711.7, 2, "直線坂下発走（3角まで約712m）"),
    (1800, 911.7, 2, "直線奥ポケット発走（3角まで約912m）"),
    (2400, 597.7, 2, "4角奥ポケット発走（1角まで約598m）"),
    (3000, 217.4, 4, "向正面中ほど発走（1角まで約217m・周回）"),
    (3200, 412.7, 4, "向正面半ば発走（1角まで約413m・周回）"),
]

# 傾向のメモ（任意）
TYPICAL_LANE_BIAS = "開催週により内有利が出やすい。長距離は持久寄り。"

# ---- レイアウト別デフォルト勾配（仮） ----
# finish_grade_pct は「%」（0.40 = 0.40%）、elevation_gain_last600_m は「m」
DEFAULTS = {
    "内回り": {"elev600": 1.4, "finish_pct": 0.35},  # 1.4m / 0.35%
    "外回り": {"elev600": 2.0, "finish_pct": 0.40},  # 2.0m / 0.40%
}

def _add_block(layout: str, distances, rails, elev600_default: float, finish_pct_default: float):
    for dist_m, first_m, turns, note in distances:
        for rs, roff, lap_m, wmin, wmax, straight_m in rails:
            _add(CourseGeometry(
                course_id="京都",
                surface="芝",
                layout=layout,
                distance_m=int(dist_m),
                direction=DIRECTION,
                straight_length_m=float(straight_m),
                start_to_first_turn_m=float(first_m),
                num_turns=int(turns),
                # ★ デフォルトの勾配・標高差を設定（None にしない）
                elevation_gain_last600_m=float(elev600_default),
                finish_grade_pct=float(finish_pct_default),
                rail_state=str(rs),
                rail_offset_m=float(roff),
                lap_length_m=float(lap_m),
                track_width_min_m=float(wmin),
                track_width_max_m=float(wmax),
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=str(note),
            ))

def register():
    """registry.register_all_turf() から呼ばれる登録エントリポイント"""
    _add_block("内回り", DIST_INNER, RAILS_INNER,
               DEFAULTS["内回り"]["elev600"], DEFAULTS["内回り"]["finish_pct"])
    _add_block("外回り", DIST_OUTER, RAILS_OUTER,
               DEFAULTS["外回り"]["elev600"], DEFAULTS["外回り"]["finish_pct"])

__all__ = ["register"]
