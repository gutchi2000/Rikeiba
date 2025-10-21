# -*- coding: utf-8 -*-
# course_geometry/chukyo_turf.py
# 中京競馬場（芝）— 左回り・A/B対応（直線412.5m・高低差3.5m・急坂~2%反映）
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "左"
TYPICAL_LANE_BIAS = (
    "直線412.5m＋直線入りで約2%の上り。差し・追い込みが水準以上に届く開催が多い。"
)

# 仮柵（A/B）の一周距離・幅員・直線長（JRA表）
RAILS = [  # (rail_state, offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0, 1705.9, 28, 30, 412.5),
    ("B", 3, 1724.8, 25, 27, 412.5),
]

# 高低差 3.5m。直線序盤の上りは約+2m/100m ≒ 2% → 代表値として finish_grade_pct=2.0
FINISH_GRADE_PCT = 2.0

# 中京の芝・公式発走距離（JRA表より）
DISTANCES = [
    (1200, "短距離（高松宮記念系）。向正面〜3角へ"),
    (1300, "向正面から。1200より1角距離が長い"),
    (1400, "向正面から。ペース速くなりやすい"),
    (1600, "マイル。向正面中ほどスタート"),
    (2000, "スタンド前4角奥Pから。長く脚を使う中距離"),
    (2200, "菊トライアル等。スタンド前〜1角長め"),
    (3000, "周回（num_turns=4）。長距離戦"),
]

def register():
    for dist, note in DISTANCES:
        turns = 4 if dist == 3000 else 2
        for rs, off, lap, wmin, wmax, straight in RAILS:
            _add(CourseGeometry(
                course_id="中京", surface="芝", layout="外回り",  # 中京は芝レイアウト1系統扱い
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=None,  # 数値ソース入手次第で更新
                num_turns=turns,
                elevation_gain_last600_m=None,
                finish_grade_pct=FINISH_GRADE_PCT,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
