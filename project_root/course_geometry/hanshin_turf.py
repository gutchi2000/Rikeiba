# -*- coding: utf-8 -*-
# course_geometry/hanshin_turf.py
# 阪神競馬場（芝）— 内回り/外回り・A/B対応、直線長・幅員・一周距離・勾配を反映
from __future__ import annotations
from .base_types import CourseGeometry
from .registry import _add

DIRECTION = "右"
TYPICAL_LANE_BIAS = (
    "外回りは直線473.6mで実力が出やすい。内回りは356.5m。"
    "ゴール前に高低差1.8mの上り（最大勾配約1.5%）。開催や馬場次第で内先行が残る。"
)

# ── 仮柵（阪神はA/Bのみ） 一周距離・幅員・直線長（JRA表）
# 直線長：内=356.5m / 外=473.6m
RAILS_INNER = [  # (rail_state, rail_offset_m, lap_len_m, width_min, width_max, straight_len_m)
    ("A", 0,   1689.0, 24, 28, 356.5),
    ("B", 3,   1713.2, 20, 25, 359.1),   # 断面図：直線+3m/曲線+4m相当
]
RAILS_OUTER = [
    ("A", 0,   2089.0, 24, 29, 473.6),
    ("B", 3,   2113.2, 20, 25, 476.3),
]

# 勾配メモ：コース紹介よりゴール前の上りは「高低差1.8m・勾配1.5%」。
# → finish_grade_pct=1.5 を代表値として採用（平均値が別に必要なら後日変更可）
FINISH_GRADE_PCT = 1.5

# ── 内回り：JRAの発走距離
DIST_INNER = [
    # (distance_m, start_to_first_turn_m, num_turns, note)
    (1200, None, 2, "内回り。向正面半ばスタート"),
    (1400, None, 2, "内回り。直線入口付近スタート"),
    (2000, None, 2, "内回り。スタンド前4角寄りから"),
    (2200, None, 2, "内回り。スタンド前4角奥ポケットから"),
    (3000, None, 4, "内回り・周回"),
    (3200, None, 4, "内回り・周回"),
]

# ── 外回り：JRAの発走距離
DIST_OUTER = [
    (1400, None, 2, "外回り。直線入口付近から3角へ"),
    (1600, None, 2, "外回り。向正面半ばスタート（東京マイルと発走位置の比較に触れられる距離）"),
    (1800, None, 2, "外回り。直線奥寄りから"),
    (2400, None, 2, "外回り"),
    (2600, None, 2, "外回り"),
    (3200, None, 4, "外回り・周回"),
]

def register():
    # 内回り
    for dist, first, turns, note in DIST_INNER:
        for rs, off, lap, wmin, wmax, straight in RAILS_INNER:
            _add(CourseGeometry(
                course_id="阪神", surface="芝", layout="内回り",
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,
                finish_grade_pct=FINISH_GRADE_PCT,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))

    # 外回り
    for dist, first, turns, note in DIST_OUTER:
        for rs, off, lap, wmin, wmax, straight in RAILS_OUTER:
            _add(CourseGeometry(
                course_id="阪神", surface="芝", layout="外回り",
                distance_m=dist, direction=DIRECTION,
                straight_length_m=straight,
                start_to_first_turn_m=first, num_turns=turns,
                elevation_gain_last600_m=None,
                finish_grade_pct=FINISH_GRADE_PCT,
                rail_state=rs, rail_offset_m=off, lap_length_m=lap,
                track_width_min_m=wmin, track_width_max_m=wmax,
                typical_lane_bias=TYPICAL_LANE_BIAS,
                notes=note,
            ))
