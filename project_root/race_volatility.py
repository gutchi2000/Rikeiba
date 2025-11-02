# -*- coding: utf-8 -*-
"""
race_volatility.py

コース×距離×馬場ごとに「どれくらいレース結果がブレやすいか」を0.0～1.0で返すユーティリティと、
過去走DataFrameから1頭ぶんの平均ボラティリティを計算する関数をまとめたモジュール。

使い方（想定）:
    from race_volatility import compute_race_volatility

    horse_df = all_runs[all_runs["馬名"] == "タイトルホルダー"]
    vol = compute_race_volatility(horse_df)

    # df_agg にマージするなら、各馬でこれを回して DataFrame 化する
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ------------------------------------------------------------
# 1) コース×馬場×距離ごとの素点
# ------------------------------------------------------------
COURSE_DISTANCE_VOL: Dict[str, Dict[str, Dict[Any, float]]] = {
    "札幌": {
        "芝": {
            1200: 0.5,
            1500: 0.5,
            1800: 0.0,
            2000: 0.0,
            2600: 0.0,
            1000: 1.0,   # ごく少数開催だけど一応置いとく
        },
        "ダ": {
            1000: 1.0,
            1700: 0.5,
            2400: 0.0,
        },
    },
    "函館": {
        "芝": {
            1000: 1.0,
            1200: 1.0,   # 夏のスプリントで一番ブレる
            1800: 0.0,
            2000: 0.5,
            2600: 0.0,
        },
        "ダ": {
            1000: 1.0,
            1700: 0.5,
            2400: 0.0,
        },
    },
    "福島": {
        "芝": {
            1200: 1.0,   # 小回り＋短直線で先行頭数次第で崩れる
            1800: 0.5,
            2000: 0.5,
            2600: 0.0,
        },
        "ダ": {
            1150: 1.0,
            1700: 0.5,
            2400: 0.0,
        },
    },
    "新潟": {
        "芝": {
            1000: 1.0,   # 千直
            1200: 0.5,
            1400: 0.5,
            1600: 0.0,   # 外回りで能力順になりやすい
            1800: 0.0,
            "2000外": 1.0,  # 外回り2000はスタート地点特殊で紛れやすい
            "2000内": 0.5,
            2200: 0.0,
            2400: 0.0,
        },
        "ダ": {
            1000: 0.5,   # あまり組まれないので0.5で置き
            1200: 0.5,   # 芝スタートでペース速くなる
            1700: 0.5,
            1800: 0.0,
            2500: 0.0,
        },
    },
    "東京": {
        "芝": {
            1400: 0.5,
            1600: 0.0,
            1800: 0.0,
            2000: 1.0,   # ポケット＋頭数制限ありで位置が超大事
            2300: 0.0,
            2400: 0.0,
            2500: 0.0,
            2600: 0.0,
            3400: 0.0,
        },
        "ダ": {
            1200: 0.5,   # ごく一部開催、芝→ダでちょい荒れ
            1300: 0.5,   # 東京だけの特殊距離
            1400: 0.5,
            1600: 0.0,
            2100: 0.0,
        },
    },
    "中山": {
        "芝": {
            1200: 1.0,
            1600: 0.5,
            1800: 1.0,
            2000: 1.0,
            2200: 1.0,
            2500: 1.0,
            3600: 0.0,
        },
        "ダ": {
            1200: 1.0,
            1800: 0.5,
            2400: 0.5,
            2500: 0.5,
        },
    },
    "中京": {
        "芝": {
            1200: 0.5,
            1400: 0.5,
            1600: 0.5,
            2000: 0.5,
            2200: 0.5,
        },
        "ダ": {
            1200: 0.5,
            1400: 0.5,
            1800: 0.5,
            1900: 0.5,
        },
    },
    "京都": {
        "芝": {
            1200: 0.5,
            "1400内": 1.0,   # 内回りで位置ゲーになりやすい
            "1400外": 0.5,
            "1600内": 0.5,
            "1600外": 0.0,
            1800: 0.0,
            2000: 0.5,
            2200: 0.5,
            2400: 0.0,
            3000: 0.0,
            3200: 0.0,
        },
        "ダ": {
            1200: 0.5,
            1400: 0.5,
            1800: 0.0,
            1900: 0.0,
        },
    },
    "阪神": {
        "芝": {
            1200: 1.0,
            1400: 0.5,
            1600: 0.0,
            1800: 0.0,
            2000: 0.5,
            2200: 1.0,  # 宝塚のとこ
            2400: 0.0,
            2600: 0.0,
            3000: 0.0,
        },
        "ダ": {
            1200: 1.0,
            1400: 0.5,
            1800: 0.0,
            2000: 0.0,
        },
    },
    "小倉": {
        "芝": {
            1200: 1.0,
            1700: 0.5,
            1800: 0.0,
            2000: 0.5,
            2600: 0.0,
        },
        "ダ": {
            1000: 1.0,
            1700: 0.5,
            2400: 0.0,
        },
    },
}

# ------------------------------------------------------------
# 2) 正規化ユーティリティ
# ------------------------------------------------------------
def _normalize_course(name: str) -> str:
    if not name:
        return ""
    name = str(name).strip()
    # よくある表記ゆれを吸収
    name = name.replace("競馬場", "")
    if name in ("東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "札幌", "函館", "小倉"):
        return name
    # 万が一別表記ならここで追加
    return name

def _normalize_surface(surface: str) -> str:
    if not surface:
        return ""
    s = str(surface)
    if "芝" in s:
        return "芝"
    if "ダ" in s or "砂" in s:
        return "ダ"
    return s

def _build_distance_key(dist: int, layout: Optional[str], course: str, surface: str) -> Any:
    """
    新潟や京都みたいに「2000外」「1400内」があるところだけ特別扱いする。
    layout は df によくある '外', '内', '外回り', '内回り' を想定。
    それ以外は素直に int の距離を返す。
    """
    if layout is None:
        return dist

    lay = str(layout)
    lay = lay.replace("回り", "")  # "外回り" → "外"
    if course == "新潟" and surface == "芝" and dist == 2000:
        if "外" in lay:
            return "2000外"
        if "内" in lay:
            return "2000内"
        # レイアウト情報がなかったらとりあえず内に寄せる
        return "2000内"

    if course == "京都" and surface == "芝" and dist in (1400, 1600):
        if "外" in lay:
            return f"{dist}外"
        if "内" in lay:
            return f"{dist}内"

    return dist

# ------------------------------------------------------------
# 3) 1走ぶんを引く関数
# ------------------------------------------------------------
def get_course_distance_vol(
    course: str,
    surface: str,
    distance: int,
    layout: Optional[str] = None,
) -> float:
    """
    単発のレース情報からボラティリティ素点を取り出す。
    なければ 0.0 を返す（＝そのコース・距離で特に荒れを意識しない）
    """
    c = _normalize_course(course)
    s = _normalize_surface(surface)

    if not c or not s or not distance:
        return 0.0

    if c not in COURSE_DISTANCE_VOL:
        return 0.0
    if s not in COURSE_DISTANCE_VOL[c]:
        return 0.0

    key = _build_distance_key(int(distance), layout, c, s)
    return COURSE_DISTANCE_VOL[c][s].get(key, 0.0)

# ------------------------------------------------------------
# 4) DataFrame(過去走) → 1頭ぶんの平均ボラを計算
# ------------------------------------------------------------
def compute_race_volatility(df_runs) -> float:
    """
    df_runs: その馬の過去走だけを切り出した DataFrame を想定
        必要な列:
            - '競馬場' or '場名' or 'コース名' のどれか
            - '馬場' or 'トラック' or 'surface' のどれか
            - '距離' or '距離m' or 'distance_m' のどれか
            - あれば '回り' 'layout' 'course_layout' など

    return: 0.0～1.0 のだいたいの平均
    """
    if df_runs is None or len(df_runs) == 0:
        return 0.0

    # 列名のゆれに対応
    def pick(colnames):
        for c in colnames:
            if c in df_runs.columns:
                return c
        return None

    col_course = pick(["競馬場", "場名", "コース名", "race_course", "course"])
    col_surface = pick(["馬場", "トラック", "surface", "芝ダ", "馬場状態"])
    col_dist = pick(["距離", "距離m", "distance_m", "距離(m)"])
    col_layout = pick(["回り", "layout", "course_layout", "コース区分"])

    vols = []
    for _, row in df_runs.iterrows():
        course = row[col_course] if col_course else ""
        surface = row[col_surface] if col_surface else ""
        dist = row[col_dist] if col_dist else 0
        layout = row[col_layout] if col_layout else None

        # NaN 対策
        if dist is None or dist == "":
            dist = 0
        try:
            dist = int(dist)
        except Exception:
            dist = 0

        v = get_course_distance_vol(course, surface, dist, layout)
        vols.append(v)

    if not vols:
        return 0.0

    # 単純平均でOK。重み付けしたかったらここで替える。
    return float(sum(vols) / len(vols))
