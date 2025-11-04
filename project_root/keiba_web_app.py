# -*- coding: utf-8 -*- 
# 競馬予想アプリ（AUTO統合版 + スペクトル解析）

from __future__ import annotations

# ===== 標準ライブラリ =====
import os, sys, io, re, json, math

# ===== 自作モジュールを優先解決（← ここを最初に）=====
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# リポジトリの親ディレクトリ（ui_style.py がある場所）もパスに入れる
ROOT = os.path.abspath(os.path.join(BASE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ===== サードパーティ =====
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# === 2歳戦検出 ===
def detect_2yo_race(horses_df, race_df) -> bool:
    import pandas as pd
    # 出走表の年齢がすべて2歳、または競走名に「2歳/二歳/2才」
    cond_age = False
    if isinstance(horses_df, pd.DataFrame) and '年齢' in horses_df.columns:
        ages = pd.to_numeric(horses_df['年齢'], errors='coerce').dropna()
        if len(ages) and ages.max() <= 2:
            cond_age = True
    name0 = ''
    if isinstance(race_df, pd.DataFrame) and '競走名' in race_df.columns and len(race_df):
        name0 = str(race_df['競走名'].iloc[0])
    cond_name = any(k in name0 for k in ['2歳','二歳','2才'])
    return bool(cond_age or cond_name)

# ← パスを通した “後” にローカルモジュールを import
from ui_style import topbar, card, pill, score_bar
from course_geometry import register_all_turf, get_course_geom
from physics_sprint1 import add_phys_s1_features
from race_volatility import compute_race_volatility

# ===== Streamlit 先にページ設定（UIを使う前に呼ぶ）=====
st.set_page_config(page_title="Rikeiba", layout="wide")

# ===== コース幾何の初期化（1回だけ）=====
@st.cache_resource
def _boot_course_geom(version: int = 1):
    # 直前の登録をクリア（定義があれば使う）
    try:
        from course_geometry.registry import clear_registry
        clear_registry()
    except Exception:
        # clear_registry が無ければ無視（_add が上書きしてくれる想定）
        pass

    # すべての *_turf.py を登録
    from course_geometry import register_all_turf
    register_all_turf()
    return True

# ← 数字を上げると Streamlit のキャッシュが破棄されて再登録される
_boot_course_geom(version=39)


# ※ races_df に対して add_phys_s1_features を“ここでは”実行しないこと。
#   実際の実行は UI 側（例：🧪 PhysS1 スモークテストボタン）で行う。

# ---- optional deps（無くても動く系）----
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

try:
    import streamlit.components.v1 as components
    COMPONENTS = True
except Exception:
    COMPONENTS = False

try:
    from sklearn.isotonic import IsotonicRegression
    SK_ISO = True
except Exception:
    SK_ISO = False

# ===== Matplotlib（ヘッドレス安全化 & 日本語フォント）=====
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境（Streamlit Cloud 等）で安全
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'IPAexGothic','IPAGothic','Noto Sans CJK JP','Yu Gothic UI','Meiryo','Hiragino Sans','MS Gothic'
]

@st.cache_resource
def get_jp_font():
    for p in [
        'ipaexg.ttf',
        '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/Hiragino Sans W3.ttc',
        'C:/Windows/Fonts/meiryo.ttc',
    ]:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
            return font_manager.FontProperties(fname=p)
    return None

_jp_font = get_jp_font()
if _jp_font is not None:
    try:
        plt.rcParams['font.family'] = _jp_font.get_name()
    except Exception:
        pass

# ===== 小ユーティリティ =====
STYLES = ['逃げ','先行','差し','追込']
_fw = str.maketrans('０１２３４５６７８９％','0123456789%')

STYLE_ALIASES = {
    '追い込み':'追込','追込み':'追込','おいこみ':'追込','おい込み':'追込',
    'さし':'差し','差込':'差し','差込み':'差し',
    'せんこう':'先行','先行 ':'先行','先行　':'先行',
    'にげ':'逃げ','逃げ ':'逃げ','逃げ　':'逃げ'
}

def normalize_style(s: str) -> str:
    s = str(s).replace('　','').strip().translate(_fw)
    s = STYLE_ALIASES.get(s, s)
    return s if s in STYLES else ''

# ===== 表示スタイル（枠色）=====
WAKU_COLORS = {
    1: "#FFFFFF",  # 白
    2: "#000000",  # 黒
    3: "#FF0000",  # 赤
    4: "#0000FF",  # 青
    5: "#FFFF00",  # 黄
    6: "#008000",  # 緑
    7: "#FF7F00",  # 橙
    8: "#FF69B4",  # 桃
}

def _text_color_for_bg(hexcode: str) -> str:
    """背景色に対して見やすい文字色(黒/白)を返す"""
    hexcode = hexcode.lstrip("#")
    try:
        r, g, b = [int(hexcode[i:i+2], 16) for i in (0, 2, 4)]
    except ValueError:
        return "#000000"
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if lum > 140 else "#FFFFFF"

def style_rank_table(df: pd.DataFrame):
    """枠セルを枠色で塗り、枠/馬番を整数表示にした Styler を返す"""
    d = df.copy()
    format_cols = []
    for col in ("枠", "馬番"):
        if col in d.columns:
            vals = pd.to_numeric(d[col], errors="coerce")
            d[col] = vals
            format_cols.append(col)

    def color_waku_col(s: pd.Series):
        out = []
        for v in s:
            if pd.isna(v):
                out.append("")
                continue
            key = str(int(round(float(v))))
            bg = WAKU_COLORS.get(int(key), "#FFFFFF")
            fg = _text_color_for_bg(bg)
            out.append(f"background-color:{bg}; color:{fg}; font-weight:700; text-align:center;")
        return out

    def fmt_int(x):
        return "" if pd.isna(x) else f"{int(round(float(x)))}"

    styler = d.style
    if format_cols:
        styler = styler.format({c: fmt_int for c in format_cols})
    if "枠" in d.columns:
        styler = styler.apply(color_waku_col, subset=["枠"])
    if "順位" in d.columns:
        def bold_top(s: pd.Series):
            out = []
            for v in s:
                try:
                    out.append("font-weight:700" if int(v) <= 7 else "")
                except Exception:
                    out.append("")
            return out
        styler = styler.apply(bold_top, subset=["順位"])
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler

def plot_scatter_waku(df: pd.DataFrame):
    """馬番×AR100 散布図（点色＝枠色, 凡例1～8, ラベル重なり回避のシンプル表示）"""
    d = df.copy()

    # 枠/馬番は整数化（表示もずれないように）
    for col in ("枠", "馬番"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").round(0).astype("Int64")

    # 枠色キー（文字列）を用意
    if "枠" in d.columns:
        d["枠_str"] = d["枠"].astype("Int64").astype(str)
    else:
        d["枠_str"] = pd.Series([""] * len(d))

    # X軸列の決定
    x_col = "馬番" if "馬番" in d.columns else (d.columns[0] if len(d.columns) else None)
    if x_col is None:
        return px.scatter()

    # ← ここがIndentationErrorの原因。改行して同じ階層に戻す
    hover_data = {"馬名": True, "AR100": ":.1f"}
    if "馬番" in d.columns:
        hover_data["馬番"] = True
    if "枠" in d.columns:
        hover_data["枠"] = True

    fig = px.scatter(
        d,
        x=x_col,
        y="AR100",
        color="枠_str",
        color_discrete_map={str(k): v for k, v in WAKU_COLORS.items()},  # 枠色と1対1対応
        category_orders={"枠_str": [str(i) for i in range(1, 9)]},
        hover_data=hover_data,
        labels={"枠_str": "枠"},
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="rgba(0,0,0,0.35)")))
    fig.update_layout(
        template="simple_white",
        legend_title_text="枠",
        xaxis=dict(title="馬番", tickmode="linear", dtick=1, tickfont=dict(size=12)),
        yaxis=dict(title="AR100", tickfont=dict(size=12)),
        margin=dict(l=40, r=20, t=10, b=40),
    )
    return fig

def plot_ar_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    name_col: str | None = None,
    waku_col: str | None = None,
    show_text: bool = False,
):
    """AR100 と任意の数値列を散布図で表示する（枠色・ラベル対応版）"""

    if x_col not in df.columns or y_col not in df.columns:
        return px.scatter()

    d = df.copy()

    # 数値列の前処理
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    x_numeric = pd.to_numeric(d[x_col], errors="coerce")
    if x_numeric.notna().any():
        d["_x_val"] = x_numeric
        x_for_plot = "_x_val"
    else:
        x_for_plot = x_col

    # 欠損を除外
    d = d.dropna(subset=[y_col])
    if d.empty:
        return px.scatter()

    color_key = None
    scatter_kwargs = {}

    # 枠色をカラーに使用
    if waku_col and waku_col in d.columns:
        d[waku_col] = pd.to_numeric(d[waku_col], errors="coerce").round(0).astype("Int64")
        d["_waku_str"] = d[waku_col].astype("Int64").astype(str)
        color_key = "_waku_str"
        scatter_kwargs.update(
            color=color_key,
            color_discrete_map={str(k): v for k, v in WAKU_COLORS.items()},
            category_orders={color_key: [str(i) for i in range(1, 9)]},
            labels={color_key: "枠"},
        )

    # Hover 情報を整理
    hover_data: dict[str, object] = {x_col: True, y_col: ":.1f"}
    if name_col and name_col in d.columns:
        hover_data[name_col] = True
    if waku_col and waku_col in d.columns:
        hover_data[waku_col] = True

    if show_text and name_col and name_col in d.columns:
        scatter_kwargs["text"] = name_col

    fig = px.scatter(
        d,
        x=x_for_plot,
        y=y_col,
        hover_data=hover_data,
        **scatter_kwargs,
    )

    if show_text and name_col and name_col in d.columns:
        fig.update_traces(textposition="top center", textfont=dict(size=12))

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="rgba(0,0,0,0.35)")))
    fig.update_layout(
        template="simple_white",
        margin=dict(l=40, r=20, t=10, b=40),
    )
    if color_key:
        fig.update_layout(legend_title_text="枠")

    # 軸ラベル・刻み幅を調整
    xaxis_conf = dict(title=x_col)
    if x_for_plot == "_x_val":
        xaxis_conf.update(tickmode="linear", dtick=1)
    fig.update_xaxes(**xaxis_conf)
    fig.update_yaxes(title=y_col)

    return fig

def build_marks_text(
    df: pd.DataFrame,
    *,
    name_col: str = "馬名",
    mark_col: str = "印",
    ar_col: str = "AR100",
    ar_label: str = "AR",
    ar_only_for_top: bool = False,
) -> str:
    """
    コピペ用の印テキストを作る（ARは小数2桁）
    出力順: ◎ → 〇 → ▲ → ☆ → △（'消'等は無視）
    
    Parameters
    ----------
    df : pd.DataFrame
        入力データ。
    name_col : str, default "馬名"
        馬名を示す列名。
    mark_col : str, default "印"
        印を示す列名。
    ar_col : str, default "AR100"
        AR 値を示す列名。
    ar_label : str, default "AR"
        AR 値のラベル接頭辞。
    ar_only_for_top : bool, default False
        True の場合は最上位の印（◎）に対してのみ AR を表示する。
    """
    required = {mark_col, name_col}
    if not required.issubset(df.columns):
        return ""
    has_ar = ar_col in df.columns
    order = ["◎", "〇", "▲", "☆", "△"]
    lines = []
    for sym in order:
        sub = df.loc[df[mark_col] == sym].copy()
        if "順位" in sub.columns:
            sub = sub.sort_values("順位")
        for _, r in sub.iterrows():
            name = str(r[name_col]).strip()
            show_ar = has_ar and (not ar_only_for_top or sym == order[0])
            line = f"{sym}{name}"
            if show_ar:
                try:
                    ar_val = float(r[ar_col])
                except Exception:
                    ar_val = float("nan")
                ar_txt = (
                    f"{ar_label}{ar_val:.2f}" if np.isfinite(ar_val) else f"{ar_label}--"
                )
                line = f"{line}　{ar_txt}"
            lines.append(line)
    return "\n".join(lines)

def render_final_view(df_ranked: pd.DataFrame):
    """最終一覧テーブル → 散布図 → コピペ用印 を一括表示"""
    st.subheader("🏁 最終一覧")
    st.dataframe(style_rank_table(df_ranked), use_container_width=True)

    fig = plot_scatter_waku(df_ranked)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("印（コピペ用）")
    st.code(build_marks_text(df_ranked), language="text")

# ===== 共通関数群 =====
def season_of(m: int) -> str:
    if 3 <= m <= 5:  return '春'
    if 6 <= m <= 8:  return '夏'
    if 9 <= m <= 11: return '秋'
    return '冬'

def z_score(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series([50] * len(s), index=s.index)
    return 50 + 10 * (s - s.mean()) / std

def _parse_time_to_sec(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    m = re.match(r'^(\d+):(\d+)\.(\d+)$', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + float('0.' + m.group(3))
    m = re.match(r'^(\d+)[\.:](\d+)[\.:](\d+)$', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10
    try:
        return float(s)
    except Exception:
        return np.nan

def _trim_name(x):
    try:
        return str(x).replace('\u3000', ' ').strip()
    except Exception:
        return str(x)

def w_std_unbiased(x, w, ddof=1):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    sw = w.sum()
    if not np.isfinite(sw) or sw <= 0:
        return np.nan
    m = np.sum(w * x) / sw
    var = np.sum(w * (x - m) ** 2) / sw
    n_eff = (sw ** 2) / np.sum(w ** 2) if np.sum(w ** 2) > 0 else 0
    if ddof and n_eff > ddof:
        var *= n_eff / (n_eff - ddof)
    return float(np.sqrt(max(var, 0.0)))

def ndcg_by_race(frame: pd.DataFrame, scores, k: int = 3) -> float:
    f = frame[['race_id', 'y']].copy().reset_index(drop=True)
    s = np.asarray(scores, float)
    if len(s) != len(f):
        s = s[:len(f)]
    vals = []
    for _, idx in f.groupby('race_id').groups.items():
        idx = np.asarray(list(idx), int)
        y_true = np.nan_to_num(f.loc[idx, 'y'].astype(float).to_numpy(), nan=0.0)
        y_pred = np.nan_to_num(s[idx].astype(float), nan=0.0)
        m = len(idx)
        if m == 0:
            continue
        if m == 1:
            vals.append(1.0 if y_true[0] > 0 else 0.0)
            continue
        kk = int(min(max(1, k), m))
        order = np.argsort(-y_pred)
        gains = (2.0 ** y_true[order] - 1.0)
        discounts = 1.0 / np.log2(np.arange(2, m + 2))
        dcg = float(np.sum(gains[:kk] * discounts[:kk]))
        order_best = np.argsort(-y_true)
        gains_best = (2.0 ** y_true[order_best] - 1.0)
        idcg = float(np.sum(gains_best[:kk] * discounts[:kk]))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else float('nan')

def safe_iso_predict(ir, p_vec: np.ndarray) -> np.ndarray:
    x = np.asarray(p_vec, float)
    x = np.nan_to_num(x, nan=1.0 / max(len(x), 1), posinf=1 - 1e-6, neginf=1e-6)
    x = np.clip(x, 1e-6, 1 - 1e-6)
    try:
        y = ir.predict(x)
        y = np.nan_to_num(y, nan=x.mean(), posinf=1 - 1e-6, neginf=1e-6)
        y = np.clip(y, 1e-6, 1 - 1e-6)
        s = y.sum()
        return (y / s) if s > 0 else x
    except Exception:
        return x

# ===== Bradley–Terry: ランキング→ペア勝敗→MM更新 =====
def _build_pairwise_from_ranks(df_rank: pd.DataFrame,
                               rid_col: str, name_col: str, rank_col: str,
                               weight_col: str | None = None):
    """
    各レース内の着順からペア勝敗カウント n_ij を構築。
    n_ij: i が j に勝った回数（重み付き）。対称な m_ij = n_ij + n_ji も返す。
    """
    # 名前→index
    names = pd.Index(sorted(df_rank[name_col].dropna().astype(str).unique()))
    idx = {n:i for i,n in enumerate(names)}
    n = len(names)
    if n == 0:
        return names, np.zeros((0,0)), np.zeros((0,0))

    N = np.zeros((n, n), float)  # n_ij
    for rid, g in df_rank.groupby(rid_col):
        g = g.dropna(subset=[name_col, rank_col]).copy()
        if g.empty:
            continue
        # 低い数値が上位という仮定（“確定着順”）
        g = g.sort_values(rank_col)
        # 重み（時系列など）を race 単位で持ちたいならここで race_w を作成
        if weight_col and weight_col in g.columns:
            w = float(pd.to_numeric(g[weight_col], errors='coerce').mean() or 1.0)
        else:
            w = 1.0

        # 上位は下位すべてに勝利
        order = g[name_col].astype(str).tolist()
        m = len(order)
        for a in range(m):
            ia = idx.get(order[a]); 
            if ia is None: 
                continue
            for b in range(a+1, m):
                ib = idx.get(order[b])
                if ib is None:
                    continue
                N[ia, ib] += w  # ia beats ib
    # 合計比較回数 m_ij
    M = N + N.T
    return names, N, M

def fit_bradley_terry(names: pd.Index, N: np.ndarray, M: np.ndarray,
                      max_iter: int = 200, tol: float = 1e-10):
    """
    Hunter(2004)のMM更新式：
      w_i^{new} = S_i / sum_j ( M_ij / (w_i + w_j) ),  S_i = sum_j N_ij
    初期は一様、スケーリング不定性は sum(w)=1 に規格化。
    """
    n = len(names)
    if n == 0:
        return pd.Series([], index=names, dtype=float)

    w = np.ones(n, float) / n
    # 勝数
    S = N.sum(axis=1)

    for _ in range(max_iter):
        wsum = w[:, None] + w[None, :]
        denom = (M / np.maximum(wsum, 1e-12)).sum(axis=1)
        w_new = S / np.maximum(denom, 1e-12)
        w_new = np.maximum(w_new, 1e-12)
        w_new /= w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=names, dtype=float)


from types import SimpleNamespace

def _canon_surface(s: str) -> str:
    s = str(s)
    if s.startswith("芝") or "turf" in s.lower(): return "芝"
    if s.startswith("ダ") or "dirt" in s.lower(): return "ダ"
    return s

def _fallback_geom(course_id: str, surface: str, distance_m: int, layout: str, rail: str):
    """
    最低限の属性だけを持つ簡易ジオメトリ（PhysS1 Sprint1に必要な分だけ）。
    ※ finish_grade_pct は不明⇒0%、start_to_first_turn_m は保守的に 800m
    """
    return SimpleNamespace(
        course_id=course_id,
        surface=surface,
        distance_m=int(distance_m),
        layout=layout,
        rail_state=rail,
        # Sprint1 で参照する属性
        track_width_min_m=25.0,
        track_width_max_m=25.0,
        num_turns=2,
        start_to_first_turn_m=800.0,
        finish_grade_pct=0.0,
    )

def resolve_course_geom(course_id: str, surface: str, distance_m: int, layout: str, rail: str,
                        *, dist_tol: int = 300, step: int = 100, allow_fallback: bool = True):
    """
    1) 指定そのまま → 2) 柵だけ変更 → 3) レイアウト変更 → 4) 距離±tol探索
    見つからなければ allow_fallback=True なら「簡易ジオメトリ」を返す。
    戻り値: (layout_ok, rail_ok, geom, used_distance, is_fallback)
    """
    surface = _canon_surface(surface)
    used_distance = int(distance_m)

    try:
        g = get_course_geom(course_id, surface, int(distance_m), layout, rail)
        if g is not None:
            return layout, rail, g, used_distance, False
    except Exception:
        pass

    for r in ["A", "B", "C", "D", ""]:
        if r == rail: 
            continue
        try:
            g = get_course_geom(course_id, surface, int(distance_m), layout, r)
            if g is not None:
                return layout, r, g, used_distance, False
        except Exception:
            continue

    cand_layouts = (LAYOUT_OPTS.get(course_id) if 'LAYOUT_OPTS' in globals() else ["内回り","外回り","直線"]) or ["内回り","外回り","直線"]
    for lay in cand_layouts:
        if lay == layout:
            continue
        for r in ["A", "B", "C", "D", ""]:
            try:
                g = get_course_geom(course_id, surface, int(distance_m), lay, r)
                if g is not None:
                    return lay, r, g, used_distance, False
            except Exception:
                continue

    dlist = [distance_m]
    for d in range(step, dist_tol + step, step):
        dlist.extend([distance_m - d, distance_m + d])
    dlist = [int(d) for d in dlist if 800 <= d <= 3600]

    for d2 in dlist:
        for lay in cand_layouts:
            for r in ["A", "B", "C", "D", ""]:
                try:
                    g = get_course_geom(course_id, surface, int(d2), lay, r)
                    if g is not None:
                        return lay, r, g, int(d2), False
                except Exception:
                    continue

    if allow_fallback and surface == "芝":
        # ← 最終手段：簡易ジオメトリで続行（京都3000mのような未登録距離対策）
        g = _fallback_geom(course_id, surface, int(distance_m), layout, rail)
        return layout, rail, g, int(distance_m), True

    return None, None, None, None, False




# ===== サイドバー =====
st.sidebar.title("⚙️ パラメタ設定（AUTO統合）")
MODE = st.sidebar.radio("モード", ["AUTO（推奨）","手動（上級者）"], index=0, horizontal=True)

# === 2歳戦モード UI ===
try:
    TWOYO_AUTO = detect_2yo_race(s1, s0)  # s1=出走表DF, s0=レース情報DF
except Exception:
    TWOYO_AUTO = False

TWOYO_MODE = False
USE_MC = True

with st.sidebar.expander("🍼 2歳戦モード", expanded=TWOYO_AUTO):
    TWOYO_MODE = st.checkbox(
        "2歳戦最適化（履歴依存を弱める）",
        value=TWOYO_AUTO,
        help="Turn/Dist/安定性など履歴依存の効きを抑え、スペクトル/調教/幾何を重視します。"
    )
    USE_MC = st.checkbox(
        "モンテカルロ（Pace/着順）を使う",
        value=not TWOYO_MODE,
        help="2歳戦ではOFF推奨。OFFならPacePtsを0固定・着順MCをスキップします。"
    )

with st.sidebar.expander("🔰 基本", expanded=True):
    lambda_part  = st.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
    grade_bonus  = st.slider("重賞実績ボーナス", 0, 20, 5)
    agari1_bonus = st.slider("上がり3F 1位ボーナス", 0, 10, 3)
    agari2_bonus = st.slider("上がり3F 2位ボーナス", 0, 5, 2)
    agari3_bonus = st.slider("上がり3F 3位ボーナス", 0, 3, 1)

with st.sidebar.expander("本レース条件", expanded=True):
    grade_opts = ["G1", "G2", "G3", "L", "OP", "3勝クラス"]
    TARGET_GRADE = st.selectbox("本レースの格", grade_opts, index=grade_opts.index("OP"))
    TARGET_SURFACE  = st.selectbox("本レースの馬場", ["芝","ダ"], index=0)
    TARGET_DISTANCE = st.number_input("本レースの距離 [m]", 1000, 3600, 1800, 100)
    
with st.sidebar.expander("📐 本レース幾何（コース設定）", expanded=True):
    VENUES = ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]
    COURSE_ID = st.selectbox("競馬場", VENUES, index=VENUES.index("東京"))

    LAYOUT_OPTS = {
        "札幌":["内回り"], "函館":["内回り"], "福島":["内回り"],
        "新潟":["内回り","外回り","直線"], "東京":["外回り"],
        "中山":["内回り","外回り"], "中京":["外回り"],
        "京都":["内回り","外回り"], "阪神":["内回り","外回り"], "小倉":["内回り"]
    }
    LAYOUT = st.selectbox("レイアウト", LAYOUT_OPTS[COURSE_ID], key="layout_select")
   
with st.sidebar.expander("📊 AR100調整", expanded=False):
    AR_MODE = st.radio(
        "AR100の計算モード",
        [
            "順位ベース（従来）",
            "スコアベース（Z値, 従来）",
            "厳格（erf＋尾部ブースト）",  # ← 新規
        ],
        index=2,
        help="厳格=フィールド内Zを強く圧縮し、さらに上位のごく一部だけロジスティックで100に近づく"
    )

    # 順位ベース用
    AR_RANK_GAMMA = st.slider("順位圧縮 γ（小=差縮む）", 0.60, 1.20, 0.85, 0.01)

    # 厳格スケール用（本体）
    AR_STRICT_S = st.slider("厳しさ s（大=さらに厳しい）", 0.60, 2.00, 1.10, 0.05)
    AR_BASE_A   = st.slider("基礎レンジ A（振幅）", 30.0, 48.0, 45.0, 0.5,
                            help="50±A*erf(...) の±A。A=45だとおおむね 5〜95 レンジ")
    # 不確実性ペナルティ
    AR_SHRINK_SIGMA = st.slider("不確実性ペナルティ（σ寄与）", 0.0, 2.0, 0.6, 0.05)
    AR_SHRINK_NEFF  = st.slider("データ薄さペナルティ（n_eff寄与）", 0.0, 2.0, 0.8, 0.05)

    # 尾部（トップだけ100に寄せる連続ブースト）
    TAIL_THRESH = st.slider("尾部発火しきい値 T（Z）", 1.8, 3.5, 2.8, 0.1,
                            help="このZを超えると上振れ側の100寄せがゆっくり効き始める（例: 2.8〜3.2）")
    TAIL_TAU    = st.slider("尾部の柔らかさ τ（大=なだらか）", 0.2, 1.2, 0.6, 0.05)
    TAIL_POW    = st.slider("尾部の厳しさ p（大=さらにレア）", 1.0, 4.0, 2.0, 0.1)

# ===== ここから スペクトル解析（FFT+DTW）を本線に統合 =====
with st.sidebar.expander("📡 スペクトル設定", expanded=True):
    spectral_weight_ui = st.slider("スペクトル適合係数", 0.0, 3.0, 1.0, 0.1)
    templ_tol_m = st.slider("テンプレ距離許容幅(±m)", 50, 400, 100, 25)

# ===== 物理（調教）ブロック =====
with st.sidebar.expander("🏇 物理（調教）", expanded=True):
    USE_PHYSICS = st.checkbox("物理ブロックを使う（調教×力学）", True)
    # スペクトル : 物理 の比率（合成の“配分”）
    spec_phys_ratio = st.slider("スペクトル : 物理 の比率", 0.0, 1.0, 0.6, 0.05)
    spec_ratio = float(spec_phys_ratio)
    phys_ratio = 1.0 - spec_ratio

    # 任意の初期値（効きが良い実戦値）
    Crr_wood = st.number_input("Crr（転がり抵抗）: ウッド", 0.0, 0.06, 0.020, 0.001, help="推奨: 0.020")
    Crr_hill = st.number_input("Crr（転がり抵抗）: 坂路", 0.0, 0.06, 0.014, 0.001, help="推奨: 0.014")
    CdA      = st.number_input("CdA（空力フロント[m²]）", 0.2, 1.6, 0.80, 0.05, help="推奨: 0.8")
    rho_air  = st.number_input("空気密度 ρ[kg/m³]", 0.8, 1.5, 1.20, 0.01)
    Pmax_wkg = st.number_input("最大発揮出力 Pmax[W/kg]", 10.0, 30.0, 20.0, 0.5)
    Emax_jkg = st.number_input("可用エネルギー Emax[J/kg/800m]", 600.0, 4000.0, 1800.0, 50.0)
    half_life_train_days = st.slider("調教寄与の半減期（日）", 3, 60, 18, 1)


with st.sidebar.expander("⚖️ 自動バランサ", expanded=False):
    USE_AUTO_BALANCER = st.checkbox(
        "自動バランサを有効化（FinalRawを再標準化し Spec/Phys を自動混合）",
        value=False,
        help="OFF推奨。ONにすると相対順位だけが効きやすくなり、特性重みの効きが小さく見えます。"
    )

    # 現在の設定で有効な柵だけに絞る
    surface_ui = "芝" if TARGET_SURFACE == "芝" else "ダ"
    dist_ui = int(TARGET_DISTANCE)

    valid_rails = []
    for r in ["A", "B", "C", "D", ""]:
        try:
            if get_course_geom(COURSE_ID, surface_ui, dist_ui, LAYOUT, r) is not None:
                valid_rails.append(r or "（指定なし）")
        except Exception:
            pass

    # 候補が無ければレイアウトの自動切替を試す
    if not valid_rails:
        switched = False
        for lay2 in (LAYOUT_OPTS.get(COURSE_ID) or ["内回り","外回り","直線"]):
            if lay2 == LAYOUT:
                continue
            vr2 = []
            for r in ["A","B","C","D",""]:
                try:
                    if get_course_geom(COURSE_ID, surface_ui, dist_ui, lay2, r) is not None:
                        vr2.append(r or "（指定なし）")
                except Exception:
                    pass
            if vr2:
                st.warning(f"選択レイアウト『{LAYOUT}』では登録が見つからないため『{lay2}』に切り替えました。")
                st.session_state['layout_select'] = lay2
                st.rerun()
                switched = True
                break

        if not switched:
            valid_rails = ["（指定なし）"]
            st.caption("※ この距離では登録された柵区分が見つかりません。上のスモークテストで利用可能な組合せを確認してください。")

    # ← ここからは常にウィジェットを出す（未定義防止）
    rail_label = st.selectbox("コース区分（A/B/C/D）", valid_rails, index=0, key="rail_select")
    RAIL = "" if rail_label == "（指定なし）" else rail_label

    DEFAULT_VENUE_TURN = {'札幌':'右','函館':'右','福島':'右','新潟':'左','東京':'左','中山':'右','中京':'左','京都':'右','阪神':'右','小倉':'右'}
    _turn_default = DEFAULT_VENUE_TURN.get(COURSE_ID, '右')
    TARGET_TURN = st.radio("回り", ["右","左"],
                           index=(0 if _turn_default == "右" else 1),
                           horizontal=True, key="turn_geom")

    TODAY_BAND = st.select_slider("通過帯域（暫定）", options=["内","中","外"], value="中", key="band_today")


with st.sidebar.expander("🧮 物理(Sprint1)の重み", expanded=True):
    PHYS_S1_GAIN = st.slider("PhysS1加点の強さ", 0.0, 3.0, 1.0, 0.1)

with st.sidebar.expander("🛠 安定化/補正", expanded=True):
    half_life_m  = st.slider("時系列半減期(月)", 0.0, 12.0, 6.0, 0.5)
    stab_weight  = st.slider("安定性(小さいほど◎)の係数", 0.0, 2.0, 0.7, 0.1)
    pace_gain    = st.slider("ペース適性係数", 0.0, 3.0, 1.0, 0.1)
    weight_coeff = st.slider("斤量ペナルティ強度(pts/kg)", 0.0, 4.0, 1.0, 0.1)
    
with st.sidebar.expander("🧩 特性重み（任意）", expanded=False):
    # ── 性別（0.00〜2.00、0.01刻み）──
    SEX_MALE  = st.slider("性別: 牡の加点", 0.0, 2.0, 0.0, 0.01, format="%.2f")
    SEX_FEMA  = st.slider("性別: 牝の加点", 0.0, 2.0, 0.0, 0.01, format="%.2f")
    SEX_GELD  = st.slider("性別: センの加点", 0.0, 2.0, 0.0, 0.01, format="%.2f")

    # ── 脚質（0.00〜2.00、0.01刻み）──
    STL_NIGE   = st.slider("脚質: 逃げの加点",  0.0, 2.0, 0.0, 0.01, format="%.2f")
    STL_SENKO  = st.slider("脚質: 先行の加点",  0.0, 2.0, 0.0, 0.01, format="%.2f")
    STL_SASHI  = st.slider("脚質: 差しの加点",  0.0, 2.0, 0.0, 0.01, format="%.2f")
    STL_OIKOMI = st.slider("脚質: 追込の加点",  0.0, 2.0, 0.0, 0.01, format="%.2f")

    # ── 年齢（ピークは整数のまま / 減衰強さは0.00〜2.00、0.01刻み）──
    AGE_PEAK   = st.slider("年齢のピーク（±で減衰）", 2, 8, 4)
    AGE_SLOPE  = st.slider("年齢の減衰強さ", 0.0, 2.0, 0.5, 0.01, format="%.2f")

    # ── 枠バイアス強さも0.00〜2.00、0.01刻みに（方向はそのまま）──
    WAKU_DIR   = st.radio("枠バイアス方向", ["なし","内有利","外有利"], index=0, horizontal=True)
    WAKU_STR   = st.slider("枠バイアス強さ", 0.0, 2.0, 1.0, 0.01, format="%.2f")

with st.sidebar.expander("📏 確率校正", expanded=False):
    do_calib = st.checkbox("等温回帰で勝率を校正", value=False)

with st.sidebar.expander("🎛 手動（上級者向け）", expanded=(MODE=="手動（上級者）")):
    besttime_w_manual = st.slider("ベストタイム重み(手動)", 0.0, 2.0, 1.0)
    dist_bw_m_manual  = st.slider("距離帯の幅[手動]", 50, 600, 200, 25)
    mc_beta_manual    = st.slider("PL温度β(手動)", 0.3, 5.0, 1.4, 0.1)

with st.sidebar.expander("🖥 表示", expanded=False):
    FULL_TABLE_VIEW = st.checkbox("全頭表示（スクロール無し）", True)
    MAX_TABLE_HEIGHT = st.slider("最大高さ(px)", 800, 10000, 5000, 200)
    SHOW_CORNER = st.checkbox("4角ポジション図を表示", False)


if st.button("🧪 PhysS1 スモークテスト"):
    surface_ui = "芝" if TARGET_SURFACE == "芝" else "ダ"
    dist_ui = int(TARGET_DISTANCE)

    lay_ok, rail_ok, geom, used_d, is_fb = resolve_course_geom(COURSE_ID, surface_ui, dist_ui, LAYOUT, RAIL)
    st.write("geom (resolved):", geom)
    st.caption(
        f"layout: {LAYOUT} → {lay_ok or '—'} / rail: {RAIL or '（指定なし）'} → {rail_ok or '（指定なし）'} / "
        f"distance: {dist_ui} → {used_d or '—'}" + (" 〔fallback: 簡易ジオメトリ〕" if is_fb else "")
    )



    if geom is None:
        st.error("この組合せのコース幾何が未登録のため PhysS1 を実行できません。下の一覧から存在する組合せを選んでください。")

        # 近傍距離でも探して一覧に出す（±300m）
        avail = []
        for d2 in range(max(800, dist_ui-300), min(3600, dist_ui+300)+1, 100):
            for lay in (LAYOUT_OPTS.get(COURSE_ID) or ["内回り","外回り","直線"]):
                for r in ["A","B","C","D",""]:
                    try:
                        g = get_course_geom(COURSE_ID, surface_ui, int(d2), lay, r)
                        if g is not None:
                            avail.append({"距離[m]": d2, "レイアウト": lay, "柵": (r or "（指定なし）")})
                    except Exception:
                        pass
        if avail:
            st.dataframe(pd.DataFrame(avail).drop_duplicates().sort_values(["距離[m]","レイアウト","柵"]))
        else:
            st.info("※ 近傍距離(±300m)でも登録が見つかりません。別の距離・コースを選んでください。")
        st.stop()
    # ここまで来たら geom あり → 従来通り DataFrame 作成 & 実行
    ...


    # ここまで来たら geom がある
    races_df_today_dbg = pd.DataFrame([{
        'race_id':'DBG',
        'course_id':COURSE_ID,
        'surface': surface_ui,
        'distance_m': dist_ui,
        'layout': lay_ok,
        'rail_state': rail_ok,
        'band': TODAY_BAND,
        'num_turns': 2
    }])

    try:
        out = add_phys_s1_features(races_df_today_dbg, group_cols=(), band_col="band", verbose=True)
        st.write("phys列:", [c for c in out.columns if c.startswith("phys_")])
        st.dataframe(out)
    except Exception as e:
        st.error(f"PhysS1失敗: {e}")



# ===== ファイルアップロード =====
topbar("Rikeiba", "①Excel読込 → ②調整 → ③分析 → ④結果出力（JSON）")
st.markdown("### Excelアップロード（sheet0=過去走 / sheet1=出走表）")
excel_file = st.file_uploader("Excel（.xlsx）", type=['xlsx'], key="excel_up")
if excel_file is None:
    st.info("まずExcelをアップロードしてください。")
    st.stop()

st.subheader("（任意）調教ファイルアップロード")
wood_file = st.file_uploader("ウッドチップ調教（.xlsx）", type=['xlsx'], key="wood_x")
hill_file = st.file_uploader("坂路調教（.xlsx）", type=['xlsx'], key="hill_x")


@st.cache_data(show_spinner=False)
def load_excel_bytes(content: bytes):
    xls = pd.ExcelFile(io.BytesIO(content))
    s0 = pd.read_excel(xls, sheet_name=0)
    s1 = pd.read_excel(xls, sheet_name=1)
    # 3枚目があれば読み込む（無ければ空DF）
    try:
        s2 = pd.read_excel(xls, sheet_name=2)
    except Exception:
        s2 = pd.DataFrame()
    return s0, s1, s2

sheet0, sheet1, sheet2 = load_excel_bytes(excel_file.getvalue())


# ===== 調教データ読み込み＆正規化 =====
def _read_train_xlsx(file, kind: str) -> pd.DataFrame:
    """
    調教Excel（複数シート可）から下記列を抽出して統一する:
      - 馬名(str), 日付(datetime64), 場所(str/任意), _kind('wood'|'hill'),
        _intensity(str/任意), _lap_sec(list[4] of float)
    ・Lap1..Lap4 or Time1..Time4（区間）優先。なければ 4F/3F/2F/1F（累計）→差分化。
    ・「12.1-11.8-…」等の文字列も許容。
    ・末尾NaNは最後の有効値で前方補完。
    """
    import numpy as np, pandas as pd, io, re

    if file is None:
        return pd.DataFrame()

    try:
        xls = pd.ExcelFile(io.BytesIO(file.getvalue()))
    except Exception:
        return pd.DataFrame()

    def _norm_cols(d: pd.DataFrame) -> pd.DataFrame:
        return d.rename(columns=lambda c: str(c).strip())

    name_pat = r'馬名|名前|出走馬|horse|Horse'
    date_pat = r'日付|年月日|調教日|日時|実施日|測定日|記録日|date|Date'

    def _to_num_like(s):
        ser = pd.Series(s).astype(str).str.replace(',', '').str.replace('\u3000', ' ').str.strip()
        num = ser.str.extract(r'([-+]?\d+(?:\.\d+)?)', expand=False)
        return pd.to_numeric(num, errors='coerce')

    def _smart_parse_date(col: pd.Series) -> pd.Series:
        s = col
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            s = s.astype('Int64').astype(str).str.replace('<NA>', '')
        s = s.astype(str).str.strip()
        msk = s.str.match(r'^\d{8}$')
        out = pd.to_datetime(
            s.where(~msk, s.str.slice(0,4)+'-'+s.str.slice(4,6)+'-'+s.str.slice(6,8)),
            errors='coerce'
        )
        out = out.fillna(pd.to_datetime(s, errors='coerce'))
        return out

    def _split4_from_cum(t4, t3, t2, t1):
        vals = [np.nan, np.nan, np.nan, np.nan]
        vals[3] = t1
        if np.isfinite(t4) and np.isfinite(t3):
            vals[0] = t4 - t3
        elif np.isfinite(t4) and np.isfinite(t1):
            vals[0] = (t4 - t1) / 3.0
        if np.isfinite(t3) and np.isfinite(t2):
            vals[1] = t3 - t2
        elif np.isfinite(t3) and not np.isfinite(t2):
            base = t1 if np.isfinite(t1) else t3 / 3.0
            vals[1] = (t3 - base) / 2.0
        elif np.isfinite(t4) and np.isfinite(t1) and not np.isfinite(t3):
            vals[1] = (t4 - t1) / 3.0
        if np.isfinite(t2) and np.isfinite(t1):
            vals[2] = t2 - t1
        elif np.isfinite(t3) and not np.isfinite(t2):
            base = t1 if np.isfinite(t1) else t3 / 3.0
            vals[2] = (t3 - base) / 2.0
        elif np.isfinite(t4) and np.isfinite(t1):
            vals[2] = (t4 - t1) / 3.0

        arr = np.array(vals, float)
        if np.isnan(arr).any():
            good = np.where(np.isfinite(arr))[0]
            if good.size:
                last = arr[good[-1]]
                arr = np.where(np.isfinite(arr), arr, last)
        return arr

    def _parse_one(df_in: pd.DataFrame) -> pd.DataFrame:
        df = _norm_cols(df_in.copy())

        name_col = next((c for c in df.columns if re.search(name_pat, c, flags=re.I)), None)
        date_col = next((c for c in df.columns if re.search(date_pat, c, flags=re.I)), None)
        if name_col is None or date_col is None:
            return pd.DataFrame()

        df['馬名'] = df[name_col].astype(str).str.replace('\u3000',' ').str.strip()
        df['日付'] = _smart_parse_date(df[date_col])

        seg_cols = []
        for i in range(1, 5):
            cand = [c for c in df.columns if re.search(fr'(^|\b)Lap{i}(\b|$)', c, flags=re.I)]
            if not cand:
                cand = [c for c in df.columns if re.search(fr'(^|\b)(L|Time){i}(\b|$)', c, flags=re.I)]
            seg_cols.append(cand)
        has_segment = all(len(x) > 0 for x in seg_cols)

        def _find_first(pats):
            for p in pats:
                cc = [c for c in df.columns if re.search(p, c, flags=re.I)]
                if cc:
                    return cc[0]
            return None

        c4 = _find_first([r'(^|\b)4\s*[fＦｆ].{0,3}(時計|ﾀｲﾑ|秒)?(\b|$)', r'800\s*m', r'(^|[^0-9])4Ｆ'])
        c3 = _find_first([r'(^|\b)3\s*[fＦｆ].{0,3}(時計|ﾀｲﾑ|秒)?(\b|$)', r'600\s*m', r'(^|[^0-9])3Ｆ'])
        c2 = _find_first([r'(^|\b)2\s*[fＦｆ].{0,3}(時計|ﾀｲﾑ|秒)?(\b|$)', r'400\s*m', r'(^|[^0-9])2Ｆ'])
        c1 = _find_first([r'(^|\b)1\s*[fＦｆ].{0,3}(時計|ﾀｲﾑ|秒)?(\b|$)', r'200\s*m', r'(ﾗｽﾄ|末|上がり).{0,3}1\s*[fＦｆ]'])
        has_cum = any([c4, c3, c2, c1])

        laps = np.full((len(df), 4), np.nan, float)

        if has_segment:
            for i in range(4):
                laps[:, i] = _to_num_like(df[seg_cols[i][0]]).to_numpy(float)
            if np.all(np.diff(laps, axis=1) < 0):
                t4, t3, t2, t1 = laps.T
                laps[:, 0] = t4 - t3
                laps[:, 1] = t3 - t2
                laps[:, 2] = t2 - t1
                laps[:, 3] = t1
        elif has_cum:
            T4 = _to_num_like(df[c4]) if c4 else pd.Series(np.nan, index=df.index)
            T3 = _to_num_like(df[c3]) if c3 else pd.Series(np.nan, index=df.index)
            T2 = _to_num_like(df[c2]) if c2 else pd.Series(np.nan, index=df.index)
            T1 = _to_num_like(df[c1]) if c1 else pd.Series(np.nan, index=df.index)
            arr = []
            for i in range(len(df)):
                arr.append(_split4_from_cum(T4.iloc[i], T3.iloc[i], T2.iloc[i], T1.iloc[i]))
            laps = np.vstack(arr)
        else:
            str_col = next(
                (c for c in df.columns if re.search(r'ラップ|区間|時計|タイム', c) and df[c].astype(str).str.contains('-').any()),
                None
            )
            if str_col:
                def _parse_seq(s):
                    xs = re.findall(r'(\d+(?:\.\d+)?)', str(s))
                    xs = [float(x) for x in xs[:4]]
                    while len(xs) < 4:
                        xs.insert(0, np.nan)
                    return xs[-4:]
                laps = np.vstack(df[str_col].apply(_parse_seq).to_list()).astype(float)
            else:
                return pd.DataFrame()

        st_col = next((c for c in df.columns if re.search(r'強弱|内容|馬なり|一杯|強め|仕掛け|軽め|流し', c)), None)
        intensity = df[st_col].astype(str) if st_col else pd.Series([''] * len(df), index=df.index)

        place_col = next((c for c in df.columns if re.search(r'場所|所属|トレセン|美浦|栗東', c)), None)

        out = pd.DataFrame({
            '馬名': df['馬名'],
            '日付': df['日付'],
            '場所': (df[place_col].astype(str) if place_col else ""),
            '_kind': kind,
            '_intensity': intensity,
            '_lap_sec': list(laps)
        })

        mask = np.isfinite(laps).any(axis=1)
        out = out[mask].dropna(subset=['馬名', '日付'])
        return out

    frames = []
    for sh in xls.sheet_names:
        try:
            df0 = pd.read_excel(xls, sheet_name=sh, header=0)
            parsed = _parse_one(df0)
            if not parsed.empty:
                frames.append(parsed)
        except Exception:
            continue

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



# ===== コース断面（坂路の傾斜プロファイル） =====
def _slope_profile(kind: str):
    # 返り値: list of (length_m, grade_perm) を距離順に
    if kind == 'hill_ritto':
        # 栗東：300m 2.0‰? → 2.0%, 続いて 570m 3.5%, 100m 4.5%, 115m 1.25%
        # 単位は「%」＝ 0.02 など
        return [(300, 0.020), (570, 0.035), (100, 0.045), (115, 0.0125)]
    elif kind == 'hill_miho':
        # 美浦：主計測800m区間の代表値として 3.0% 近傍を一定とみなす（終端付近は4.688%）
        # 800mを 600m@3.0% + 200m@4.7% に
        return [(600, 0.030), (200, 0.04688)]
    else:
        return [(800, 0.000)]  # フラット

def _intensity_gain(txt: str) -> float:
    s = str(txt)
    if re.search(r'一杯|強め', s): return 1.12
    if re.search(r'強', s):       return 1.08
    if re.search(r'馬なり', s):   return 0.94
    if re.search(r'軽め|流し', s):return 0.90
    return 1.00

def _seg_energy_wkg(v, a, g, grade, Crr, CdA, rho):
    # v[m/s], a[m/s2], 体重は後で掛けるので W/kg と J/kg で返す
    # 抵抗：転がり + 重力 + 空力
    Fr = Crr * 9.80665 * np.cos(np.arctan(grade))            # ≈ Crr*g
    Fg = g * grade                                           # g*sinθ ≒ g*grade
    Fa = 0.5 * rho * CdA * v*v / 500.0                       # 係数縮尺（群れ流・姿勢緩和の平均的低減）
    P_over_m = v*(Fr + Fg) + v*a                             # W/kg
    P_over_m += Fa*v/75.0                                    # 空力寄与を弱めに（実測合わせ）
    return max(P_over_m, 0.0)

def _derive_training_metrics(train_df: pd.DataFrame,
                             s0_races: pd.DataFrame,
                             Crr_wood, Crr_hill, CdA, rho,
                             Pmax_wkg, Emax_jkg, half_life_days: int):
    """
    各調教1本→ EAP[J/kg/m], PeakWkg[W/kg], EffReserve を計算。
    馬体重は「その調教日の直近“前”レースの馬体重」を参照（なければ全体中央値）。
    4F×200m想定。坂路はプロファイルに沿って grade を付与。
    """
    if train_df.empty:
        return pd.DataFrame(columns=['馬名','日付','EAP','PeakWkg','EffReserve','PhysicsZ'])

    # 参照体重マップ
    bw_map = {}
    if {'馬名','レース日','馬体重'}.issubset(s0_races.columns):
        tmp = s0_races[['馬名','レース日','馬体重']].dropna().sort_values(['馬名','レース日'])
        for name, g in tmp.groupby('馬名'):
            bw_map[name] = list(zip(g['レース日'].to_numpy(), g['馬体重'].to_numpy()))
    bw_median = float(pd.to_numeric(s0_races.get('馬体重', pd.Series([480])), errors='coerce')
                      .median(skipna=True) or 480.0)

    out = []
    for _, r in train_df.iterrows():
        name = r['馬名']
        day  = pd.to_datetime(r['日付'])

        laps = np.array(r['_lap_sec'], dtype=float)
        if laps.size != 4 or np.isnan(laps).all():
            continue

        # 欠損は「最後に観測できた区間」で前方補完
        laps = np.where(np.isfinite(laps), laps, np.nan)
        if np.isnan(laps).any():
            good = np.where(np.isfinite(laps))[0]
            if good.size == 0:
                continue
            fill_val = float(laps[good[-1]])
            laps = np.where(np.isfinite(laps), laps, fill_val)

        # 調教日の直前レースの体重（無ければ全体中央値）
        bw = bw_median
        if name in bw_map:
            prev = [w for (d, w) in bw_map[name] if pd.to_datetime(d) <= day]
            if prev:
                bw = float(pd.to_numeric(prev[-1], errors='coerce') or bw_median)

        # 速度・加速度（200mごと）
        d = 200.0
        v = d / laps
        a = np.diff(v, prepend=v[0]) / laps

        # コース種別（坂路/ウッド）と勾配
        kind  = str(r.get('_kind', 'wood'))
        place = str(r.get('場所', ''))
        if kind == 'hill':
            is_miho  = bool(re.search(r'美浦|miho', place, flags=re.I))
            prof_key = 'hill_miho' if is_miho else 'hill_ritto'
            prof = _slope_profile(prof_key)

            grades = []
            remain = 800.0
            for L, G in prof:
                take = min(L, remain)
                nseg = int(round(take / 200.0))
                grades += [G] * max(1, nseg)
                remain -= take
                if remain <= 0:
                    break
            if not grades:
                grades = [0.03] * 4
            while len(grades) < 4:
                grades.append(grades[-1])
            grade = np.array(grades[:4], float)
            Crr = Crr_hill
        else:
            grade = np.zeros(4, float)
            Crr   = Crr_wood

        # 強弱による係数
        gain = _intensity_gain(r.get('_intensity', ''))

        # 区間あたりの出力密度（W/kg）
        P = np.array([_seg_energy_wkg(v[i], a[i], 9.80665, grade[i], Crr, CdA, rho) * gain
                      for i in range(4)], float)

        # 指標
        PeakWkg    = float(P.max())
        work_jkg   = float(np.sum(P * laps))               # J/kg
        EAP        = float(work_jkg / 800.0)               # J/kg/m
        EffReserve = float(max(0.0, Emax_jkg - work_jkg)) / Emax_jkg  # 0..1

        out.append({'馬名': name, '日付': day,
                    'EAP': EAP, 'PeakWkg': PeakWkg, 'EffReserve': EffReserve})

    df = pd.DataFrame(out)
    if df.empty:
        return df

    # 直近重み付け → PhysicsZ
    df = df.sort_values(['馬名','日付'])
    today = pd.Timestamp.today()
    df['_w'] = 0.5 ** ((today - df['日付']).dt.days.clip(lower=0) / float(max(1, half_life_days)))

    agg = (df.groupby('馬名')
             .apply(lambda g: pd.Series({
                 'EAP':        np.average(g['EAP'],        weights=g['_w']),
                 'PeakWkg':    np.average(g['PeakWkg'],    weights=g['_w']),
                 'EffReserve': np.average(g['EffReserve'], weights=g['_w']),
             }))
             .reset_index())

    # 「小さいほど良い」EAP を反転してZ化（平均50, σ10）
    agg['PhysicsCore'] = -pd.to_numeric(agg['EAP'], errors='coerce')
    mu = float(agg['PhysicsCore'].mean())
    sd = float(agg['PhysicsCore'].std(ddof=0) or 1.0)
    agg['PhysicsZ'] = (agg['PhysicsCore'] - mu) / sd * 10 + 50

    return agg[['馬名','EAP','PeakWkg','EffReserve','PhysicsZ']]


# ===== 列マッピング（軽量） =====

def _norm_col(s: str) -> str:
    s=str(s).strip(); s=re.sub(r'\s+','',s)
    return s.translate(str.maketrans('０１２３４５６７８９％','0123456789%')).replace('（','(').replace('）',')')

def _auto_pick(df, pats):
    cmap={c:_norm_col(c) for c in df.columns}
    for c, n in cmap.items():
        for p in pats:
            if re.search(p, n, flags=re.I):
                return c
    return None

def _map_ui(df, patterns, required, title, key_prefix):
    cols = list(df.columns)
    auto = {k: _auto_pick(df, pats) for k, pats in patterns.items()}
    with st.expander(f"列マッピング：{title}", expanded=False):
        mapping = {}
        for k, _ in patterns.items():
            options = ['<未選択>'] + cols
            default = auto.get(k) if auto.get(k) in cols else '<未選択>'
            mapping[k] = st.selectbox(k, options, index=options.index(default), key=f"map:{key_prefix}:{k}")
    for k in required:
        if mapping.get(k, '<未選択>') == '<未選択>':
            st.error(f"{title} 必須列が未選択: {k}")
            st.stop()
    return {k: (v if v != '<未選択>' else None) for k, v in mapping.items()}

# ここに PCI / PCI3 / Ave-3F も拾うキーを追加
# --- ① 列名の正規化（全角→半角、F/ハイフン統一）を先に ---
def _normalize_cols(df):
    trans = str.maketrans({
        '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
        'Ｆ':'F','ｆ':'f','ー':'-','－':'-','　':' '
    })
    out = df.copy()
    out.columns = [str(c).translate(trans).strip() for c in out.columns]
    return out

sheet0 = _normalize_cols(sheet0)

# === sheet0（過去走）を先に正規化して必須列を作る ===
def _normalize_sheet0(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()

    # 3) 走破タイム → 秒（例: "1:33.5" / "33.5"）
    if '走破タイム秒' not in s.columns and '走破タイム' in s.columns:
        def _to_sec(x):
            if pd.isna(x): return pd.NA
            t = str(x).strip()
            # mm:ss.s / m:ss.s / ss.s に対応
            if ':' in t:
                mm, ss = t.split(':', 1)
                try:
                    return float(ss) + 60.0 * float(mm)
                except:
                    return pd.NA
            try:
                return float(t)
            except:
                return pd.NA
        s['走破タイム秒'] = s['走破タイム'].map(_to_sec)

    if 'レース日' in s.columns:
        ser = s['レース日']
        if pd.api.types.is_numeric_dtype(ser):
            # Excelシリアル対策
            s['レース日'] = pd.to_datetime(ser, unit='d', origin='1899-12-30', errors='coerce')
        else:
            s['レース日'] = pd.to_datetime(ser, errors='coerce')
        # tz 無しに統一（混在対策）
        try:  s['レース日'] = s['レース日'].dt.tz_convert(None)
        except: pass
        try:  s['レース日'] = s['レース日'].dt.tz_localize(None)
        except: pass

    return s
# ここで正規化
sheet0 = _normalize_sheet0(sheet0)

# （以降はそのまま君のマッピングロジックでOK）
PAT_S0 = {
    '馬名':[r'馬名|名前|出走馬'],
    'レース日':[r'レース日|日付(?!S)|年月日|施行日|開催日'],
    '競走名':[r'競走名|レース名|名称'],
    'クラス名':[r'クラス名|格|条件|レースグレード'],
    '頭数':[r'頭数|出走頭数'],
    '確定着順':[r'確定着順|着順(?!率)'],
    '枠':[r'枠|枠番'],
    '番':[r'馬番|番'],
    '斤量':[r'斤量'],
    '馬体重':[r'馬体重|体重'],
    '上がり3Fタイム':[r'上がり3Fタイム|上がり3F|上3F'],
    '上3F順位':[r'上がり3F順位|上3F順位'],
    '通過4角':[r'通過.*4角|4角.*通過|第4コーナー|4角通過順'],
    '性別':[r'性別'],
    '年齢':[r'年齢|馬齢'],
    '走破タイム秒':[r'走破タイム.*秒|走破タイム|タイム$'],
    '距離':[r'距離'],
    '芝・ダ':[r'芝.?・.?ダ|芝ダ|コース|馬場種別|Surface'],
    '馬場':[r'馬場(?!.*指数)|馬場状態'],
    '場名':[r'場名|場所|競馬場|開催(地|場|場所)'],
    'PCI':[r'\bPCI(?!G)|ＰＣＩ'],
    'PCI3':[r'\bPCI3\b|ＰＣＩ3'],
    'Ave-3F':[r'Ave[-_]?3F|平均.*3F'],
}
REQ_S0 = ['馬名','レース日','競走名','頭数','確定着順']

MAP_S0 = {k: _auto_pick(sheet0, v) for k,v in PAT_S0.items()}
missing = [k for k in REQ_S0 if MAP_S0.get(k) is None or sheet0[MAP_S0[k]].isna().all()]
if missing:
    MAP_S0 = _map_ui(sheet0, PAT_S0, REQ_S0, 'sheet0（過去走）', 's0')

s0 = pd.DataFrame({k: sheet0[col] for k,col in MAP_S0.items() if col in sheet0.columns})



# --- ④ s0 の作成（合成列をここで実体化） ---
s0 = pd.DataFrame()
for k, col in MAP_S0.items():
    if col == '__SYNTH_YMD__':
        y = pd.to_numeric(sheet0['年'], errors='coerce').astype('Int64')
        m = pd.to_numeric(sheet0['月'], errors='coerce').astype('Int64')
        d = pd.to_numeric(sheet0['日'], errors='coerce').astype('Int64')
        s0['レース日'] = pd.to_datetime(
            y.astype(str) + '-' + m.astype(str) + '-' + d.astype(str),
            errors='coerce'
        )
    elif col and col in sheet0.columns:
        s0[k] = sheet0[col]

# --- ⑤ 走破タイム → 秒 に正規化（元が mm:ss.s 文字列想定） ---
if '走破タイム秒' not in s0.columns and '走破タイム' in sheet0.columns:
    import re, numpy as np
    def _to_sec(x):
        s = str(x)
        m = re.match(r'(?:(\d+):)?(\d+)(?:\.(\d+))?$', s)
        if not m: return np.nan
        mm = int(m.group(1) or 0)
        ss = int(m.group(2) or 0)
        frac = float('0.' + (m.group(3) or '0'))
        return mm*60 + ss + frac
    s0['走破タイム秒'] = pd.to_numeric(sheet0['走破タイム'], errors='coerce')
    mask = s0['走破タイム秒'].isna()
    if mask.any():
        s0.loc[mask, '走破タイム秒'] = sheet0.loc[mask, '走破タイム'].map(_to_sec)

# --- ⑥ 型の整備（数値化できるもの） ---
for col in ['頭数','枠','番','斤量','馬体重','上3F順位']:
    if col in s0.columns:
        s0[col] = pd.to_numeric(s0[col], errors='coerce')


# シート1
PAT_S1={
    '馬名':[r'馬名|名前|出走馬'],
    '枠':[r'枠|枠番'],
    '番':[r'馬番|番'],
    '性別':[r'性別'],
    '年齢':[r'年齢|馬齢'],
    '斤量':[r'斤量'],
    '馬体重':[r'馬体重|体重'],
    '脚質':[r'脚質'],
    '勝率':[r'勝率(?!.*率)|\b勝率\b'],
    '連対率':[r'連対率|連対'],
    '複勝率':[r'複勝率|複勝'],
    'ベストタイム':[r'ベスト.*タイム|Best.*Time|ﾍﾞｽﾄ.*ﾀｲﾑ|タイム.*(最速|ベスト)'],
}
REQ_S1=['馬名','枠','番']  # ← 性別/年齢は必須から外す
MAP_S1 = {k: _auto_pick(sheet1, v) for k,v in PAT_S1.items()}
miss1=[k for k in REQ_S1 if MAP_S1.get(k) is None]
if miss1:
    MAP_S1=_map_ui(sheet1, PAT_S1, REQ_S1, 'sheet1（出走表）', 's1')

s1=pd.DataFrame()
for k, col in MAP_S1.items():
    if col and col in sheet1.columns:
        s1[k]=sheet1[col]

for c in ['枠','番','斤量','馬体重']:
    if c in s1: s1[c]=pd.to_numeric(s1[c], errors='coerce')
if 'ベストタイム' in s1: s1['ベストタイム秒']=s1['ベストタイム'].apply(_parse_time_to_sec)

# 先頭空行/空馬名除去
s1 = s1.replace(r'^\s*$', np.nan, regex=True).dropna(how='all')
if '馬名' in s1:
    s1['馬名']=s1['馬名'].astype(str).str.replace('\u3000',' ').str.strip()
    s1=s1[s1['馬名'].ne('')]
s1=s1.reset_index(drop=True)

# === NEW: sheet2 から 性別/年齢 を補完 ===
if sheet2 is not None and isinstance(sheet2, pd.DataFrame) and not sheet2.empty:
    PAT_S2 = {
        '馬名':[r'馬名|名前|出走馬|Horse|horse'],
        '性別':[r'性別'],
        '年齢':[r'年齢|馬齢']
    }
    MAP_S2 = {k: _auto_pick(sheet2, v) for k, v in PAT_S2.items()}
    s2 = pd.DataFrame()
    for k, col in MAP_S2.items():
        if col and col in sheet2.columns:
            s2[k] = sheet2[col]

    if not s2.empty and '馬名' in s2.columns:
        s2['馬名'] = s2['馬名'].astype(str).str.replace('\u3000', ' ').str.strip()
        if '年齢' in s2.columns:
            s2['年齢'] = pd.to_numeric(s2['年齢'], errors='coerce')

        s1 = s1.merge(s2[['馬名','性別','年齢']].drop_duplicates('馬名'), on='馬名', how='left', suffixes=('', '_s2'))
        if '性別_s2' in s1.columns:
            s1['性別'] = s1['性別'].where(s1['性別'].astype(str).str.strip().ne(''), s1['性別_s2'])
            s1.drop(columns=['性別_s2'], inplace=True)
        if '年齢_s2' in s1.columns:
            s1['年齢'] = s1['年齢'].fillna(s1['年齢_s2'])
            s1.drop(columns=['年齢_s2'], inplace=True)


# 脚質エディタ
if '脚質' not in s1.columns: s1['脚質']=''
if '斤量' not in s1.columns: s1['斤量']=np.nan
if '馬体重' not in s1.columns: s1['馬体重']=np.nan

st.subheader("馬一覧（必要なら脚質/斤量/体重を調整）")

def auto_style_from_history(df: pd.DataFrame, n_recent=5, hl_days=180):
    # 必須列がなければ空で返す（落ちないように）
    need = {'馬名','レース日','頭数','通過4角'}
    if not need.issubset(df.columns):
        return pd.DataFrame({'馬名': [], '推定脚質': []})

    # 任意列（上3F順位）は存在チェックしてから選ぶ
    base_cols = ['馬名','レース日','頭数','通過4角']
    if '上3F順位' in df.columns:
        base_cols.append('上3F順位')

    t = (
        df[base_cols]
        .dropna(subset=['馬名','レース日','頭数','通過4角'])
        .copy()
    )

    # 並べ替えと最近n件
    t = t.sort_values(['馬名','レース日'], ascending=[True, False])
    t['_rn'] = t.groupby('馬名').cumcount() + 1
    t = t[t['_rn'] <= n_recent].copy()

    today = pd.Timestamp.today()
    t['_days'] = (today - pd.to_datetime(t['レース日'], errors='coerce')).dt.days.clip(lower=0).fillna(9999)
    t['_w'] = 0.5 ** (t['_days'] / float(hl_days))

    # 4角位置→先行度（0〜1）
    denom = (pd.to_numeric(t['頭数'], errors='coerce') - 1).replace(0, np.nan)
    pos_ratio = (pd.to_numeric(t['通過4角'], errors='coerce') - 1) / denom
    pos_ratio = pos_ratio.clip(0, 1).fillna(0.5)

    # 上がり順位があれば終い脚寄与、なければ0
    if '上3F順位' in t.columns:
        ag = pd.to_numeric(t['上3F順位'], errors='coerce')
        close = ((3.5 - ag) / 3.5).clip(0, 1).fillna(0.0)
    else:
        close = pd.Series(0.0, index=t.index)

    # ロジット
    b = {'逃げ': -1.2, '先行': 0.6, '差し': 0.3, '追込': -0.7}
    t['L_逃げ'] = b['逃げ'] + 1.6*(1 - pos_ratio) - 1.2*close
    t['L_先行'] = b['先行'] + 1.1*(1 - pos_ratio) - 0.1*close
    t['L_差し'] = b['差し'] + 1.1*(pos_ratio)     + 0.9*close
    t['L_追込'] = b['追込'] + 1.6*(pos_ratio)     + 0.5*close

    # 馬ごと重み平均 → softmax → 最頻脚質
    rows = []
    for name, g in t.groupby('馬名'):
        w = g['_w'].to_numpy(); sw = w.sum()
        if sw <= 0: 
            continue
        vec = np.array([
            float((g['L_逃げ']*w).sum()/sw),
            float((g['L_先行']*w).sum()/sw),
            float((g['L_差し']*w).sum()/sw),
            float((g['L_追込']*w).sum()/sw),
        ])
        vec = vec - vec.max()
        p = np.exp(vec); p /= p.sum()
        rows.append([name, STYLES[int(np.argmax(p))]])
    return pd.DataFrame(rows, columns=['馬名','推定脚質'])

pred_style = auto_style_from_history(s0.copy())

s1['脚質']=s1['脚質'].map(normalize_style)
if not pred_style.empty:
    s1=s1.merge(pred_style, on='馬名', how='left')
    s1['脚質']=s1['脚質'].where(s1['脚質'].astype(str).str.strip().ne(''), s1['推定脚質'])
    s1.drop(columns=['推定脚質'], inplace=True)

H = (lambda n: int(min(MAX_TABLE_HEIGHT, 38 + 35*max(1,int(n)) + 28)) if FULL_TABLE_VIEW else 460)
edit = st.data_editor(
    s1[['枠','番','馬名','性別','年齢','脚質','斤量','馬体重']].copy(),
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=STYLES),
        '斤量': st.column_config.NumberColumn('斤量', min_value=45, max_value=65, step=0.5),
        '馬体重': st.column_config.NumberColumn('馬体重', min_value=300, max_value=600, step=1),
    },
    use_container_width=True,
    num_rows='static',
    height=H(len(s1)),
    hide_index=True,
)
horses = edit.copy()

# === NEW: sheet0 から 勝率/連対率/複勝率 を自動集計（履歴ベース） ===
rate_src = s0.dropna(subset=['馬名','確定着順']).copy()
rate_src['確定着順'] = pd.to_numeric(rate_src['確定着順'], errors='coerce')
rate_src = rate_src[rate_src['確定着順'] >= 1]

hist_rate = pd.DataFrame(columns=['馬名','勝率%_HIST','連対率%_HIST','複勝率%_HIST','出走数_HIST'])
if not rate_src.empty:
    grp = rate_src.groupby('馬名')['確定着順']
    hist_rate = pd.DataFrame({
        '勝率%_HIST': (grp.apply(lambda s: (s == 1).mean()) * 100.0).round(2),
        '連対率%_HIST': (grp.apply(lambda s: (s <= 2).mean()) * 100.0).round(2),
        '複勝率%_HIST': (grp.apply(lambda s: (s <= 3).mean()) * 100.0).round(2),
        '出走数_HIST': grp.size(),
    }).reset_index()

# horses に履歴率をマージ（既存の sheet1 の数値は保持し、こちらは別名で追加）
if not hist_rate.empty:
    horses = horses.merge(hist_rate, on='馬名', how='left')

# ===== 入力チェック =====
problems=[]
for c in ['馬名','レース日','競走名','頭数','確定着順']:
    if c not in s0.columns: problems.append(f"sheet0 必須列が不足: {c}")
if '通過4角' in s0.columns and '頭数' in s0.columns:
    tmp=s0[['通過4角','頭数']].dropna()
    if len(tmp)>0 and ((tmp['通過4角']<1)|(tmp['通過4角']>tmp['頭数'])).any():
        problems.append('sheet0 通過4角が頭数レンジ外')
if problems:
    st.warning("入力チェック:\n- "+"\n- ".join(problems))

# ===== マージ =====
for dup in ['枠','番','性別','年齢','斤量','馬体重','脚質']:
    s0.drop(columns=[dup], errors='ignore', inplace=True)
df = s0.merge(horses[['馬名','枠','番','性別','年齢','斤量','馬体重','脚質']], on='馬名', how='left')

# ===== 1走スコア =====
CLASS_PTS={'G1':10,'G2':8,'G3':6,'リステッド':5,'オープン特別':4}

def normalize_grade_text(x: str|None) -> str|None:
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    s=str(x).translate(_fw)
    s=(s.replace('Ｇ','G').replace('（','(').replace('）',')')
        .replace('Ⅰ','I').replace('Ⅱ','II').replace('Ⅲ','III'))
    s=re.sub(r'G\s*III','G3',s,flags=re.I)
    s=re.sub(r'G\s*II','G2',s,flags=re.I)
    s=re.sub(r'G\s*I','G1',s,flags=re.I)
    s=re.sub(r'ＪＰＮ','Jpn',s,flags=re.I)
    s=re.sub(r'JPN','Jpn',s,flags=re.I)
    s=re.sub(r'Jpn\s*III','Jpn3',s,flags=re.I)
    s=re.sub(r'Jpn\s*II','Jpn2',s,flags=re.I)
    s=re.sub(r'Jpn\s*I','Jpn1',s,flags=re.I)
    m=re.search(r'(?:G|Jpn)\s*([123])', s, flags=re.I)
    return f"G{m.group(1)}" if m else None

def class_points(row) -> int:
    g=normalize_grade_text(row.get('クラス名')) if 'クラス名' in row else None
    if not g and '競走名' in row: g=normalize_grade_text(row.get('競走名'))
    if g in CLASS_PTS: return CLASS_PTS[g]
    name=str(row.get('クラス名',''))+' '+str(row.get('競走名',''))
    if re.search(r'3\s*勝', name): return 3
    if re.search(r'2\s*勝', name): return 2
    if re.search(r'1\s*勝', name): return 1
    if re.search(r'新馬|未勝利', name): return 1
    if re.search(r'オープン', name): return 4
    if re.search(r'リステッド|L\b', name, flags=re.I): return 5
    return 1

if 'レース日' not in df.columns:
    st.error('レース日が見つかりません。'); st.stop()

_df = df.copy()

def calc_score(r):
    gpt = class_points(r)
    # 基本：着順逆転ポイント + 出走ボーナスλ
    base = gpt * (pd.to_numeric(r['頭数'], errors='coerce') + 1 - pd.to_numeric(r['確定着順'], errors='coerce'))
    base = float(base) if np.isfinite(base) else 0.0
    base += float(lambda_part) * gpt

    # 重賞ボーナス（スライダー連動）
    gtxt = normalize_grade_text(r.get('クラス名')) or normalize_grade_text(r.get('競走名'))
    bonus_grade = int(grade_bonus) if gtxt in ['G1','G2','G3'] else 0

    # 上がりボーナス（スライダー連動）
    ao = pd.to_numeric(r.get('上3F順位', np.nan), errors='coerce')
    if ao == 1:    bonus_agari = int(agari1_bonus)
    elif ao == 2:  bonus_agari = int(agari2_bonus)
    elif ao == 3:  bonus_agari = int(agari3_bonus)
    else:          bonus_agari = 0

    return base + bonus_grade + bonus_agari

_df['score_raw'] = _df.apply(calc_score, axis=1)

if _df['score_raw'].max()==_df['score_raw'].min():
    _df['score_norm']=50.0
else:
    _df['score_norm'] = (_df['score_raw'] - _df['score_raw'].min()) / (_df['score_raw'].max()-_df['score_raw'].min())*100


now = pd.Timestamp.today()
half_life_m_default = 6.0
_race_dates = pd.to_datetime(_df['レース日'], errors='coerce')
try:
    if _race_dates.dt.tz is not None:
        _race_dates = _race_dates.dt.tz_convert(None)
except (TypeError, AttributeError):
    try:
        _race_dates = _race_dates.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
_df['_days_ago'] = (now - _race_dates).dt.days
_df['_w'] = 0.5 ** (_df['_days_ago'] / (half_life_m*30.4375 if half_life_m>0 else half_life_m_default*30.4375))

# ===== A) レース内デフレート =====
def _make_race_id_for_hist(dfh: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dfh['レース日'], errors='coerce').dt.strftime('%Y%m%d').fillna('00000000') + '_' + dfh['競走名'].astype(str).fillna('')

_df['rid_hist'] = _make_race_id_for_hist(_df)
med = _df.groupby('rid_hist')['score_norm'].transform('median')
_df['score_adj'] = _df['score_norm'] - med

# ===== ロバスト標準化（週×クラス：median/MAD）=====
# 週キー（W-MON）とクラスキー（G1/G2/G3…）
_df['WeekKey'] = pd.to_datetime(_df['レース日'], errors='coerce').dt.to_period('W-MON').astype(str)

def _class_key_row(row):
    g = normalize_grade_text(row.get('クラス名')) if 'クラス名' in row else None
    if not g and '競走名' in row:
        g = normalize_grade_text(row.get('競走名'))
    return g or 'OTHER'

_df['ClassKey'] = _df.apply(_class_key_row, axis=1)

# group: 週×クラスで median/MAD
def _mad(s):
    s = pd.to_numeric(s, errors='coerce')
    med = s.median()
    mad = np.median(np.abs(s - med))
    return med, mad

grp = _df.groupby(['WeekKey','ClassKey'])['score_adj']
med = grp.transform('median')
mad = grp.transform(lambda x: np.median(np.abs(pd.to_numeric(x, errors='coerce') - np.nanmedian(pd.to_numeric(x, errors='coerce')))))
mad = mad.replace(0, np.nan)  # ゼロ割回避
rb_z = (pd.to_numeric(_df['score_adj'], errors='coerce') - med) / (1.4826 * mad)

# 偏差値（50±10）に写像（欠損は全体で埋め）
rb_t = 50.0 + 10.0 * rb_z
rb_t = rb_t.fillna(50.0 + 10.0 * ((pd.to_numeric(_df['score_adj'], errors='coerce') - _df['score_adj'].median()) /
                                  (1.4826 * np.median(np.abs(pd.to_numeric(_df['score_adj'], errors='coerce') - _df['score_adj'].median()) + 1e-9))))

_df['ScoreT_RB'] = rb_t.clip(0, 100)  # 表示安定のため 0–100 にクリップ

# ===== 右/左回り（推定） =====
DEFAULT_VENUE_TURN = {'札幌':'右','函館':'右','福島':'右','新潟':'左','東京':'左','中山':'右','中京':'左','京都':'右','阪神':'右','小倉':'右'}
def infer_turn_row(row):
    # まず場名で判定
    venue = str(row.get('場名','')).strip()
    if venue in DEFAULT_VENUE_TURN:
        return DEFAULT_VENUE_TURN[venue]
    # 次に競走名から推定（従来互換）
    name = str(row.get('競走名',''))
    for v, t in DEFAULT_VENUE_TURN.items():
        if v in name:
            return t
    return np.nan
if '回り' not in _df.columns:
    _df['回り'] = _df.apply(infer_turn_row, axis=1)

# ===== B) β自動チューニング =====
def tune_beta(df_hist: pd.DataFrame, betas=np.linspace(0.6, 2.4, 19)) -> float:
    dfh = df_hist.dropna(subset=['score_adj','確定着順']).copy()
    dfh['rid']=_make_race_id_for_hist(dfh)
    def logloss(beta: float):
        tot=0.0; n=0
        for _, g in dfh.groupby('rid'):
            x=g['score_adj'].astype(float).to_numpy()
            y=(pd.to_numeric(g['確定着順'], errors='coerce')==1).astype(int).to_numpy()
            if len(x)<2: continue
            p=np.exp(beta*(x-x.max())); s=p.sum();
            if s<=0 or not np.isfinite(s): continue
            p/=s; p=np.clip(p,1e-9,1-1e-9)
            tot+=-np.mean(y*np.log(p) + (1-y)*np.log(1-p)); n+=1
        return tot/max(n,1)
    return float(min(betas, key=logloss))

# === タイム分位回帰（GBR） + 加重CV ===
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

def _recent_weight_from_dates(dates: pd.Series, half_life_days: float) -> np.ndarray:
    days = (pd.Timestamp.today() - pd.to_datetime(dates, errors='coerce')).dt.days
    days = days.clip(lower=0).fillna(9999)
    H = max(1.0, float(half_life_days))
    return (0.5 ** (days / H)).to_numpy(float)

def _weighted_mu_sigma(X: np.ndarray, w: np.ndarray):
    w = w / max(w.sum(), 1e-12)
    mu = (w[:, None] * X).sum(axis=0)
    sd = np.sqrt(np.maximum((w[:, None] * (X - mu) ** 2).sum(axis=0), 1e-12))
    return mu, sd

def _standardize_with_mu_sigma(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd

def _build_time_features(df_hist: pd.DataFrame, dist_turn_df: pd.DataFrame | None = None):
    need = {'走破タイム秒', '距離', '斤量', '芝・ダ', 'レース日', '馬名'}
    if not need.issubset(df_hist.columns):
        return None
    d = df_hist.copy()
    d = d.dropna(subset=list(need))
    if d.empty:
        return None
    feats = ['距離', '斤量', 'is_dirt']
    d['is_dirt'] = d['芝・ダ'].astype(str).str.contains('ダ').astype(int)
    for c in ['PCI', 'PCI3', 'Ave-3F', '上がり3Fタイム']:
        if c in d.columns:
            feats.append(c)
    if 'PCI' in d.columns:
        d['距離xPCI'] = d['距離'] * d['PCI']; feats.append('距離xPCI')
    if 'PCI3' in d.columns and 'PCI' in d.columns:
        d['PCIgap'] = d['PCI3'] - d['PCI']; feats.append('PCIgap')
    if dist_turn_df is not None and 'DistTurnZ' in dist_turn_df.columns:
        dt = dist_turn_df[['馬名', 'DistTurnZ']].drop_duplicates()
        d = d.merge(dt, on='馬名', how='left')
        feats.append('DistTurnZ')
    X = d[feats].astype(float).to_numpy()
    y = pd.to_numeric(d['走破タイム秒'], errors='coerce').to_numpy(float)
    groups = d['馬名'].astype(str).to_numpy()
    dates = pd.to_datetime(d['レース日'], errors='coerce')
    col_means = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, col_means)
    return {'frame': d, 'X': X, 'y': y, 'feats': feats, 'groups': groups, 'dates': dates, 'col_means': col_means}

def fit_time_quantile_model(df_hist: pd.DataFrame, dist_turn_today_df: pd.DataFrame,
                            half_life_days: float, q_list=(0.2, 0.5, 0.8),
                            param_grid=(80, 120, 160), random_state=42):
    built = _build_time_features(df_hist, dist_turn_today_df)
    if built is None:
        return None

    X = built['X']; y = built['y']; groups = built['groups']; dates = built['dates']
    feats = built['feats']; col_means = built['col_means']

    # 時系列重み & 標準化
    w = _recent_weight_from_dates(dates, half_life_days=max(1.0, float(half_life_days)))
    mu, sd = _weighted_mu_sigma(X, w)
    Xs = _standardize_with_mu_sigma(X, mu, sd)

    # GroupKFold で木本数選定
    n_groups = int(len(np.unique(groups)))
    n_splits = max(2, min(5, n_groups))
    best_param, best_rmse = None, 1e18

    if n_groups >= 2 and Xs.shape[0] >= 20:
        gkf = GroupKFold(n_splits=n_splits)
        for n_est in param_grid:
            rmse_fold = []
            for tr, va in gkf.split(Xs, y, groups=groups):
                Xt, Xv = Xs[tr], Xs[va]
                yt, yv = y[tr], y[va]
                wt, wv = w[tr], w[va]
                med = GradientBoostingRegressor(loss='quantile', alpha=0.5,
                                                n_estimators=n_est, max_depth=3,
                                                random_state=random_state)
                med.fit(Xt, yt, sample_weight=wt)
                pred = med.predict(Xv)
                rmse = np.sqrt(np.average((yv - pred) ** 2, weights=wv))
                rmse_fold.append(rmse)
            score = float(np.mean(rmse_fold)) if rmse_fold else 1e18
            if score < best_rmse:
                best_rmse, best_param = score, n_est
    else:
        best_param = sorted(param_grid)[len(param_grid)//2]

    if best_param is None:
        best_param = sorted(param_grid)[len(param_grid)//2]

    models = {}
    for q in q_list:
        m = GradientBoostingRegressor(loss='quantile', alpha=q,
                                      n_estimators=best_param, max_depth=3,
                                      random_state=random_state)
        m.fit(Xs, y, sample_weight=w)
        models[q] = m

    resid = y - models[0.5].predict(Xs)
    sigma_hat = float(np.sqrt(np.maximum(np.average(resid ** 2, weights=w), 1e-6)))

    return {
        'feats': feats, 'mu': mu, 'sd': sd, 'col_means': col_means,
        'models': models, 'n_estimators': int(best_param), 'sigma_hat': sigma_hat
    }

def _field_pci_from_pace(pace_type: str) -> float:
    return {'ハイペース': 46.0, 'ミドルペース': 50.0, 'ややスローペース': 53.0, 'スローペース': 56.0}.get(str(pace_type), 50.0)

def build_today_design(horses_today: pd.DataFrame, s0_hist: pd.DataFrame,
                       target_distance: int, target_surface: str,
                       dist_turn_today_df: pd.DataFrame, feats: list[str],
                       pace_type: str):
    # 過去の指数の時間減衰平均を作る
    rec_w = None
    if not s0_hist.empty and 'レース日' in s0_hist:
        rec_w = 0.5 ** ((pd.Timestamp.today() - pd.to_datetime(s0_hist['レース日'], errors='coerce')).dt.days.clip(lower=0) / 180.0)

    def _wmean(s, w):
        s = pd.to_numeric(s, errors='coerce')
        w = pd.to_numeric(w, errors='coerce')
        m = np.nansum(s * w); sw = np.nansum(w)
        return float(m / sw) if sw > 0 else np.nan

    pci_wmean, pci3_wmean, ave3_wmean, a3_wmedian = {}, {}, {}, {}
    if 'PCI' in s0_hist.columns:
        pci_wmean = s0_hist.assign(_w=rec_w).groupby('馬名').apply(lambda g: _wmean(g['PCI'], g['_w'])).to_dict()
    if 'PCI3' in s0_hist.columns:
        pci3_wmean = s0_hist.assign(_w=rec_w).groupby('馬名').apply(lambda g: _wmean(g['PCI3'], g['_w'])).to_dict()
    if 'Ave-3F' in s0_hist.columns:
        ave3_wmean = s0_hist.assign(_w=rec_w).groupby('馬名').apply(lambda g: _wmean(g['Ave-3F'], g['_w'])).to_dict()
    if '上がり3Fタイム' in s0_hist.columns:
        a3_wmedian = s0_hist.groupby('馬名')['上がり3Fタイム'].median().to_dict()

    dtt = dist_turn_today_df[['馬名', 'DistTurnZ']].drop_duplicates() if ('DistTurnZ' in dist_turn_today_df.columns) \
        else pd.DataFrame({'馬名': horses_today['馬名'], 'DistTurnZ': np.nan})
    H = horses_today.merge(dtt, on='馬名', how='left')

    rows = []
    pci_field = _field_pci_from_pace(pace_type)  # ★ 明示引数で受け取る

    for _, r in H.iterrows():
        name = str(r['馬名'])
        x = {}
        x['距離'] = float(target_distance)
        x['斤量'] = float(r.get('斤量', np.nan))
        x['is_dirt'] = 1.0 if str(target_surface).startswith('ダ') else 0.0

        if 'PCI' in feats:
            x['PCI'] = float(pci_wmean.get(name, np.nan))
            if not np.isfinite(x['PCI']): x['PCI'] = pci_field
        if 'PCI3' in feats:
            x['PCI3'] = float(pci3_wmean.get(name, np.nan))
            if not np.isfinite(x['PCI3']): x['PCI3'] = (x.get('PCI', pci_field) + 1.0)
        if 'Ave-3F' in feats:
            x['Ave-3F'] = float(ave3_wmean.get(name, np.nan))
        if '上がり3Fタイム' in feats:
            x['上がり3Fタイム'] = float(a3_wmedian.get(name, np.nan))
        if '距離xPCI' in feats:
            x['距離xPCI'] = x['距離'] * x.get('PCI', pci_field)
        if 'PCIgap' in feats:
            x['PCIgap'] = x.get('PCI3', x.get('PCI', pci_field) + 1.0) - x.get('PCI', pci_field)
        if 'DistTurnZ' in feats:
            x['DistTurnZ'] = float(r.get('DistTurnZ', np.nan))

        rows.append({'馬名': name, **x})
    return pd.DataFrame(rows)


# ===== E) 距離バンド幅 自動 =====

def auto_h_m(x_all: np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(x_all), errors='coerce').dropna().to_numpy(float)
    if x.size<3: return 300.0
    q75,q25=np.percentile(x,75),np.percentile(x,25)
    iqr=q75-q25
    sigma=np.std(x)
    bw=0.9*min(sigma, iqr/1.34)*(x.size**(-1/5))
    return float(np.clip(bw, 120.0, 600.0))

# ===== 距離×回り 近傍化（NW、score_adj使用） =====

def kish_neff(w: np.ndarray) -> float:
    w=np.asarray(w,float); sw=w.sum(); s2=np.sum(w**2)
    return float((sw*sw)/s2) if s2>0 else 0.0

def nw_mean(x, y, w, h):
    x=np.asarray(x,float); y=np.asarray(y,float); w=np.asarray(w,float)
    if len(x)==0: return np.nan
    K=np.exp(-0.5*(x/max(1e-9,h))**2) * w
    sK=K.sum()
    return float((K*y).sum()/sK) if sK>0 else np.nan

# 準備
hist_for_turn = _df[['馬名','距離','回り','score_adj','_w']].dropna(subset=['馬名','距離','score_adj','_w']).copy()

# h選定
h_auto = auto_h_m(hist_for_turn['距離'].to_numpy())

# 指定距離・回りでの推定
def dist_turn_profile(name: str, df_hist: pd.DataFrame, target_d: int, target_turn: str, h_m: float, opp_turn_w: float=0.5):
    g=df_hist[df_hist['馬名'].astype(str).str.strip()==str(name)].copy()
    if g.empty: return {'DistTurnZ':np.nan,'n_eff_turn':0.0,'BestDist_turn':np.nan,'DistTurnZ_best':np.nan}
    w0 = g['_w'].to_numpy(float) * np.where(g['回り'].astype(str)==str(target_turn), 1.0, float(opp_turn_w))
    x  = g['距離'].to_numpy(float)
    y  = g['score_adj'].to_numpy(float)
    msk=np.isfinite(x)&np.isfinite(y)&np.isfinite(w0)
    x,y,w0=x[msk],y[msk],w0[msk]
    if x.size==0: return {'DistTurnZ':np.nan,'n_eff_turn':0.0,'BestDist_turn':np.nan,'DistTurnZ_best':np.nan}
    z_hat = nw_mean(x-float(target_d), y, w0, h_m)
    w_eff = kish_neff(np.exp(-0.5*((x-float(target_d))/max(1e-9,h_m))**2)*w0)
    # プロファイル
    ds=np.arange(int(target_d-600), int(target_d+600)+1, 100)
    best_d,best_val=np.nan,-1e18
    for d0 in ds:
        z=nw_mean(x-float(d0), y, w0, h_m)
        if np.isfinite(z) and z>best_val:
            best_val=float(z); best_d=int(d0)
    return {
        'DistTurnZ': float(z_hat) if np.isfinite(z_hat) else np.nan,
        'n_eff_turn': float(w_eff),
        'BestDist_turn': float(best_d) if np.isfinite(best_d) else np.nan,
        'DistTurnZ_best': float(best_val) if np.isfinite(best_val) else np.nan
    }

# ===== スペクトル: 曲線生成 / DTW / テンプレ構築 =====
def pseudo_curve(dist_m: float, time_s: float, a3_s: float, pci: float, pci3: float, rpci: float, pos_ratio: float,
                 n_points: int = 128) -> np.ndarray:
    """
    距離・総タイム・上がり3Fなどから擬似速度曲線（正規化）を合成。
    - 距離: m, タイム: s
    - pos_ratio: 先行度（1=先行、0=後方）
    """
    # フォールバック
    if not np.isfinite(dist_m) or not np.isfinite(time_s) or time_s <= 0:
        return np.full(n_points, np.nan)

    # ベース速度（平均）
    v_avg = dist_m / time_s

    # PCI 系の形状寄与
    pci = np.clip(pci if np.isfinite(pci) else 50.0, 35.0, 65.0)
    pci3 = pci3 if np.isfinite(pci3) else (pci + 1.0)
    rpci = rpci if np.isfinite(rpci) else 0.0  # 任意列

    x = np.linspace(0, 1, n_points)

    # 先行/差しによる序盤/終盤ウェイト
    lead = np.clip(pos_ratio if np.isfinite(pos_ratio) else 0.5, 0.0, 1.0)
    w_front = 0.8 + 0.6*(lead - 0.5)    # 先行ほど序盤↑
    w_back  = 0.8 - 0.6*(lead - 0.5)

    # PCI→中盤の「緩み/締まり」をガウス形で付与
    mid_center = 0.55
    mid_width  = 0.20
    mid_bump   = (50.0 - pci) / 25.0  # ハイペース(PCI低)だと中盤落ち込み
    shape_mid  = 1.0 - mid_bump * np.exp(-0.5*((x - mid_center)/mid_width)**2)

    # 上がり3F（終盤）による末脚寄与
    if np.isfinite(a3_s) and a3_s > 0:
        # コース全体速度に対する終盤速度比（粗い近似）
        last_600 = 600.0
        if dist_m > last_600:
            v_last = last_600 / a3_s
            ratio  = np.clip(v_last / v_avg, 0.5, 1.5)
        else:
            ratio = 1.0
    else:
        ratio = 1.0 + 0.10*((pci3 - pci)/5.0)  # PCI3>PCI なら末上げ

    # 周波数ドメインの軽い起伏（FFT相当の特徴を少し持たせる）
    low = 0.04*np.sin(2*np.pi*1*x)
    mid = 0.03*np.sin(2*np.pi*2*x + 1.3)
    high= 0.02*np.sin(2*np.pi*4*x + 0.7)
    wav = 1.0 + low + mid + high

    # 合成
    base = v_avg * (w_front*(1 - x) + w_back*x)   # 直線補間の勾配
    v = base * shape_mid * (1.0 + (ratio - 1.0)*x**1.2) * wav

    # 正規化（面積=1に近づける）
    v = np.maximum(v, 1e-8)
    v = v / np.trapz(v, x)
    return v

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """シンプルDTW（O(N^2)、NaNは大きい罰則）"""
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return np.inf
    if len(a)==0 or len(b)==0 or np.isnan(a).all() or np.isnan(b).all():
        return np.inf
    A = np.nan_to_num(a, nan=1e6)
    B = np.nan_to_num(b, nan=1e6)
    n, m = len(A), len(B)
    D = np.full((n+1, m+1), np.inf)
    D[0,0] = 0.0
    for i in range(1, n+1):
        ai = A[i-1]
        for j in range(1, m+1):
            cost = abs(ai - B[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m])

def build_template_curves(df_curves: pd.DataFrame, target_dist: int, target_surface: str,
                          tol: int = 100, n_points: int = 128):
    """
    同距離帯(±tol) & 同Surface の '_curve' を集めて中央値テンプレを作る。
    ついでにFFT帯域からレース型 Gate を推定（0=持久,1=中庸,2=瞬発）。
    """
    if df_curves.empty or '_curve' not in df_curves.columns:
        return np.full(n_points, 1.0/n_points), {'Gate': 1}

    dd = df_curves.copy()
    dd['距離'] = pd.to_numeric(dd['距離'], errors='coerce')
    dd = dd[dd['距離'].between(target_dist - tol, target_dist + tol, inclusive='both')]
    dd = dd[dd['芝・ダ'].astype(str).str.startswith(str(target_surface))]

    curves = [c for c in dd['_curve'].values if isinstance(c, np.ndarray)]
    if not curves:
        return np.full(n_points, 1.0/n_points), {'Gate': 1}

    # 長さを合わせて中央値
    L = min(min(len(c) for c in curves), n_points)
    mat = np.vstack([np.interp(np.linspace(0,1,L), np.linspace(0,1,len(c)), c) for c in curves])
    templ = np.median(mat, axis=0)

    # FFT帯域比から Gate を簡易判定
    f = np.fft.rfft(templ - templ.mean())
    pow_spec = np.abs(f)**2
    # 低/中/高の簡易帯域比
    n = len(pow_spec)
    low = pow_spec[:max(2, n//6)].sum()
    mid = pow_spec[max(2, n//6):max(4, n//3)].sum()
    high= pow_spec[max(4, n//3):].sum()

    # 高周波(瞬発)が強ければ2、低周波(持久)が強ければ0、その他1
    if high > max(low, mid) * 1.15:
        gate = 2
    elif low > max(mid, high) * 1.15:
        gate = 0
    else:
        gate = 1

    return templ, {'Gate': gate}

def infer_gate_from_curve(curve: np.ndarray, n_points: int = 128):
    """1本の速度曲線から 0=持久 / 1=中庸 / 2=瞬発 をFFT帯域比で推定"""
    if not isinstance(curve, np.ndarray) or curve.size == 0 or np.isnan(curve).all():
        return np.nan
    L = min(len(curve), n_points)
    x = np.interp(np.linspace(0,1,L), np.linspace(0,1,len(curve)), curve)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    f = np.fft.rfft(x)
    ps = np.abs(f)**2
    n = len(ps)
    low  = ps[:max(2, n//6)].sum()
    mid  = ps[max(2, n//6):max(4, n//3)].sum()
    high = ps[max(4, n//3):].sum()
    if high > max(low, mid) * 1.15:
        return 2  # 瞬発
    elif low > max(mid, high) * 1.15:
        return 0  # 持久
    else:
        return 1  # 中庸

# ===== 馬ごと集計 =====
agg=[]
for name, g in _df.groupby('馬名'):
    avg=g['score_norm'].mean()
    std=g['score_norm'].std(ddof=0)
    wavg = np.average(g['score_norm'], weights=g['_w']) if g['_w'].sum()>0 else np.nan
    wstd = w_std_unbiased(g['score_norm'], g['_w'], ddof=1) if len(g)>=2 else np.nan
    agg.append({'馬名':_trim_name(name),'AvgZ':avg,'Stdev':std,'WAvgZ':wavg,'WStd':wstd,'Nrun':len(g)})

df_agg=pd.DataFrame(agg)
if df_agg.empty:
    st.error('過去走の集計が空です。'); st.stop()

# WStdの床
wstd_nontrivial=df_agg.loc[df_agg['Nrun']>=2,'WStd']
def_std=float(wstd_nontrivial.median()) if wstd_nontrivial.notna().any() else 6.0
min_floor=max(1.0, def_std*0.6)
df_agg['WStd']=df_agg['WStd'].fillna(def_std)
df_agg.loc[df_agg['WStd']<min_floor,'WStd']=min_floor

# 今日情報
for df_ in [df_agg, horses]:
    if '馬名' in df_.columns: df_['馬名']=df_['馬名'].map(_trim_name)

if '脚質' in horses.columns: horses['脚質']=horses['脚質'].map(normalize_style)

cols_to_merge = ['馬名','枠','番','脚質','性別','年齢',
                 '勝率%_HIST','連対率%_HIST','複勝率%_HIST']  # NEW
cols_to_merge = [c for c in cols_to_merge if c in horses.columns]
df_agg = df_agg.merge(horses[cols_to_merge], on='馬名', how='left')

# === 新規: 特性Pts（性別・脚質・年齢・枠） ===
idx = df_agg.index

# 性別
sex_map = {'牡': SEX_MALE, '牝': SEX_FEMA, 'セ': SEX_GELD, '騙': SEX_GELD, 'せん': SEX_GELD}
sex_series = df_agg['性別'] if '性別' in df_agg.columns else pd.Series(['']*len(idx), index=idx)
df_agg['SexPts'] = sex_series.astype(str).map(sex_map).fillna(0.0).astype(float)

# 脚質
style_map = {'逃げ': STL_NIGE, '先行': STL_SENKO, '差し': STL_SASHI, '追込': STL_OIKOMI}
style_series = df_agg['脚質'] if '脚質' in df_agg.columns else pd.Series(['']*len(idx), index=idx)
df_agg['StylePts'] = style_series.astype(str).map(style_map).fillna(0.0).astype(float)

# 年齢（ピークからの距離で減点）
age_series = pd.to_numeric(df_agg['年齢'] if '年齢' in df_agg.columns else pd.Series([np.nan]*len(idx), index=idx), errors='coerce')
df_agg['AgePts'] = (-float(AGE_SLOPE) * (age_series - int(AGE_PEAK)).abs()).fillna(0.0)

# 枠（内外バイアス）
w = pd.to_numeric(df_agg['枠'] if '枠' in df_agg.columns else pd.Series([np.nan]*len(idx), index=idx), errors='coerce')
centered = (4.5 - w) / 3.5   # 枠1=+1, 枠8=-1（内が正）
if WAKU_DIR == "内有利":
    waku_raw = centered
elif WAKU_DIR == "外有利":
    waku_raw = -centered
else:
    waku_raw = pd.Series(0.0, index=idx)
df_agg['WakuPts'] = (float(WAKU_STR) * pd.to_numeric(waku_raw)).fillna(0.0)

# === 右/左集計（score_adjの重み平均） ===
turn_base = (
    _df[['馬名','回り','score_adj','_w']]
    .dropna(subset=['馬名','回り','score_adj','_w'])
    .copy()
)

def _wavg_score(g: pd.DataFrame) -> float:
    w = pd.to_numeric(g['_w'], errors='coerce').to_numpy(float)
    x = pd.to_numeric(g['score_adj'], errors='coerce').to_numpy(float)
    sw = np.nansum(w)
    return float(np.nansum(w * x) / sw) if sw > 0 else np.nan

# 右回り
right = (
    turn_base[turn_base['回り'].astype(str) == '右']
    .groupby('馬名', as_index=False)
    .apply(lambda g: pd.Series({'RightZ': _wavg_score(g)}))
    .reset_index(drop=True)
)

# 左回り
left = (
    turn_base[turn_base['回り'].astype(str) == '左']
    .groupby('馬名', as_index=False)
    .apply(lambda g: pd.Series({'LeftZ': _wavg_score(g)}))
    .reset_index(drop=True)
)

# 出走本数カウント
counts = (
    turn_base.assign(_one=1)
    .pivot_table(index='馬名', columns='回り', values='_one', aggfunc='sum', fill_value=0)
    .rename(columns={'右': 'nR', '左': 'nL'})
    .reset_index()
)

# まとめて結合
turn_pref = (
    pd.merge(right, left, on='馬名', how='outer')
    .merge(counts, on='馬名', how='left')
    .fillna({'nR': 0, 'nL': 0})
)

# 欠損保険
for c in ['RightZ','LeftZ','nR','nL']:
    if c not in turn_pref.columns:
        turn_pref[c] = np.nan if c in ['RightZ','LeftZ'] else 0

# 以降は元の計算のまま
turn_pref['TurnGap'] = (turn_pref['RightZ'].fillna(0) - turn_pref['LeftZ'].fillna(0))
turn_pref['n_eff_turn'] = (turn_pref['nR'].fillna(0) + turn_pref['nL'].fillna(0)).clip(lower=0)
conf = np.clip(turn_pref['n_eff_turn'] / 3.0, 0.0, 1.0)
turn_pref['TurnPrefPts'] = np.clip(turn_pref['TurnGap'] / 1.5, -1.0, 1.0) * conf

df_agg = df_agg.merge(
    turn_pref[['馬名','RightZ','LeftZ','TurnGap','n_eff_turn','TurnPrefPts']],
    on='馬名',
    how='left'
)



# 距離×回り（自動h）
rows=[]
for nm in df_agg['馬名'].astype(str):
    prof=dist_turn_profile(nm, hist_for_turn, int(TARGET_DISTANCE), str(TARGET_TURN), h_auto, opp_turn_w=0.5)
    rows.append({'馬名':nm, **prof})
_dfturn = pd.DataFrame(rows)
df_agg = df_agg.merge(_dfturn, on='馬名', how='left')
neff_cols = [c for c in df_agg.columns if c.startswith('n_eff_turn')]
if neff_cols:
    neff_series = (
        df_agg[neff_cols]
        .apply(pd.to_numeric, errors='coerce')
        .bfill(axis=1)                # 先にある列を優先
        .iloc[:, 0]                   # 代表列を1本だけ選ぶ
        .fillna(0.0)
    )
    df_agg['n_eff_turn'] = neff_series
    # 余分な *_x, *_y は落としておくと後段で安全
    for c in neff_cols:
        if c != 'n_eff_turn':
            df_agg.drop(columns=c, inplace=True, errors='ignore')
else:
    # 列自体が無いケースの保険
    df_agg['n_eff_turn'] = 0.0


# 過去走ごとに速度曲線を構築（前半で定義した pseudo_curve を使用）
s0_spec = s0.copy()
for need in ['距離','走破タイム秒','上がり3Fタイム']:
    if need not in s0_spec.columns: s0_spec[need] = np.nan
for opt in ['PCI','PCI3','RPCI','通過4角','頭数','芝・ダ']:
    if opt not in s0_spec.columns: s0_spec[opt] = np.nan

def _pos_ratio_4c(row) -> float:
    p = pd.to_numeric(row.get('通過4角', np.nan), errors='coerce')
    n = pd.to_numeric(row.get('頭数', np.nan), errors='coerce')
    if not np.isfinite(p) or not np.isfinite(n) or n<=1: return 0.5
    return float((n - p) / (n - 1))  # 先行=1.0, 後方=0.0

s0_spec['_pos'] = s0_spec.apply(_pos_ratio_4c, axis=1)
s0_spec['_curve'] = s0_spec.apply(
    lambda r: pseudo_curve(
        pd.to_numeric(r['距離'], errors='coerce'),
        pd.to_numeric(r['走破タイム秒'], errors='coerce'),
        pd.to_numeric(r['上がり3Fタイム'], errors='coerce'),
        pd.to_numeric(r['PCI'], errors='coerce'),
        pd.to_numeric(r['PCI3'], errors='coerce'),
        pd.to_numeric(r['RPCI'], errors='coerce'),
        float(r['_pos'])
    ),
    axis=1
)

# テンプレ曲線（距離±100m & 同Surfaceの中央値）
templ_curve, templ_info = build_template_curves(
    s0_spec[['馬名','距離','芝・ダ','_curve']].copy(),
    int(TARGET_DISTANCE),
    str(TARGET_SURFACE),
    tol=int(templ_tol_m)
)


# 各馬のDTW最小距離→Z化（大きいほど適合良）
rows=[]
for name, g in s0_spec.groupby('馬名'):
    dists=[]; gates=[]
    for v in g['_curve'].values:
        if isinstance(v, np.ndarray):
            dists.append(dtw_distance(v, templ_curve))
            gates.append(infer_gate_from_curve(v))
    DTWmin = float(np.nanmin(dists)) if dists else np.nan
    gate_hat = float(np.nanmedian([x for x in gates if np.isfinite(x)])) if gates else np.nan
    rows.append({'馬名':name, 'DTWmin':DTWmin, 'SpecGate_horse': gate_hat})
df_spec = pd.DataFrame(rows)

if not df_spec.empty:
    mu = float(df_spec['DTWmin'].mean(skipna=True))
    sd = float(df_spec['DTWmin'].std(ddof=0, skipna=True))
    if not np.isfinite(sd) or sd==0.0: sd=1.0
    df_spec['SpecFitZ'] = -(df_spec['DTWmin'] - mu) / sd
else:
    df_spec = pd.DataFrame({'馬名': df_agg['馬名'], 'SpecFitZ': np.nan, 'SpecGate_horse': np.nan})

# テンプレ（レース全体）の型は全員同じ値として列を持たせる
df_spec['SpecGate_templ'] = templ_info.get('Gate', 1)

df_agg = df_agg.merge(df_spec, on='馬名', how='left')
# 数値ゲート(0/1/2) → ラベルへ
_gate_label = {0: '持久', 1: '中庸', 2: '瞬発'}
df_agg['SpecGate_horse']  = pd.to_numeric(df_agg['SpecGate_horse'], errors='coerce')  # 念のため
df_agg['SpecGate_templ']  = pd.to_numeric(df_agg['SpecGate_templ'], errors='coerce')

df_agg['SpecGate_horse_lbl'] = df_agg['SpecGate_horse'].map(_gate_label)
df_agg['SpecGate_templ_lbl'] = df_agg['SpecGate_templ'].map(_gate_label)

# ===== 調教（物理）→ PhysicsZ を作る =====
df_phys = pd.DataFrame()

if USE_PHYSICS:
    valid_trains = []

    # 入ってくる各ファイルを読み、必須列が揃っているものだけ採用
    for f, kind in [(wood_file, 'wood'), (hill_file, 'hill')]:
        if f is None:
            continue
        tdf = _read_train_xlsx(f, kind)  # 期待列: 馬名, 日付, _kind, _lap_sec, _intensity
        if isinstance(tdf, pd.DataFrame) and not tdf.empty:
            need = {'馬名', '日付', '_lap_sec'}
            if need.issubset(set(tdf.columns)):
                # 余計な列はあっても良いが、最低限の列はそろえておく
                keep = [c for c in ['馬名','日付','_kind','_lap_sec','_intensity'] if c in tdf.columns]
                valid_trains.append(tdf[keep])

    if valid_trains:
        T = pd.concat(valid_trains, ignore_index=True)
        # ここで KeyError が出ないように列存在を再確認
        if {'馬名','日付'}.issubset(T.columns):
            T = T.dropna(subset=['馬名','日付'])
            if not T.empty:
                df_phys = _derive_training_metrics(
                    train_df=T, s0_races=_df.copy(),
                    Crr_wood=Crr_wood, Crr_hill=Crr_hill,
                    CdA=CdA, rho=rho_air,
                    Pmax_wkg=Pmax_wkg, Emax_jkg=Emax_jkg,
                    half_life_days=int(half_life_train_days)
                )
with st.expander("調教パース状況", expanded=False):
    st.write("woodファイル: ", wood_file.name if wood_file else None,
             " / hillファイル: ", hill_file.name if hill_file else None)
    try:
        trains_dbg = []
        if wood_file: trains_dbg.append(_read_train_xlsx(wood_file, 'wood'))
        if hill_file: trains_dbg.append(_read_train_xlsx(hill_file, 'hill'))
        T_dbg = pd.concat(trains_dbg, ignore_index=True) if trains_dbg else pd.DataFrame()

        st.write("抽出行数(調教):", len(T_dbg))
        if not T_dbg.empty:
            st.dataframe(T_dbg.head(10))
            st.write("列:", list(T_dbg.columns))
            st.write("調教 日付範囲:", T_dbg['日付'].min(), "→", T_dbg['日付'].max())

            # マージできる名前の突合
            base_names = set(df_agg['馬名'].astype(str))
            phys_names = set(T_dbg['馬名'].astype(str))
            inter = sorted(base_names & phys_names)
            miss  = sorted(base_names - phys_names)
            st.write("出走表∩調教 交差:", len(inter))
            if miss:
                st.write("未マッチ（出走表にあるが調教に無い）例:", miss[:10])
    except Exception as e:
        st.write("デバッグ中に例外:", e)

    st.write("df_phys 行数:", 0 if df_phys.empty else len(df_phys))
    if not df_phys.empty:
        st.dataframe(df_phys.head(10))


# 物理DFが空ならダミー列を用意（以降の merge でコケないように）
if df_phys.empty:
    df_phys = pd.DataFrame({
        '馬名': df_agg['馬名'],
        'EAP': np.nan, 'PeakWkg': np.nan, 'EffReserve': np.nan, 'PhysicsZ': np.nan
    })
# ゆるい結合（difflib）で救済：交差が少ないときだけ
try:
    base_names = df_agg['馬名'].astype(str).tolist()
    phys_names = df_phys['馬名'].astype(str).tolist()
    inter = set(base_names) & set(phys_names)
    if len(inter) <= max(2, len(base_names)//4):
        import difflib
        # 調教側 → 出走側 へ置換マップを作る
        rep = {}
        for pn in phys_names:
            m = difflib.get_close_matches(pn, base_names, n=1, cutoff=0.90)
            if m:
                rep[pn] = m[0]
        if rep:
            df_phys = df_phys.assign(__join=df_phys['馬名'].replace(rep)).drop(columns=['馬名']).rename(columns={'__join':'馬名'})
except Exception:
    pass

df_agg = df_agg.merge(df_phys, on='馬名', how='left')
for c in ['PhysicsZ', 'PeakWkg', 'EAP']:
    df_agg[c] = pd.to_numeric(df_agg.get(c), errors='coerce')

# ===== RecencyZ / StabZ =====
base_for_recency = df_agg.get('WAvgZ', pd.Series(np.nan, index=df_agg.index)).fillna(df_agg.get('AvgZ', pd.Series(0.0, index=df_agg.index)))
df_agg['RecencyZ']=z_score(pd.to_numeric(base_for_recency, errors='coerce').fillna(0.0))
wstd_fill=pd.to_numeric(df_agg['WStd'], errors='coerce')
if not np.isfinite(wstd_fill).any(): wstd_fill=pd.Series(6.0, index=df_agg.index)
df_agg['StabZ']=z_score(-(wstd_fill.fillna(wstd_fill.median())))

# D) ベストタイム重み 自己学習
if 'ベストタイム秒' in s1.columns and s1['ベストタイム秒'].notna().any():
    bt = _df.merge(s1[['馬名','ベストタイム秒']], on='馬名', how='left')
    bt_min=bt['ベストタイム秒'].min(skipna=True); bt_max=bt['ベストタイム秒'].max(skipna=True)
    span=(bt_max-bt_min) if (pd.notna(bt_min) and pd.notna(bt_max) and bt_max>bt_min) else 1.0
    bt['BT_norm']=((bt_max - bt['ベストタイム秒'])/span).clip(0,1)
    corr = np.corrcoef(bt['BT_norm'].fillna(0.0), bt['score_adj'].fillna(0.0))[0,1]
    if not np.isfinite(corr): corr=0.0
    w_bt = float(np.clip(corr, 0.0, 1.2)) if MODE=="AUTO（推奨）" else float(besttime_w_manual)
else:
    w_bt = 0.0 if MODE=="AUTO（推奨）" else float(besttime_w_manual)

# ===== 最終スコア（未キャリブレーション指標） =====
turn_gain = 1.0
pace_gain = float(pace_gain)
stab_weight = float(stab_weight)
dist_gain = 1.0

# === 欠損クレンジング（FinalRawの直前で一度だけ） ===
df_agg['RecencyZ']    = pd.to_numeric(df_agg['RecencyZ'], errors='coerce').fillna(df_agg['RecencyZ'].median())
df_agg['StabZ']       = pd.to_numeric(df_agg['StabZ'],    errors='coerce').fillna(df_agg['StabZ'].median())
df_agg['TurnPrefPts'] = pd.to_numeric(df_agg['TurnPrefPts'], errors='coerce').fillna(0.0)
df_agg['DistTurnZ']   = pd.to_numeric(df_agg['DistTurnZ'],   errors='coerce').fillna(0.0)

# === 2歳戦モードの重み補正 ===
if 'df_agg' in locals() and TWOYO_MODE:
    # 履歴の形状適性は弱める（なければ無視される）
    for col in ('TurnPrefPts', 'DistTurnZ'):
        if col in df_agg.columns:
            df_agg[col] = 0.0
    # 安定性ウェイトが変数なら40%に弱体化（存在すれば）
    if 'stab_weight' in locals():
        try:
            stab_weight = float(stab_weight) * 0.4
        except Exception:
            pass

df_agg['PacePts']=0.0  # 後でMCから
# FinalRaw（基礎：Recency/Stab/Turn/Dist + 特性）
df_agg['FinalRaw'] = (
    df_agg['RecencyZ']
    + float(stab_weight) * df_agg['StabZ']
    + 1.0 * df_agg['TurnPrefPts']
    + 1.0 * df_agg['DistTurnZ'].fillna(0.0)
    + df_agg['SexPts'] + df_agg['StylePts'] + df_agg['AgePts'] + df_agg['WakuPts']
)

df_agg['FinalRaw_before_balance'] = df_agg['FinalRaw'].astype(float)

# 斤量ペナルティ（中央値基準）
if '斤量' in df_agg.columns and pd.to_numeric(df_agg['斤量'], errors='coerce').notna().any():
    kg = pd.to_numeric(df_agg['斤量'], errors='coerce')
    kg_med = float(np.nanmedian(kg))
    df_agg['FinalRaw'] -= float(weight_coeff) * (kg - kg_med).fillna(0.0)

# BTを加点
if 'ベストタイム秒' in s1.columns:
    btmap = s1.set_index('馬名')['ベストタイム秒'].to_dict()
    btvals = df_agg['馬名'].map(btmap)
    if pd.Series(btvals).notna().any():
        bts = pd.Series(btvals)
        bt_min=bts.min(skipna=True); bt_max=bts.max(skipna=True)
        span=(bt_max-bt_min) if (pd.notna(bt_min) and pd.notna(bt_max) and bt_max>bt_min) else 1.0
        BT_norm = ((bt_max - bts)/span).clip(0,1).fillna(0.0)
        df_agg['FinalRaw'] += w_bt * BT_norm

# ★ スペクトル寄与を最後に合成
# === 2歳戦: スペクトル比重の下限を引き上げ（任意） ===
if 'spec_ratio' in locals() and TWOYO_MODE:
    try:
        spec_ratio = max(float(spec_ratio), 0.65)
    except Exception:
        pass
# ★ スペクトル寄与（ユーザーの係数 × 配分）
df_agg['SpecFitZ'] = pd.to_numeric(df_agg['SpecFitZ'], errors='coerce')
df_agg['FinalRaw'] += spec_ratio * float(spectral_weight_ui) * df_agg['SpecFitZ'].fillna(0.0)

# ★ 物理寄与（Z=50を0基準, 10刻みで他Zとスケール合わせ）× 配分
df_agg['PhysicsZ'] = pd.to_numeric(df_agg['PhysicsZ'], errors='coerce')
df_agg['FinalRaw'] += phys_ratio * ((df_agg['PhysicsZ'] - 50.0) / 10.0).fillna(0.0)

beta_pl = tune_beta(_df.copy()) if MODE=="AUTO（推奨）" else float(mc_beta_manual)

# === 2歳戦: PL鋭さの緩和 ===
try:
    if TWOYO_MODE:
        beta_pl = max(0.9, float(beta_pl) * 0.85)
except Exception:
    pass

# === 2歳戦: ペースMCトグル ===
if USE_MC:
    # ===== ペースMC（反対称Gumbelで分散低減） =====
    mark_rule={
        'ハイペース':      {'逃げ':'△','先行':'△','差し':'◎','追込':'〇'},
        'ミドルペース':    {'逃げ':'〇','先行':'◎','差し':'〇','追込':'△'},
        'ややスローペース': {'逃げ':'〇','先行':'◎','差し':'△','追込':'×'},
        'スローペース':    {'逃げ':'◎','先行':'〇','差し':'△','追込':'×'},
    }
    mark_to_pts={'◎':2,'〇':1,'○':1,'△':0,'×':-1}

    name_list=df_agg['馬名'].tolist()
    P=np.zeros((len(name_list),4),float)
    for i, nm in enumerate(name_list):
        stl = df_agg.loc[df_agg['馬名']==nm, '脚質'].values
        stl = stl[0] if len(stl)>0 else ''
        if stl in STYLES:
            P[i, STYLES.index(stl)] = 1.0
        else:
            P[i,:]=0.25

    epi_alpha, epi_beta = 1.0, 0.6
    thr_hi, thr_mid, thr_slow = 0.52, 0.30, 0.18

    rng = np.random.default_rng(24601)
    draws = 10000
    Hn=len(name_list)
    sum_pts=np.zeros(Hn,float); pace_counter={'ハイペース':0,'ミドルペース':0,'ややスローペース':0,'スローペース':0}
    for _ in range(draws//2):
        sampled = [np.argmax(P[i]) for i in range(Hn)]
        nige  = sum(1 for s in sampled if s==0)
        sengo = sum(1 for s in sampled if s==1)
        epi=(epi_alpha*nige + epi_beta*sengo)/max(1,Hn)
        if   epi>=thr_hi:   pace_t='ハイペース'
        elif epi>=thr_mid:  pace_t='ミドルペース'
        elif epi>=thr_slow: pace_t='ややスローペース'
        else:               pace_t='スローペース'
        pace_counter[pace_t]+=2
        mk=mark_rule[pace_t]
        for i,s in enumerate(sampled):
            sum_pts[i]+=2*mark_to_pts[ mk[STYLES[s]] ]

    df_agg['PacePts']=sum_pts/max(1,draws)
    pace_type=max(pace_counter, key=lambda k: pace_counter[k]) if sum(pace_counter.values())>0 else 'ミドルペース'
else:
    # MCを使わない場合はペース点を固定
    if 'df_agg' in locals():
        df_agg['PacePts'] = 0.0
    pace_type = 'ミドルペース'
    
# ===== タイム分布 → 着順MC =====
half_life_days = int(half_life_m * 30.4375) if half_life_m > 0 else 99999

time_model_pkg = fit_time_quantile_model(
    df_hist=_df.copy(),
    dist_turn_today_df=_dfturn.copy(),
    half_life_days=half_life_days,
    q_list=(0.2, 0.5, 0.8),
    param_grid=(80, 120, 160),
    random_state=24601
)

if time_model_pkg is not None:
    feats = time_model_pkg['feats']
    mu, sd = time_model_pkg['mu'], time_model_pkg['sd']
    models = time_model_pkg['models']

    todayX = build_today_design(
        horses_today=horses,
        s0_hist=s0,
        target_distance=int(TARGET_DISTANCE),
        target_surface=str(TARGET_SURFACE),
        dist_turn_today_df=_dfturn,
        feats=feats,
        pace_type=pace_type  # ← 明示渡し
    )

    v = todayX[feats].astype(float).to_numpy()
    v = np.where(np.isfinite(v), v, time_model_pkg['col_means'])
    vs = (v - mu) / sd

    Q20 = models[0.2].predict(vs)
    Q50 = models[0.5].predict(vs)
    Q80 = models[0.8].predict(vs)

    z08 = 0.8416212335729143
    sigma_from_span = (Q80 - Q20) / (2.0 * z08)
    sigma_raw = np.maximum(sigma_from_span, 0.25)

    sigma_hat = time_model_pkg['sigma_hat']
    sigma = np.sqrt(0.5 * sigma_raw ** 2 + 0.5 * sigma_hat ** 2)

    pred_time = pd.DataFrame({
        '馬名': todayX['馬名'],
        'PredTime_s': Q50,
        'PredTime_p20': Q20,
        'PredTime_p80': Q80,
        'PredSigma_s': sigma
    })

    draws_mc = 120000
    rng_t = np.random.default_rng(13579)
    names = pred_time['馬名'].tolist()
    mu_vec = pred_time['PredTime_s'].to_numpy(float)
    sig_vec = pred_time['PredSigma_s'].to_numpy(float)
    n = len(mu_vec)

    tau = 0.30 * float(np.nanmedian(sig_vec))
    E_id = rng_t.normal(size=(draws_mc, n)) * sig_vec[None, :]
    E_cm = rng_t.normal(size=(draws_mc, 1)) * tau
    T = mu_vec[None, :] + E_id + E_cm

    rk = np.argsort(T, axis=1)
    win = np.bincount(rk[:, 0], minlength=n)
    top3 = np.zeros(n, int)
    for k in range(3):
        top3 += np.bincount(rk[:, k], minlength=n)
    exp_rank = (rk.argsort(axis=1) + 1).mean(axis=0)

    pred_time['勝率%_TIME'] = (100.0 * win / draws_mc).round(2)
    pred_time['複勝率%_TIME'] = (100.0 * top3 / draws_mc).round(2)
    pred_time['期待着順_TIME'] = np.round(exp_rank, 3)

    df_agg = df_agg.merge(pred_time, on='馬名', how='left')
else:
    df_agg['PredTime_s'] = np.nan
    df_agg['PredTime_p20'] = np.nan
    df_agg['PredTime_p80'] = np.nan
    df_agg['PredSigma_s'] = np.nan
    df_agg['勝率%_TIME'] = np.nan
    df_agg['複勝率%_TIME'] = np.nan
    df_agg['期待着順_TIME'] = np.nan

# ===== PhysS1（コース幾何）を 予測タイム入り で再計算 → 全馬へ付与 =====
try:
    # レース代表タイム（各馬PredTime中央値）
    race_pred_time = float(pd.to_numeric(df_agg.get('PredTime_s'), errors='coerce').median()) \
                     if 'PredTime_s' in df_agg.columns else np.nan

    # 幾何の解決
    lay_ok, rail_ok, geom, used_d, is_fb = resolve_course_geom(
        COURSE_ID,
        "芝" if TARGET_SURFACE == "芝" else "ダ",
        int(TARGET_DISTANCE),
        LAYOUT,
        RAIL
    )

    # ← ここは try の「中」に入れる（SyntaxError 回避）
    if geom is None:
        raise ValueError(
            f"未対応のコース設定: {COURSE_ID}/{TARGET_SURFACE}/{int(TARGET_DISTANCE)}m/{LAYOUT}-{RAIL}"
        )
    if is_fb:
        st.warning("course_geometry に登録が無い距離のため、簡易ジオメトリで PhysS1 を計算しました（保守的な推定）。")

    # ★ ここから PhysS1 本計算はフォールバックの有無に関わらず“常に”実行
    fld_size = int(len(horses)) if len(horses) > 0 else None
    races_df_today = []
    for _, r in horses.iterrows():
        gate_no = pd.to_numeric(r.get('番', np.nan), errors='coerce')
        races_df_today.append({
            'race_id': 'TODAY',
            'course_id': COURSE_ID,
            'surface': "芝" if TARGET_SURFACE == "芝" else "ダ",
            'distance_m': int(TARGET_DISTANCE),
            'layout': lay_ok,
            'rail_state': rail_ok,
            'num_turns': 2,
            'final_time_sec': race_pred_time if np.isfinite(race_pred_time) else None,
            'gate_no': gate_no,
            'field_size': fld_size,
            'break_loss_sec': 0.0,
            '馬名': r.get('馬名', None)
        })
    races_df_today = pd.DataFrame(races_df_today)

    phys1 = add_phys_s1_features(
        races_df_today,
        group_cols=("race_id",),
        band_col=None,
        verbose=False
    ).rename(columns={
        'phys_corner_load': 'CornerLoadS1',
        'phys_start_cost': 'StartCostS1',
        'phys_finish_grade': 'FinishGradeS1',
        'phys_s1_score': 'PhysS1'
    })

    df_agg = df_agg.merge(
        phys1[['馬名','CornerLoadS1','StartCostS1','FinishGradeS1','PhysS1']],
        on='馬名', how='left'
    )
    df_agg['FinalRaw'] += float(PHYS_S1_GAIN) * pd.to_numeric(df_agg['PhysS1'], errors='coerce').fillna(0.0)

except Exception as e:
    st.warning(f"PhysS1の計算に失敗しました: {e}")
    for k in ['CornerLoadS1','StartCostS1','FinishGradeS1','PhysS1']:
        df_agg[k] = np.nan



# PacePts反映
df_agg['PacePts'] = pd.to_numeric(df_agg['PacePts'], errors='coerce').fillna(0.0)
df_agg['FinalRaw'] += float(pace_gain) * df_agg['PacePts']


# === Bias-lite（場×距離帯×回り：内中外×脚質の軽い補正） ===
try:
    s0b = _df[['場名','距離','回り','枠','脚質','確定着順']].copy()
    s0b['距離帯'] = (pd.to_numeric(s0b['距離'], errors='coerce')/200).round()*200
    s0b['band'] = pd.cut(pd.to_numeric(s0b['枠'], errors='coerce'),
                         bins=[0,3,5,8], labels=['内','中','外'])
    s0b['win'] = (pd.to_numeric(s0b['確定着順'], errors='coerce')==1).astype(int)

    key = (COURSE_ID, int(TARGET_DISTANCE//200*200), TARGET_TURN)
    sub = s0b[(s0b['場名']==key[0]) & (s0b['距離帯']==key[1]) & (s0b['回り']==key[2])].dropna(subset=['band','脚質'])
    if len(sub) >= 50:
        g = sub.groupby(['band','脚質'])['win'].agg(['mean','count']).reset_index()
        # Wilson補正で中心化した差分 → ±0.8pt の範囲で反映
        z=1.96
        p=g['mean'].to_numpy(float); n=g['count'].to_numpy(float)
        phat=(p + z*z/(2*n)) / (1 + z*z/n)
        adj = (phat - phat.mean())
        bias_map = {}
        for (_, row), val in zip(g.iterrows(), adj):
            bias_map[(row['band'], row['脚質'])] = float(np.clip(val*4.0, -0.8, 0.8))
        df_agg['BiasPts'] = [
            bias_map.get((
                '内' if (int(r.get('枠',0)) in [1,2,3]) else ('中' if int(r.get('枠',0)) in [4,5] else '外'),
                str(r.get('脚質',''))
            ), 0.0) for _, r in df_agg.iterrows()
        ]
        df_agg['FinalRaw'] += df_agg['BiasPts'].fillna(0.0)
except Exception:
    pass
# === Traffic-lite（短1角×多頭数×外枠×出脚不安の簡易リスク） ===
try:
    field = max(1, len(df_agg))
    short_1c = 1 if LAYOUT in ['内回り'] else (0 if LAYOUT in ['外回り','直線'] else 0)
    k1, k2, k3, k4 = 0.06, 0.30, 0.10, 0.20  # 係数は軽め（過信抑制）
    risk = []
    for _, r in df_agg.iterrows():
        waku = pd.to_numeric(r.get('枠'), errors='coerce')
        stl  = str(r.get('脚質',''))
        base = k1*max(0, field-12) + k2*short_1c + (k3*max(0, (waku-5)/3) if np.isfinite(waku) else 0.0)
        esc  = k4*(1.0 if stl in ['逃げ','先行'] else 0.0)  # 出脚で逃げ先行は緩和
        risk.append(np.clip(base - esc, 0.0, 1.5))
    df_agg['TrafficPts'] = risk
    df_agg['FinalRaw'] -= df_agg['TrafficPts'].fillna(0.0)
except Exception:
    pass
# === 自動重み調整（不確実性でMath寄りを弱め、Spec/Physを少し足す） ===
if USE_AUTO_BALANCER:
    try:
        # 既存のバランサ中身はそのまま流用
        wM, wP = 0.65, 0.35
        neff = pd.to_numeric(df_agg['n_eff_turn'], errors='coerce').fillna(0.0)
        sigma = pd.to_numeric(df_agg.get('PredSigma_s'), errors='coerce')
        sigma_med = float(np.nanmedian(sigma)) if np.isfinite(sigma).any() else 0.0
        u_M = np.clip((1.0/(1.0 + neff/3.0)) + (0.5 * (sigma / (sigma_med + 1e-9)).median(skipna=True)), 0.0, 0.30)

        spec_ratio_safe = float(spec_ratio)
        pace_amp = float(np.abs(df_agg.get('PacePts', 0)).median())
        u_P = np.clip(spec_ratio_safe*0.2 + pace_amp*0.1, 0.0, 0.20)

        wM_ = np.clip(wM - float(u_M), 0.30, 0.80)
        wP_ = 1.0 - wM_

        Z_math = pd.to_numeric(df_agg['FinalRaw'], errors='coerce').fillna(df_agg['FinalRaw'].median())
        Z_math = (Z_math - Z_math.mean()) / (Z_math.std() + 1e-9)

        # ★★ FIX: スペクトル側にも UI の係数を必ず掛ける
        Z_spec = (
            float(spectral_weight_ui)
            * pd.to_numeric(df_agg.get('SpecFitZ'), errors='coerce').fillna(0.0)
        )

        Z_phys = ((pd.to_numeric(df_agg.get('PhysicsZ'), errors='coerce') - 50.0) / 10.0).fillna(0.0)
        Z_ph   = (spec_ratio_safe * Z_spec + (1.0 - spec_ratio_safe) * Z_phys)

        Z_final = wM_ * Z_math + wP_ * Z_ph
        df_agg['FinalRaw'] = 50 + 10 * ((Z_final - np.nanmean(Z_final)) / (np.nanstd(Z_final) + 1e-9))
    except Exception:
        pass

# === 2歳戦: 着順MCトグル ===
if USE_MC:
    # ===== Monte Carlo (FinalRawベースの着順サンプリング) =====
    try:
        rng_mc = np.random.default_rng(20251102)
        sims = 500000  # 回数は好きに増やしてOK

        names_mc = df_agg['馬名'].tolist()
        score_mc = pd.to_numeric(df_agg['FinalRaw'], errors='coerce').to_numpy(float)
        n_h = len(names_mc)

        # Gumbelノイズを足して順位にする
        win_ct = np.zeros(n_h, int)
        top3_ct = np.zeros(n_h, int)
        rank_sum = np.zeros(n_h, float)

        for _ in range(sims):
            # スコアの大きい方が強い前提
            noise = rng_mc.gumbel(loc=0.0, scale=1.0, size=n_h)
            sample = score_mc + noise
            order = np.argsort(-sample)  # 降順
            # 1着
            win_ct[order[0]] += 1
            # 3着まで
            for k in range(min(3, n_h)):
                top3_ct[order[k]] += 1
            # 各馬の順位を足す（期待着順用）
            inv = np.empty(n_h, int)
            inv[order] = np.arange(1, n_h+1)
            rank_sum += inv

        df_agg['勝率%_MC'] = (win_ct / sims * 100).round(2)
        df_agg['複勝率%_MC'] = (top3_ct / sims * 100).round(2)
        df_agg['期待着順_MC'] = (rank_sum / sims).round(3)

    except Exception as e:
        st.warning(f"Monte Carloブロックでエラー: {e}")
else:
    if 'df_agg' in locals():
        for c in ('勝率%_MC','複勝率%_MC','期待着順_MC'):
            df_agg[c] = np.nan

# OFFのときは FinalRaw_before_balance をそのまま使う（※ここでは何もしない）


# ===== BT学習用データを整形（週×クラスロバスト値も併記可）=====
bt_base = _df[['馬名','確定着順','rid_hist','レース日','ClassKey','WeekKey','ScoreT_RB']].dropna(subset=['馬名','確定着順','rid_hist']).copy()
bt_base['確定着順'] = pd.to_numeric(bt_base['確定着順'], errors='coerce')
bt_base = bt_base[bt_base['確定着順'] >= 1]

bt_base = (
    bt_base
    .groupby('rid_hist', group_keys=False)
    .filter(lambda g: len(g) >= 2 and g['確定着順'].nunique() >= 2)
)
if bt_base.empty:
    # ここで止めない。後段のフォールバックで均等に落とす
    pass
# 時系列重み（例：ロバストに寄せてやや緩め）
bt_base['_w_time'] = 0.5 ** ((pd.Timestamp.today() - pd.to_datetime(bt_base['レース日'], errors='coerce')).dt.days.clip(lower=0) / (half_life_m*30.4375 if half_life_m>0 else 180.0))

# 追加で“重要レース”に重みを足す（例：G1/G2に+20%など）
def _class_boost(c):
    return 1.2 if c in ['G1','G2'] else (1.1 if c=='G3' else 1.0)
bt_base['_w_cls'] = bt_base['ClassKey'].map(_class_boost).fillna(1.0)

bt_base['_w'] = bt_base['_w_time'] * bt_base['_w_cls']

# ===== ペア勝敗 → BT学習 =====
names_bt, N_bt, M_bt = _build_pairwise_from_ranks(
    bt_base, rid_col='rid_hist', name_col='馬名', rank_col='確定着順', weight_col='_w'
)
worth_bt = fit_bradley_terry(names_bt, N_bt, M_bt)  # Series(index=馬名, value=worth)

# ★ 追記: 行列が実質ゼロ or 学習結果が空なら、均等重みにフォールバック
try:
    total_comp = float(np.nansum(M_bt)) if isinstance(M_bt, np.ndarray) else 0.0
except Exception:
    total_comp = 0.0

if (worth_bt is None) or (len(worth_bt) == 0) or (not np.isfinite(total_comp) or total_comp <= 0):
    worth_bt = pd.Series(1.0, index=names_bt, dtype=float)

# ===== 今日の勝率（BT） =====
runners = horses['馬名'].astype(str).tolist()

# 学習結果が空/ほぼゼロのときは均等フォールバック
try:
    total_comp = float(np.nansum(M_bt)) if isinstance(M_bt, np.ndarray) else 0.0
except Exception:
    total_comp = 0.0
if (worth_bt is None) or (len(worth_bt) == 0) or (not np.isfinite(total_comp) or total_comp <= 0):
    worth_bt = pd.Series(1.0, index=names_bt, dtype=float)

# 今日の出走馬に対応する強さベクトル
w_today = pd.Series([worth_bt.get(nm, np.nan) for nm in runners], index=runners, dtype=float)

# すべて欠損→均等、欠損→バックオフ
if not np.isfinite(w_today).any():
    w_today = pd.Series(1.0, index=runners, dtype=float)
w_back = float(np.nanmean(w_today)) if np.isfinite(np.nanmean(w_today)) else (1.0 / max(len(runners), 1))
w_today = w_today.fillna(0.8 * w_back)

# 正規化
p_today = w_today / max(w_today.sum(), 1e-12)

# ★ dict→map で“安定当て込み”（None 混入を防ぐ）
bt_map = p_today.to_dict()
df_agg['勝率%_BT'] = (
    df_agg['馬名'].astype(str).map(lambda n: bt_map.get(str(n), np.nan))
    .astype(float).mul(100.0).round(2)
)

# 念のため：数値化を強制（Styler の "None" 表示回避）
df_agg['勝率%_BT'] = pd.to_numeric(df_agg['勝率%_BT'], errors='coerce')



# ===== 勝率（PL解析解）＆ Top3（Gumbel反対称） =====
calibrator = None
if do_calib and SK_ISO:
    dfh = _df.dropna(subset=['score_adj','確定着順']).copy()
    dfh['rid'] = _make_race_id_for_hist(dfh)
    X, Y = [], []
    for _, g in dfh.groupby('rid'):
        xs = g['score_adj'].astype(float).to_numpy()
        y  = (pd.to_numeric(g['確定着順'], errors='coerce') == 1).astype(int).to_numpy()
        if len(xs) >= 2 and np.unique(y).size >= 2:
            p = np.exp(beta_pl * (xs - xs.max()))
            s = p.sum()
            if np.isfinite(s) and s > 0:
                p = np.clip(p / s, 1e-6, 1-1e-6)
                X.append(p); Y.append(y)
    if X:
        calibrator = IsotonicRegression(out_of_bounds='clip').fit(np.concatenate(X), np.concatenate(Y))

S = pd.to_numeric(df_agg['FinalRaw'], errors='coerce')
finite = np.isfinite(S)

p_win = np.zeros(len(S), float)
if finite.any():
    A = S[finite].to_numpy(float)
    m = A.mean(); s = A.std() + 1e-9
    z = (A - m) / s
    x = beta_pl * (z - z.max())
    ex = np.exp(x)
    p = ex / ex.sum()
    p_win[finite] = p
    if (~finite).any():
        p_win[~finite] = 1e-6
        p_win = p_win / p_win.sum()
else:
    p_win[:] = 1.0 / max(len(S), 1)

if calibrator is not None:
    p_win = safe_iso_predict(calibrator, p_win)

df_agg['勝率%_PL'] = (100 * p_win).round(2)

# Top3近似（Gumbel反対称）
if finite.any():
    A = S.copy()
    A[~finite] = A[finite].mean()
    m = A.mean(); s = A.std() + 1e-9
    abilities = ((A - m) / s).to_numpy(float)
else:
    abilities = np.zeros(len(S), float)

draws_top3 = 80000
rng3 = np.random.default_rng(13579)
G = rng3.gumbel(size=(draws_top3//2, len(abilities)))
U = np.vstack([beta_pl*abilities[None,:] + G, beta_pl*abilities[None,:] - G])
rank_idx = np.argsort(-U, axis=1)
counts = np.zeros(len(abilities), float)
for k in range(3):
    counts += np.bincount(rank_idx[:,k], minlength=len(abilities)).astype(float)
df_agg['複勝率%_PL'] = (100*(counts / draws_top3)).round(2)

# ===== H) AR100 =====
S = pd.to_numeric(df_agg['FinalRaw'], errors='coerce')
S = S.fillna(S.median())

# --- 不確実性に基づく Z 縮約（上振れ抑制） ---
sig = pd.to_numeric(df_agg.get('PredSigma_s'), errors='coerce')  # 予測タイムの不確かさ
sig_med = float(np.nanmedian(sig)) if np.isfinite(sig).any() else 0.0
neff = pd.to_numeric(df_agg['n_eff_turn'], errors='coerce').fillna(0.0)

# 縮約係数 u: 0< u ≤1
u = 1.0 / np.sqrt(
        1.0
        + (AR_SHRINK_SIGMA * (sig / (sig_med + 1e-9)))**2
        + (AR_SHRINK_NEFF  / (1.0 + neff))  # データ薄いほど寄与↑
    )
u = np.clip(u.fillna(1.0), 0.5, 1.0)

# フィールド内Z（縮約後）
mu = float(S.mean()); sd = float(S.std(ddof=0) or 1.0)
z = (S - mu) / (sd + 1e-9)
z_eff = z * u

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

if AR_MODE.startswith("厳格"):
    # 1) 基礎スコア（erfで強圧縮：上が出にくい）
    base = 50.0 + AR_BASE_A * np.vectorize(math.erf)(
        z_eff / (np.sqrt(2.0) * AR_STRICT_S)
    )
    base = np.clip(base, 0.0, 99.999)

    # 2) 尾部ブースト（トップだけ 100 に緩やかに近づく・連続関数）
    #    sig ~ logistic((z - T)/τ) の p 乗で非常に厳しく
    sigm = logistic((z_eff - TAIL_THRESH) / TAIL_TAU)
    tail_gain = (100.0 - base) * np.power(sigm, TAIL_POW)
    ar = np.clip(base + tail_gain, 0.0, 100.0)
    df_agg['AR100'] = ar

elif AR_MODE.startswith("スコアベース"):
    # 従来Z→0..100（上が出やすい）
    ar = 50.0 + 10.0 * (S - mu) / (sd or 1.0)
    df_agg['AR100'] = ar.clip(0, 100)

else:
    # 順位ベース（上限は100のまま、ただし上位差分はγで圧縮）
    ranks = S.rank(method='average', pct=True).fillna(0.5).to_numpy(float)
    ranks_eff = np.power(ranks, float(AR_RANK_GAMMA))
    df_agg['AR100'] = np.interp(ranks_eff, [0.0, 1.0], [20.0, 100.0])

# --- バンド（厳しめ） ---
def to_band(v):
    if not np.isfinite(v): return 'E'
    if v >= 96: return 'SS'
    if v >= 90: return 'S'
    if v >= 82: return 'A'
    if v >= 74: return 'B'
    if v >= 66: return 'C'
    return 'E'

df_agg['Band'] = df_agg['AR100'].map(to_band)


# ===== テーブル整形（日本語ラベル付き） =====
# ▼ スペクトル列を追加した完成版
_dfdisp = df_agg.copy().sort_values(['AR100','勝率%_PL'], ascending=[False, False]).reset_index(drop=True)
_dfdisp['順位'] = np.arange(1, len(_dfdisp)+1)

# ---- クイックサマリー（上位の要約） ----
if '_dfdisp' not in globals() or _dfdisp.empty:
    st.error("集計結果が空です。入力やマッピングを確認してください。")
    st.stop()

# 表示ラベル（最小限）
JP = {
    '順位':'順位','枠':'枠','番':'馬番','馬名':'馬名','脚質':'脚質',
    'AR100':'AR100','Band':'評価帯',
    '勝率%_PL':'勝率%（PL）','複勝率%_PL':'複勝率%（PL）',
    '勝率%_TIME':'勝率%（タイム）','複勝率%_TIME':'複勝率%（タイム）','期待着順_TIME':'期待着順（タイム）',
    'PredTime_s':'予測タイム中央値[s]','PredTime_p20':'20%速い側[s]','PredTime_p80':'80%遅い側[s]','PredSigma_s':'タイム分散σ[s]',
    'RecencyZ':'近走Z','StabZ':'安定性Z','PacePts':'ペースPts','TurnPrefPts':'回り加点','DistTurnZ':'距離×回りZ',
    'SpecFitZ':'スペクトル適合Z',
    'SpecGate_horse_lbl':'走法型(馬)','SpecGate_templ_lbl':'走法型(テンプレ)',
    'PhysicsZ':'物理Z','PhysS1':'PhysS1','CornerLoadS1':'コーナー荷重','StartCostS1':'スタート損失','FinishGradeS1':'ゴール前勾配',
    '勝率%_BT':'勝率%（BT）','勝率%_MC':'勝率%（MC）','複勝率%_MC':'複勝率%（MC）','期待着順_MC':'期待着順（MC）',
}

# ===== 印（能力上位順）=====
# ルール: ◎1頭, 〇1頭, ▲1頭, ☆1頭, △3頭, 残りは「消」
# デフォルトは AR100 → 同値は 勝率%_PL → RecencyZ の順でタイブレーク
rank_key = ['AR100','勝率%_PL','RecencyZ']
for k in rank_key:
    if k not in _dfdisp.columns:
        _dfdisp[k] = np.nan
_dfdisp['_sort_tuple'] = list(
    zip(_dfdisp['AR100'].fillna(-1e9),
        _dfdisp['勝率%_PL'].fillna(-1e9),
        _dfdisp['RecencyZ'].fillna(-1e9))
)
_dfdisp = _dfdisp.sort_values('_sort_tuple', ascending=False).reset_index(drop=True)

marks = ['◎','〇','▲','☆','△','△','△']
印 = ['消'] * len(_dfdisp)
for i, m in enumerate(marks):
    if i < len(印):
        印[i] = m
_dfdisp['印'] = 印

# 主要表示列
disp_cols = [
    '順位','印','枠','番','馬名','脚質',
    'AR100','Band','勝率%_PL','複勝率%_PL',
    '勝率%_TIME','複勝率%_TIME','期待着順_TIME',
    'PredTime_s','PredSigma_s',
    'SpecFitZ','SpecGate_horse_lbl','SpecGate_templ_lbl',
    'PhysicsZ','PhysS1','PacePts','TurnPrefPts','DistTurnZ'
]
disp_cols = [c for c in disp_cols if c in _dfdisp.columns]

table = _dfdisp[disp_cols].copy()
table = table.rename(columns=JP)

render_final_view(table)

# ===== 購入案（上位6頭ボックス / 一頭軸フォーメーションの雛形）=====
top6 = _dfdisp.head(6)['馬名'].tolist()
head = _dfdisp.iloc[0]['馬名'] if len(_dfdisp)>0 else None

with st.expander("🧾 買い目ひな形（上位6頭ベース）", expanded=True):
    st.markdown(f"- 上位6頭（ボックス候補）: **{', '.join(top6)}**" if top6 else "- 上位6頭なし")
    if head:
        st.markdown(f"- 一頭軸（軸候補）: **{head}**")

# ===== ダウンロード（CSV / JSON）=====
export_cols = [c for c in _dfdisp.columns if c != '_sort_tuple']
res_csv = _dfdisp[export_cols].to_csv(index=False).encode('utf-8-sig')
st.download_button("⬇️ 結果CSVダウンロード", data=res_csv, file_name="result_rikeiba.csv", mime="text/csv")

# JSON（軽量サマリー）
json_cols = ['印','枠','番','馬名','脚質','AR100','勝率%_PL','複勝率%_PL','PredTime_s','PredSigma_s','SpecFitZ','PhysicsZ','PhysS1']
json_cols = [c for c in json_cols if c in _dfdisp.columns]
res_json = _dfdisp[json_cols].to_dict(orient='records')
st.download_button("⬇️ 結果JSONダウンロード", data=json.dumps(res_json, ensure_ascii=False, indent=2).encode('utf-8'),
                   file_name="result_rikeiba.json", mime="application/json")

# ===== ちょい可視化（Plotly版 AR100 vs 馬番）=====
st.markdown("## 📈 AR100 と 馬番の関係")
show_labels = st.checkbox("ラベルを表示する（重なる場合があります）", value=False)
df_plot = _dfdisp.copy()
if '番' in df_plot.columns and '馬番' not in df_plot.columns:
    df_plot['馬番'] = df_plot['番']
fig = plot_ar_scatter(
    df_plot,
    x_col="馬番" if "馬番" in df_plot.columns else "番",
    y_col="AR100",
    name_col="馬名",
    waku_col="枠",
    show_text=show_labels,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### 印（コピペ用）")
marks_text = build_marks_text(
    _dfdisp,
    name_col="馬名",
    mark_col="印",
    ar_col="AR100",
    ar_label="AR",
    ar_only_for_top=True,
)
st.code(marks_text, language="text")

st.success("完了：スコア算出・印付け・出力を生成しました。")

