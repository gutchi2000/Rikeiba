# keiba_web_app_final.py
# サイドバー互換（expander）、縦軸堅牢化、年齢/枠重み・MC・血統HTML 完備
# + 右/左回り 自動判定&加点 + AR100/Bandキャリブレーション（B中心化）+ 見送りロジック
# + ★ 先頭の空行を確実に除去 / 4角ポジション図の安全化
# + ★ 脚質エディタの値をセッションに保存・復元（リランしても消えない）
# + ★ 脚質の表記ゆれを吸収（「追い込み」「差込」など→正規化）
# + ★ ndcg_by_race を堅牢実装でトップレベルに定義（重複import/インデント崩れ修正）
# + ★ 18桁 race_id → NetKeiba 12桁 に自動変換（学習前処理）
# + ★ 【置き換え】勝率は Plackett–Luce（softmax）で解析計算、Top3は軽量MC。等温回帰で確率校正（任意）。
# + ★ 【削除予定】LightGBM学習UI/ロジック（後半で完全削除＆評価タブへ差し替え）
# app.py（あなたの既存アプリ）


import streamlit as st
import pandas as pd
import numpy as np
import re, io, json
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Wedge, Rectangle, Circle
from itertools import combinations

# ===== Optional: Altair（散布図） =====
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

# ===== Optional: HTML 埋め込み（血統表示用） =====
try:
    import streamlit.components.v1 as components
    COMPONENTS = True
except Exception:
    COMPONENTS = False

# ===== Optional: 確率校正（等温回帰 / Platt用ロジット） =====
try:
    from sklearn.isotonic import IsotonicRegression
    SK_ISO = True
except Exception:
    SK_ISO = False

try:
    from sklearn.linear_model import LogisticRegression
    SK_PLATT = True
except Exception:
    SK_PLATT = False

# ---- 基本設定とフォント ----
# ---- 日本語フォント設定（置き換え）----
import os
from matplotlib import font_manager

@st.cache_resource
def get_jp_font():
    # 同梱 or OSの代表的な場所を順に探索
    candidates = [
        "ipaexg.ttf",  # リポジトリ直下に置いた場合（推奨：IPAexGothic）
        "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/Hiragino Sans W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)  # ← 実体を登録
            except Exception:
                pass
            return font_manager.FontProperties(fname=p)
    return None

jp_font = get_jp_font()

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の豆腐対策
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP',
    'Yu Gothic UI', 'Meiryo', 'Hiragino Sans', 'MS Gothic'
]

# 見つかった実フォント名を優先で適用
if jp_font is not None:
    try:
        plt.rcParams['font.family'] = jp_font.get_name()
    except Exception:
        pass

st.set_page_config(page_title="競馬予想アプリ（修正版）", layout="wide")

# ---- 便利CSS（sidebar 幅だけ調整）----
# 枠→HEX
# ── 枠の色（近似：JRA/NetKeibaの枠色）
def _waku_hex(v: int) -> str:
    palette = {1:"#ffffff",2:"#000000",3:"#e6002b",4:"#1560bd",5:"#ffd700",6:"#00a04b",7:"#ff7f27",8:"#f19ec2"}
    return palette.get(int(v), "#ffffff")

def _style_waku(s: pd.Series):
    out = []
    for v in s:
        if pd.isna(v):
            out.append("")
        else:
            v = int(v)
            bg = _waku_hex(v)
            fg = "#000" if v == 1 else "#fff"   # ← 1枠は黒文字、それ以外は白文字
            out.append(f"background-color:{bg}; color:{fg}; font-weight:700; text-align:center;")
    return out

STYLES = ['逃げ','先行','差し','追込']
_fwid = str.maketrans('０１２３４５６７８９％','0123456789%')

# === NEW: 脚質の表記ゆれ吸収（正規化関数） ===
STYLE_ALIASES = {
    '追い込み':'追込','追込み':'追込','おいこみ':'追込','おい込み':'追込',
    'さし':'差し','差込':'差し','差込み':'差し',
    'せんこう':'先行','先行 ':'先行','先行　':'先行',
    'にげ':'逃げ','逃げ ':'逃げ','逃げ　':'逃げ'
}
def normalize_style(s: str) -> str:
    s = str(s).replace('　','').strip().translate(_fwid)
    s = STYLE_ALIASES.get(s, s)
    return s if s in STYLES else ''

# ======================== ユーティリティ ========================
def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

def z_score(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([50]*len(s), index=s.index)
    return 50 + 10 * (s - s.mean()) / std

def _parse_time_to_sec(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = str(x).strip()
    m = re.match(r'^(\d+):(\d+)\.(\d+)$', s)
    if m: return int(m.group(1))*60 + int(m.group(2)) + float('0.'+m.group(3))
    m = re.match(r'^(\d+)[\.:](\d+)[\.:](\d+)$', s)
    if m: return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/10
    try:  return float(s)
    except: return np.nan

def normalize_grade_text(x: str | None) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)): 
        return None
    s = str(x).translate(_fwid)
    s = (s.replace('Ｇ','G').replace('（','(').replace('）',')')
         .replace('Ⅰ','I').replace('Ⅱ','II').replace('Ⅲ','III'))
    s = re.sub(r'G\s*III', 'G3', s, flags=re.I)
    s = re.sub(r'G\s*II',  'G2', s, flags=re.I)
    s = re.sub(r'G\s*I',   'G1', s, flags=re.I)
    s = re.sub(r'ＪＰＮ', 'Jpn', s, flags=re.I)
    s = re.sub(r'JPN',    'Jpn', s, flags=re.I)
    s = re.sub(r'Jpn\s*III', 'Jpn3', s, flags=re.I)
    s = re.sub(r'Jpn\s*II',  'Jpn2', s, flags=re.I)
    s = re.sub(r'Jpn\s*I',   'Jpn1', s, flags=re.I)
    m = re.search(r'(?:G|Jpn)\s*([123])', s, flags=re.I)
    return f"G{m.group(1)}" if m else None

def safe_take(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()

def _trim_name(x):
    try:
        return str(x).replace('\u3000',' ').strip()
    except Exception:
        return str(x)

# 安定した重み付き標準偏差（不偏補正つき）
def w_std_unbiased(x, w, ddof=1):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    if not np.isfinite(sw) or sw <= 0: return np.nan
    m = np.sum(w * x) / sw
    var = np.sum(w * (x - m)**2) / sw
    n_eff = (sw**2) / np.sum(w**2) if np.sum(w**2) > 0 else 0
    if ddof and n_eff > ddof:
        var *= n_eff / (n_eff - ddof)
    return float(np.sqrt(max(var, 0.0)))
# --- 距離×回り 近傍化の補助 ---
def _kish_neff(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    sw = w.sum()
    s2 = np.sum(w ** 2)
    return float((sw * sw) / s2) if s2 > 0 else 0.0

def _nw_mean(x, y, w, h):
    # Nadaraya–Watson (Gaussian kernel)
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    if len(x) == 0: return np.nan
    K = np.exp(-0.5 * (x / max(1e-9, h)) ** 2) * w
    sK = K.sum()
    return float((K * y).sum() / sK) if sK > 0 else np.nan

def _dist_turn_profile_for_horse(
    name: str, df_hist: pd.DataFrame, target_distance: int, target_turn: str, surface: str,
    h_m: float, opp_turn_w: float, prior_mode: str, tau: float,
    grid_step: int, grid_span_m: int, df_agg_for_prior: pd.DataFrame
):
    g = df_hist[df_hist['馬名'].astype(str).str.strip() == str(name)].copy()
    # 距離/回り/馬場の前処理
    g['距離'] = pd.to_numeric(g['距離'], errors='coerce')
    g = g[g['距離'].notna() & (g['距離'] > 0)]
    if '馬場' in g.columns:
        g = g[g['馬場'].astype(str) == str(surface)]
    if '回り' not in g.columns or g.empty:
        return {'DistTurnZ': np.nan, 'n_eff_turn': 0.0, 'BestDist_turn': np.nan, 'DistTurnZ_best': np.nan}

    # 基本重み：時間減衰（_w）× 回り係数 × 1
    w_time = pd.to_numeric(g.get('_w', 1.0), errors='coerce').fillna(1.0).to_numpy(float)
    w_turn = np.where(g['回り'].astype(str) == str(target_turn), 1.0, float(opp_turn_w))
    w0 = w_time * w_turn

    x_dist = g['距離'].to_numpy(float)
    y_z    = pd.to_numeric(g['score_norm'], errors='coerce').to_numpy(float)
    msk    = np.isfinite(x_dist) & np.isfinite(y_z) & np.isfinite(w0)
    x_dist, y_z, w0 = x_dist[msk], y_z[msk], w0[msk]
    if x_dist.size == 0:
        return {'DistTurnZ': np.nan, 'n_eff_turn': 0.0, 'BestDist_turn': np.nan, 'DistTurnZ_best': np.nan}

    # 事前平均 μ0
    mu0 = np.nan
    if prior_mode == "WAvgZベース":
        mu0 = float(df_agg_for_prior.set_index('馬名').get('WAvgZ', pd.Series()).get(name, np.nan))
    elif prior_mode == "Right/LeftZベース":
        if str(target_turn) == '右':
            mu0 = float(df_agg_for_prior.set_index('馬名').get('RightZ', pd.Series()).get(name, np.nan))
        else:
            mu0 = float(df_agg_for_prior.set_index('馬名').get('LeftZ', pd.Series()).get(name, np.nan))
        if not np.isfinite(mu0):
            mu0 = float(df_agg_for_prior.set_index('馬名').get('WAvgZ', pd.Series()).get(name, np.nan))

    # ターゲット距離での後平均（擬似ベイズ: τ で事前を混合）
    z_hat_t = _nw_mean(x_dist - float(target_distance), y_z, w0, h_m)
    w_eff_t = _kish_neff(np.exp(-0.5 * ((x_dist - float(target_distance))/max(1e-9, h_m))**2) * w0)

    if np.isfinite(mu0) and tau > 0:
        if np.isfinite(z_hat_t):
            z_hat_t = (w_eff_t * z_hat_t + tau * mu0) / max(1e-9, (w_eff_t + tau))
        else:
            z_hat_t = mu0

    # プロファイル走査（ベスト距離の参考用）
    ds = np.arange(int(target_distance - grid_span_m),
                   int(target_distance + grid_span_m) + 1, int(grid_step))
    best_d, best_val = np.nan, -1e18
    for d0 in ds:
        z = _nw_mean(x_dist - float(d0), y_z, w0, h_m)
        if np.isfinite(mu0) and tau > 0:
            # ベイズ混合（有効本数を疑似サンプルとして加算）
            we = _kish_neff(np.exp(-0.5 * ((x_dist - float(d0))/max(1e-9, h_m))**2) * w0)
            z = (we * z + tau * mu0) / max(1e-9, (we + tau)) if np.isfinite(z) else mu0
        if np.isfinite(z) and z > best_val:
            best_val, best_d = float(z), int(d0)

    return {
        'DistTurnZ': float(z_hat_t) if np.isfinite(z_hat_t) else np.nan,
        'n_eff_turn': float(w_eff_t),
        'BestDist_turn': float(best_d) if np.isfinite(best_d) else np.nan,
        'DistTurnZ_best': float(best_val) if np.isfinite(best_val) else np.nan
    }

@st.cache_data(show_spinner=False)
def load_excel_bytes(content: bytes):
    xls = pd.ExcelFile(io.BytesIO(content))
    s0 = pd.read_excel(xls, sheet_name=0)
    s1 = pd.read_excel(xls, sheet_name=1)
    return s0, s1

def validate_inputs(df_score: pd.DataFrame, horses: pd.DataFrame):
    problems = []
    req = ['馬名','レース日','競走名','頭数','確定着順']
    for c in req:
        if c not in df_score.columns:
            problems.append(f"sheet0 必須列が見つからない: {c}")
    if '斤量' in horses:
        bad = horses['斤量'].dropna()
        if len(bad)>0 and ((bad<45)|(bad>65)).any():
            problems.append("sheet1 斤量がレンジ外（45–65）")
    if {'通過4角','頭数'}.issubset(df_score.columns):
        tmp = df_score[['通過4角','頭数']].dropna()
        if len(tmp)>0 and ((tmp['通過4角']<1) | (tmp['通過4角']>tmp['頭数'])).any():
            problems.append("sheet0 通過4角が頭数レンジ外")
    if problems: st.warning("⚠ 入力チェック：\n- " + "\n- ".join(problems))

# ===== 右/左回りユーティリティ =====
DEFAULT_VENUE_TURN = {
    '札幌':'右','函館':'右','福島':'右','新潟':'左','東京':'左',
    '中山':'右','中京':'左','京都':'右','阪神':'右','小倉':'右'
}
VENUE_SYNONYM_TO_VENUE = {'府中':'東京','淀':'京都'}

def _normalize_turn_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = { '場名':None, '競走名':None, '回り':None, '正規表現':None }
    for c in df.columns:
        s = str(c).strip()
        if s in ['場名','競馬場','コース','開催','開催場']: cols['場名']=c
        if s in ['競走名','レース名','名称']:              cols['競走名']=c
        if s in ['回り','右左','向き']:                    cols['回り']=c
        if s.lower() in ['regex','正規表現','正規表現?']:   cols['正規表現']=c
    out = pd.DataFrame()
    if cols['場名'] is not None:    out['場名'] = df[cols['場名']].astype(str).str.strip()
    if cols['競走名'] is not None:  out['競走名'] = df[cols['競走名']].astype(str).str.strip()
    if cols['回り'] is not None:
        out['回り'] = (df[cols['回り']].astype(str)
                       .str.replace('回り','').str.replace('周り','').str.strip().str[:1])
    else:
        out['回り'] = None
    out['正規表現'] = (df[cols['正規表現']] if cols['正規表現'] is not None else False).astype(bool)
    out = out[(out['回り'].isin(['右','左']))]
    return out

def _infer_venue_from_racename(name: str) -> str | None:
    s = str(name)
    for v in DEFAULT_VENUE_TURN.keys():
        if v in s: return v
    for syn, venue in VENUE_SYNONYM_TO_VENUE.items():
        if syn in s: return venue
    return None

def _attach_turn_to_scores(df_score: pd.DataFrame,
                           turn_df: pd.DataFrame | None,
                           use_default: bool=True) -> pd.DataFrame:
    df = df_score.copy()
    if '場名' in df.columns:
        df['_場名推定'] = df['場名'].astype(str).apply(_infer_venue_from_racename)
    else:
        df['_場名推定'] = df['競走名'].astype(str).apply(_infer_venue_from_racename)
    venue_map = {}
    if use_default:
        venue_map.update(DEFAULT_VENUE_TURN)
    if turn_df is not None and '場名' in turn_df.columns:
        for v, t in turn_df[['場名','回り']].dropna().values:
            if str(v).strip():
                venue_map[str(v).strip()] = t
    df['回り'] = df['_場名推定'].map(venue_map)
    if turn_df is not None and '競走名' in turn_df.columns:
        patt = turn_df.dropna(subset=['競走名'])
        for _, row in patt.iterrows():
            pat = str(row['競走名']).strip()
            trn = row['回り']
            is_re = bool(row.get('正規表現', False))
            mask = (df['競走名'].astype(str).str.contains(pat, regex=is_re, na=False)
                    if is_re else
                    df['競走名'].astype(str).str.contains(re.escape(pat), regex=True, na=False))
            df.loc[mask, '回り'] = trn
    return df

# ===== NDCG（レース単位）安全実装 =====
def ndcg_by_race(frame: pd.DataFrame, scores, k: int = 3) -> float:
    """レース単位で NDCG@k を計算（安全版）。"""
    f = frame[['race_id', 'y']].copy().reset_index(drop=True)
    s = np.asarray(scores, dtype=float)
    if len(s) != len(f):
        s = s[:len(f)]
    vals = []
    for _, idx in f.groupby('race_id').groups.items():
        idx = np.asarray(list(idx), dtype=int)
        y_true = np.nan_to_num(f.loc[idx, 'y'].values.astype(float), nan=0.0, neginf=0.0, posinf=0.0)
        y_pred = np.nan_to_num(s[idx].astype(float),                nan=0.0, neginf=0.0, posinf=0.0)
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

# ======================== サイドバー ========================
st.sidebar.title("⚙️ パラメタ設定")

with st.sidebar.expander("🔰 よく使う（基本）", expanded=True):
    lambda_part  = st.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
    besttime_w   = st.slider("ベストタイム重み", 0.0, 2.0, 1.0)
    with st.expander("戦績率の重み（当該馬場）", expanded=False):
        win_w  = st.slider("勝率の重み",   0.0, 5.0, 1.0, 0.1, key="w_win")
        quin_w = st.slider("連対率の重み", 0.0, 5.0, 0.7, 0.1, key="w_quin")
        plc_w  = st.slider("複勝率の重み", 0.0, 5.0, 0.5, 0.1, key="w_plc")
    with st.expander("各種ボーナス設定", expanded=False):
        grade_bonus  = st.slider("重賞実績ボーナス", 0, 20, 5)
        agari1_bonus = st.slider("上がり3F 1位ボーナス", 0, 10, 3)
        agari2_bonus = st.slider("上がり3F 2位ボーナス", 0, 5, 2)
        agari3_bonus = st.slider("上がり3F 3位ボーナス", 0, 3, 1)
        bw_bonus     = st.slider("馬体重適正ボーナス(±10kg)", 0, 10, 2)
    with st.expander("本レース条件（BT重み用）", expanded=True):
        TARGET_GRADE    = st.selectbox("本レースの格", ["G1", "G2", "G3", "L", "OP"], index=4)
        TARGET_SURFACE  = st.selectbox("本レースの馬場", ["芝", "ダ"], index=0)
        TARGET_DISTANCE = st.number_input("本レースの距離 [m]", min_value=1000, max_value=3600, value=1800, step=100)

with st.sidebar.expander("🛠 詳細（補正/脚質/ペース）", expanded=False):
    half_life_m  = st.slider("時系列半減期(月)", 0.0, 12.0, 6.0, 0.5)
    stab_weight  = st.slider("安定性(小さいほど◎)の係数", 0.0, 2.0, 0.7, 0.1)
    pace_gain    = st.slider("ペース適性係数", 0.0, 3.0, 1.0, 0.1)
    weight_coeff = st.slider("斤量ペナルティ強度(pts/kg)", 0.0, 4.0, 1.0, 0.1)

    with st.expander("斤量ベース（WFA簡略）", expanded=False):
        race_date = pd.to_datetime(st.date_input("開催日", value=pd.Timestamp.today().date()))
        use_wfa_base = st.checkbox("WFA基準を使う（推奨）", value=True)
        wfa_2_early_m = st.number_input("2歳（〜9月） 牡/せん [kg]", 50.0, 60.0, 55.0, 0.5)
        wfa_2_early_f = st.number_input("2歳（〜9月） 牝 [kg]"    , 48.0, 60.0, 54.0, 0.5)
        wfa_2_late_m  = st.number_input("2歳（10-12月） 牡/せん [kg]", 50.0, 60.0, 56.0, 0.5)
        wfa_2_late_f  = st.number_input("2歳（10-12月） 牝 [kg]"    , 48.0, 60.0, 55.0, 0.5)
        wfa_3p_m      = st.number_input("3歳以上 牡/せん [kg]" , 50.0, 62.0, 57.0, 0.5)
        wfa_3p_f      = st.number_input("3歳以上 牝 [kg]"     , 48.0, 60.0, 55.0, 0.5)

    with st.expander("属性重み（1走スコア係数）", expanded=False):
        gender_w = {g: st.slider(f"{g}", 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
        style_w  = {s: st.slider(f"{s}", 0.0, 2.0, 1.0) for s in STYLES}
        season_w = {s: st.slider(f"{s}", 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
        age_w    = {str(age): st.slider(f"{age}歳", 0.0, 2.0, 1.0, 0.05) for age in range(3, 11)}
        frame_w  = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0) for i in range(1,8+1)}

    with st.expander("脚質自動推定 / ペース推定", expanded=False):
        auto_style_on   = st.checkbox("脚質の自動推定を使う（空欄を埋める）", True)
        AUTO_OVERWRITE  = st.checkbox("手入力より自動を優先して上書き", False)
        NRECENT         = st.slider("直近レース数（脚質推定）", 1, 10, 5)
        HL_DAYS_STYLE   = st.slider("半減期（日・脚質用）", 30, 365, 180, 15)
        pace_mc_draws   = st.slider("ペースMC回数", 500, 30000, 5000, 500)
        pace_mode = st.radio("ペースの扱い", ["自動（MC）", "固定（手動）"], index=0)
        pace_fixed = st.selectbox("固定ペース", ["ハイペース","ミドルペース","ややスローペース","スローペース"],
                                  index=1, disabled=(pace_mode=="自動（MC）"))
        epi_alpha = st.slider("逃げ係数 α", 0.0, 2.0, 1.0, 0.05)
        epi_beta  = st.slider("先行係数 β", 0.0, 2.0, 0.60, 0.05)
        thr_hi    = st.slider("閾値: ハイペース ≥", 0.30, 1.00, 0.52, 0.01)
        thr_mid   = st.slider("閾値: ミドル ≥",    0.10, 0.99, 0.30, 0.01)
        thr_slow  = st.slider("閾値: ややスロー ≥",0.00, 0.98, 0.18, 0.01)

# === NEW: 回り（右/左）設定セクション ===
with st.sidebar.expander("🔄 回り（右/左）", expanded=False):
    TARGET_TURN = st.radio("本レースの回り", ["右","左"], index=0, horizontal=True)
    turn_gain   = st.slider("回り適性 係数（FinalRawへ加点）", 0.0, 3.0, 1.0, 0.1)
    turn_gap_thr= st.slider("得意判定の閾値（RightZ−LeftZ の最小差）", 0.0, 10.0, 1.0, 0.1)
    use_default_venue_map = st.checkbox("JRA標準の『場名→回り』で補完する", True)
    st.caption("※ 場名既定表＋競走名から自動推定。")

# === NEW: 距離×回り（精密） ===
with st.sidebar.expander("🎯 距離×回り（精密）", expanded=False):
    USE_DIST_TURN     = st.checkbox("有効化（距離×右左で近傍化）", True)
    dist_bw_m         = st.slider("距離帯の幅 [m]", 50, 600, 200, 25)
    opp_turn_discount = st.slider("逆回りの重み係数", 0.0, 1.0, 0.5, 0.05)
    dist_prior_mode   = st.selectbox("距離適性の事前分布",
                                     ["無し","WAvgZベース","Right/LeftZベース"], index=1)
    dist_tau          = st.slider("事前重み τ（小=事後重視）", 0.0, 5.0, 1.0, 0.1)
    dist_turn_gain    = st.slider("距離×回り 係数（FinalRawへ）", 0.0, 3.0, 1.0, 0.1)
    grid_step         = st.slider("プロファイル距離刻み [m]", 50, 400, 100, 50)
    grid_span_m       = st.slider("プロファイルの±範囲 [m]", 200, 1200, 600, 50)
    st.caption("近傍化: Nadaraya–Watson（ガウス核） / 有効本数: Kish の式")

# === バンド校正（B中心化） ===
with st.sidebar.expander("🏷 バンド校正（B中心化）", expanded=True):
    band_mid_target = st.slider("中央値→何点に合わせる？", 40, 80, 65, 1,
                                help="AR100でレースの真ん中を何点に置くか（Bの真ん中=65推奨）")
    band_A_share = st.slider("A以上の目標割合(%)", 1, 60, 25, 1,
                             help="AR100が70以上（A, S, SS）になる頭数の割合ターゲット。Bを厚くしたいなら小さめに。")
    band_clip_lo = st.slider("下限クリップ", 0, 60, 40, 1)
    band_clip_hi = st.slider("上限クリップ", 80, 100, 100, 1)

# === NEW: PL/Top3と保存系 ===
with st.sidebar.expander("🧪 勝率化 / 保存", expanded=False):
    mc_iters   = st.slider("Top3近似サンプル数（軽量MC）", 1000, 50000, 8000, 1000)
    mc_beta    = st.slider("PL温度 β（大=鋭い）", 0.3, 5.0, 1.4, 0.1)
    mc_seed    = st.number_input("乱数Seed", 0, 999999, 42, 1)
    st.markdown("---")
    total_budget = st.slider("合計予算", 500, 50000, 10000, 100)
    min_unit     = st.selectbox("最小賭け単位", [100, 200, 300, 500], index=0)
    max_lines    = st.slider("最大点数(連系)", 1, 60, 20, 1)
    st.markdown("---")

    # ---- JSON保存を安全化（非シリアライズ品を文字列化） ----
    def _jsonable(x):
        import numpy as _np
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, (_np.integer,)):
            return int(x)
        if isinstance(x, (_np.floating,)):
            return float(x)
        if isinstance(x, (_np.bool_,)):
            return bool(x)
        if isinstance(x, (list, tuple, set)):
            return [_jsonable(i) for i in x]
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        return str(x)

    col_a, col_b = st.columns(2)
    if col_a.button("設定を保存"):
        try:
            cfg_dict = {}
            for k, v in st.session_state.items():
                if str(k).startswith('_'):
                    continue
                if k in ('excel_up','cfg_up','pedi_html_up'):
                    continue
                cfg_dict[k] = _jsonable(v)
            cfg = json.dumps(cfg_dict, ensure_ascii=False, indent=2)
            st.download_button("JSONをDL", data=cfg, file_name="keiba_config.json", mime="application/json")
        except Exception as e:
            st.error(f"設定保存に失敗しました: {e}")

    cfg_file = col_b.file_uploader("設定読み込み", type=["json"], key="cfg_up")
    if cfg_file is not None:
        try:
            cfg = json.loads(cfg_file.read().decode("utf-8"))
            for k,v in cfg.items():
                st.session_state[k]=v
            st.success("設定を読み込みました（必要なら再実行）。")
        except Exception as e:
            st.error(f"設定ファイルの読み込みエラー: {e}")

# === NEW: 確率校正の設定 ===
with st.sidebar.expander("📏 確率校正（任意）", expanded=False):
    do_calib = st.checkbox("勝率を校正する（等温回帰/Platt）", value=False)
    calib_method = st.radio("校正法", ["Isotonic（推奨）","Platt（ロジット）"], index=0, horizontal=True, disabled=not do_calib)
    st.caption("※ sheet0の過去走からレース内softmax→1着ラベルで学習。芝/ダ分割などは後半の評価タブで。")

# === 表示設定 ===
with st.sidebar.expander("🖥 表示設定", expanded=True):
    FULL_TABLE_VIEW = st.checkbox("表は全頭表示（内部スクロールを無くす）", value=True)
    MAX_TABLE_HEIGHT = st.slider("全頭表示の最大高さ(px)", 800, 10000, 5000, 200)

def auto_table_height(n_rows: int, row_px: int = 35, header_px: int = 38, pad_px: int = 28) -> int:
    h = int(header_px + row_px * max(1, int(n_rows)) + pad_px)
    return min(h, int(MAX_TABLE_HEIGHT))

def H(df, default_px: int) -> int:
    try:
        return auto_table_height(len(df)) if FULL_TABLE_VIEW else int(default_px)
    except Exception:
        return int(default_px)

# === 便利リセット ===
with st.sidebar.expander("🧹 トラブル時のリセット", expanded=False):
    if st.button("列マッピングをリセット"):
        for key in list(st.session_state.keys()):
            if key.startswith("s0:") or key.startswith("map:s0:") or key.startswith("s1:") or key.startswith("map:s1:"):
                del st.session_state[key]
        st.success("列マッピングをリセットしました。Excelを再読み込みしてください。")

# ======================== ファイルアップロード ========================
st.title("競馬予想アプリ（修正版）")
st.subheader("ファイルアップロード")
excel_file = st.file_uploader("Excel（sheet0=過去走 / sheet1=出走表）", type=['xlsx'], key="excel_up")
if excel_file is None:
    st.info("まずExcel（.xlsx）をアップロードしてください。")
    st.stop()

sheet0, sheet1 = load_excel_bytes(excel_file.getvalue())

# === sheet0 / sheet1 マッピング ===
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'\s+', '', s)
    s = s.translate(str.maketrans('０１２３４５６７８９','0123456789'))
    s = s.replace('（','(').replace('）',')').replace('％','%')
    return s
def _auto_guess(col_map, pats):
    for orig, normed in col_map.items():
        for p in pats:
            if re.search(p, normed, flags=re.I):
                return orig
    return None

def _interactive_map(df, patterns, required_keys, title, state_key, show_ui=False):
    cols = list(df.columns)
    cmap = {c: _norm_col(c) for c in cols}

    auto = {k: (st.session_state.get(f"{state_key}:{k}") or _auto_guess(cmap, pats))
            for k, pats in patterns.items()}

    for k, v in list(auto.items()):
        if v not in cols:
            auto[k] = None

    if not show_ui:
        missing = [k for k in required_keys if not auto.get(k)]
        if not missing:
            for k, v in auto.items():
                if v:
                    st.session_state[f"{state_key}:{k}"] = v
            return auto
        else:
            st.warning(f"{title} の必須列が自動認識できませんでした: " + ", ".join(missing))
            show_ui = True

    with st.expander(f"列マッピング：{title}", expanded=True):
        mapping = {}
        for key, pats in patterns.items():
            default = st.session_state.get(f"{state_key}:{key}") or auto.get(key)
            options = ['<未選択>'] + cols
            idx = options.index(default) if (default in cols) else 0
            mapping[key] = st.selectbox(key, options=options, index=idx, key=f"map:{state_key}:{key}")
            if mapping[key] != '<未選択>':
                st.session_state[f"{state_key}:{key}"] = mapping[key]

    missing = [k for k in required_keys if mapping.get(k) in (None, '<未選択>')]
    if missing:
        st.stop()

    return {k: (None if v=='<未選択>' else v) for k, v in mapping.items()}

# パターン
PAT_S0 = {
    '馬名'         : [r'馬名|名前|出走馬'],
    'レース日'     : [r'レース日|日付(?!S)|年月日|施行日|開催日'],
    '競走名'       : [r'競走名|レース名|名称'],
    'クラス名'     : [r'クラス名|格|条件|レースグレード'],
    '頭数'         : [r'頭数|出走頭数'],
    '確定着順'     : [r'確定着順|着順(?!率)'],
    '枠'           : [r'枠\b|枠番'],
    '番'           : [r'馬番|番'],
    '斤量'         : [r'斤量'],
    '馬体重'       : [r'馬体重|体重'],
    '上がり3Fタイム': [r'上がり3Fタイム|上がり3F|上3Fタイム|上3F|Ave-?3F'],
    '上3F順位'     : [r'上がり3F順位|上3F順位'],
    '通過4角'      : [r'通過.*4角|4角.*通過|第4コーナー.*|4角順位|4角通過順'],
    '性別'         : [r'性別'],
    '年齢'         : [r'年齢|馬齢'],
    '走破タイム秒' : [r'走破タイム.*秒|走破タイム|タイム$'],
    '距離'         : [r'距離'],
    '馬場'         : [r'馬場(?!.*指数)|馬場状態'],
    '場名'         : [r'場名|場所|競馬場|開催(地|場|場所)'],
}
REQ_S0 = ['馬名','レース日','競走名','頭数','確定着順']
MAP_S0 = _interactive_map(sheet0, PAT_S0, REQ_S0, "sheet0（過去走）", "s0", show_ui=False)

df_score = pd.DataFrame()
for k, col in MAP_S0.items():
    if (col is None) or (col not in sheet0.columns):
        continue
    df_score[k] = sheet0[col]

df_score['レース日'] = pd.to_datetime(df_score['レース日'], errors='coerce')
for c in ['頭数','確定着順','枠','番','斤量','馬体重','上3F順位','通過4角','距離']:
    if c in df_score: df_score[c] = pd.to_numeric(df_score[c], errors='coerce')
if '走破タイム秒' in df_score: df_score['走破タイム秒'] = df_score['走破タイム秒'].apply(_parse_time_to_sec)
if '上がり3Fタイム' in df_score: df_score['上がり3Fタイム'] = df_score['上がり3Fタイム'].apply(_parse_time_to_sec)

# 通過4角/頭数の整形
if '頭数' in df_score.columns:
    df_score['頭数'] = (
        df_score['頭数'].astype(str).str.extract(r'(\d+)')[0]
        .apply(pd.to_numeric, errors='coerce')
    )
if '通過4角' in df_score.columns:
    s = df_score['通過4角']
    if s.dtype.kind not in 'iu':
        last_num = s.astype(str).str.extract(r'(\d+)(?!.*\d)')[0]
        df_score['通過4角'] = pd.to_numeric(last_num, errors='coerce')
    ok = df_score['頭数'].notna() & df_score['通過4角'].notna()
    bad = ok & ((df_score['通過4角'] < 1) | (df_score['通過4角'] > df_score['頭数']))
    df_score.loc[df_score['通過4角'].eq(0), '通過4角'] = np.nan
    df_score.loc[bad, '通過4角'] = np.nan

# === sheet1（当日出走表） ===
PAT_S1 = {
    '馬名'   : [r'馬名|名前|出走馬'],
    '枠'     : [r'枠\b|枠番'],
    '番'     : [r'馬番|番'],
    '性別'   : [r'性別'],
    '年齢'   : [r'年齢|馬齢'],
    '斤量'   : [r'斤量'],
    '馬体重' : [r'馬体重|体重'],
    '脚質'   : [r'脚質'],
    '勝率'   : [r'勝率(?!.*率)|\b勝率\b'],
    '連対率' : [r'連対率|連対'],
    '複勝率' : [r'複勝率|複勝'],
    'ベストタイム': [r'ベスト.*タイム|Best.*Time|ﾍﾞｽﾄ.*ﾀｲﾑ|タイム.*(最速|ベスト)'],
}
REQ_S1 = ['馬名','枠','番','性別','年齢']
MAP_S1 = _interactive_map(sheet1, PAT_S1, REQ_S1, "sheet1（出走表）", "s1", show_ui=False)

attrs = pd.DataFrame()
for k, col in MAP_S1.items():
    if (col is None) or (col not in sheet1.columns):
        continue
    attrs[k] = sheet1[col]
for c in ['枠','番','斤量','馬体重']:
    if c in attrs: attrs[c] = pd.to_numeric(attrs[c], errors='coerce')
if 'ベストタイム' in attrs: attrs['ベストタイム秒'] = attrs['ベストタイム'].apply(_parse_time_to_sec)

# ★★★ 先頭の空行（全列NaN）や馬名が空の行を除去してからエディタへ ★★★
attrs = attrs.replace(r'^\s*$', np.nan, regex=True)   # 空文字→NaN
attrs = attrs.dropna(how='all')                       # 全列NaN行を削除
if '馬名' in attrs.columns:
    attrs['馬名'] = attrs['馬名'].astype(str).str.replace('\u3000',' ').str.strip()
    attrs = attrs[attrs['馬名'].ne('')]
attrs = attrs.reset_index(drop=True)

# 入力UI（脚質・斤量・馬体重編集）
if '脚質' not in attrs.columns: attrs['脚質'] = ''
if '斤量' not in attrs.columns: attrs['斤量'] = np.nan
if '馬体重' not in attrs.columns: attrs['馬体重'] = np.nan

# === NEW: エディタの内容をセッションから復元（リラン対策） ===
if 'horses_df' in st.session_state and isinstance(st.session_state['horses_df'], pd.DataFrame) and not st.session_state['horses_df'].empty:
    prev = st.session_state['horses_df'][['馬名','脚質','斤量','馬体重']].copy()
    attrs = attrs.merge(prev, on='馬名', how='left', suffixes=('','_prev'))
    for c in ['脚質','斤量','馬体重']:
        attrs[c] = attrs[c].where(attrs[c].notna() & (attrs[c] != ''), attrs.get(f'{c}_prev'))
    drop_cols = [f'{c}_prev' for c in ['脚質','斤量','馬体重'] if f'{c}_prev' in attrs.columns]
    if drop_cols:
        attrs.drop(columns=drop_cols, inplace=True)

st.subheader("馬一覧・脚質・斤量・当日馬体重入力")

with st.form("horses_form", clear_on_submit=False):
    edited = st.data_editor(
        attrs[['枠','番','馬名','性別','年齢','脚質','斤量','馬体重']].copy(),
        column_config={
            '脚質': st.column_config.SelectboxColumn('脚質', options=STYLES),
            '斤量': st.column_config.NumberColumn('斤量', min_value=45, max_value=65, step=0.5),
            '馬体重': st.column_config.NumberColumn('馬体重', min_value=300, max_value=600, step=1)
        },
        use_container_width=True,
        num_rows='static',
        height=auto_table_height(len(attrs)) if st.session_state.get('FULL_TABLE_VIEW', True) else 420,
        hide_index=True,
        key="horses_editor",
    )

    # 全頭埋まっているか確認（空文字・NaNを許さない）
    need_cols = ['脚質']  # 斤量や馬体重も必須にするなら追加: '斤量','馬体重'
    def _all_filled(df, cols):
        ok = True
        for c in cols:
            s = df[c].astype(str).str.strip()
            ok &= s.ne('').all()
        return bool(ok)

    submitted = st.form_submit_button("全頭入力完了 → 計算")

if not submitted:
    st.info("脚質を全頭入力して『全頭入力完了 → 計算』を押すと集計します。")
    st.stop()

horses = edited.copy()
# この後は従来どおり
horses['脚質'] = horses['脚質'].map(normalize_style)
st.session_state['horses_df'] = horses.copy()


validate_inputs(df_score, horses)

# --- 脚質 自動推定（略、既存ロジック維持） ---
# （ここは元のコードと同一のため省略せず残しています）
df_style = pd.DataFrame({'馬名': [], 'p_逃げ': [], 'p_先行': [], 'p_差し': [], 'p_追込': [], '推定脚質': []})
need_cols = {'馬名','レース日','頭数','通過4角'}
if auto_style_on and need_cols.issubset(df_score.columns):
    tmp = (
        df_score[['馬名','レース日','頭数','通過4角','上3F順位']].copy()
        .dropna(subset=['馬名','レース日','頭数','通過4角'])
        .sort_values(['馬名','レース日'], ascending=[True, False])
    )
    tmp['_rn'] = tmp.groupby('馬名').cumcount()+1
    tmp = tmp[tmp['_rn'] <= int(NRECENT)].copy()
    today = pd.Timestamp.today()
    tmp['_days'] = (today - pd.to_datetime(tmp['レース日'], errors='coerce')).dt.days.clip(lower=0).fillna(9999)
    tmp['_w'] = 0.5 ** (tmp['_days'] / float(HL_DAYS_STYLE))

    denom = (pd.to_numeric(tmp['頭数'], errors='coerce') - 1).replace(0, np.nan)
    pos_ratio = (pd.to_numeric(tmp['通過4角'], errors='coerce') - 1) / denom
    pos_ratio = pos_ratio.clip(0, 1).fillna(0.5)

    if '上3F順位' in tmp.columns:
        ag = pd.to_numeric(tmp['上3F順位'], errors='coerce')
        close_strength = ((3.5 - ag) / 3.5).clip(lower=0, upper=1).fillna(0.0)
    else:
        close_strength = pd.Series(0.0, index=tmp.index)

    b_nige, b_sengo, b_sashi, b_oikomi = -1.2, 0.6, 0.3, -0.7
    tmp['L_nige']   = b_nige  + 1.6*(1 - pos_ratio) - 1.2*close_strength
    tmp['L_sengo']  = b_sengo + 1.1*(1 - pos_ratio) - 0.1*close_strength
    tmp['L_sashi']  = b_sashi + 1.1*(pos_ratio)     + 0.9*close_strength
    tmp['L_oikomi'] = b_oikomi+ 1.6*(pos_ratio)     + 0.5*close_strength

    rows = []
    for name, g in tmp.groupby('馬名'):
        w  = g['_w'].to_numpy(); sw = w.sum()
        if sw <= 0: continue
        def wavg(v): return float((v*w).sum()/sw)
        vec = np.array([wavg(g['L_nige']), wavg(g['L_sengo']), wavg(g['L_sashi']), wavg(g['L_oikomi'])], dtype=float)
        vec = vec - vec.max()
        p = np.exp(vec); p = p / p.sum()
        pred = STYLES[int(np.argmax(p))]

        pr = (pd.to_numeric(g['通過4角'], errors='coerce') - 1) / (pd.to_numeric(g['頭数'], errors='coerce') - 1)
        pr = pr.clip(0,1).fillna(0.5)
        wpr = float((pr*w).sum()/sw)
        if pred == '逃げ' and not (wpr <= 0.22 or ((pr <= 0.15)*w).sum()/sw >= 0.25): pred='先行'
        if pred == '追込' and not (wpr >= 0.78 or ((pr >= 0.85)*w).sum()/sw >= 0.25): pred='差し'

        rows.append([name, *p.tolist(), pred])
    if rows:
        df_style = pd.DataFrame(rows, columns=['馬名','p_逃げ','p_先行','p_差し','p_追込','推定脚質'])
        manual_has_nige = ('脚質' in horses.columns) and horses['脚質'].eq('逃げ').any()
        if (df_style['推定脚質'].eq('逃げ').sum() == 0) and (not manual_has_nige):
            early = tmp.assign(early=(1 - pos_ratio).clip(0, 1), w=tmp['_w'].values)\
                      .groupby('馬名').apply(lambda g: float((g['early']*g['w']).sum()/g['w'].sum()))
            nige_cand = early.idxmax()
            df_style.loc[df_style['馬名'] == nige_cand, '推定脚質'] = '逃げ'

# --- 戦績率・ベストタイム ---
rate_cols = [c for c in ['勝率','連対率','複勝率'] if c in attrs.columns]
if rate_cols:
    rate = attrs[['馬名'] + rate_cols].copy()
    for c in rate_cols:
        rate[c] = rate[c].astype(str).str.replace('%','', regex=False).str.replace('％','', regex=False)
        rate[c] = pd.to_numeric(rate[c], errors='coerce')
    mx = pd.concat([rate[c] for c in rate_cols], axis=1).max().max()
    if pd.notna(mx) and mx <= 1.0:
        for c in rate_cols: rate[c] *= 100.0
    if 'ベストタイム秒' in attrs:
        rate = rate.merge(attrs[['馬名','ベストタイム秒']], on='馬名', how='left')
else:
    rate = pd.DataFrame({'馬名':[]})

# === 過去走から「適正馬体重」推定 ===
best_bw_map = {}
if {'馬名','馬体重','確定着順'}.issubset(df_score.columns):
    _bw = df_score[['馬名','馬体重','確定着順']].dropna()
    _bw['確定着順'] = pd.to_numeric(_bw['確定着順'], errors='coerce')
    _bw = _bw[_bw['確定着順'].notna()]
    try:
        best_idx = _bw.groupby('馬名')['確定着順'].idxmin()
        best_bw_map = _bw.loc[best_idx].set_index('馬名')['馬体重'].astype(float).to_dict()
    except Exception:
        best_bw_map = {}

# 重複ガード
try:
    horses.drop_duplicates('馬名', keep='first', inplace=True)
except Exception:
    pass
try:
    df_score = df_score.drop_duplicates(subset=['馬名','レース日','競走名'], keep='first')
except Exception:
    pass

# ===== マージ =====
for dup in ['枠','番','性別','年齢','斤量','馬体重','脚質']:
    df_score.drop(columns=[dup], errors='ignore', inplace=True)
df_score = df_score.merge(horses[['馬名','枠','番','性別','年齢','斤量','馬体重','脚質']], on='馬名', how='left')
if len(rate) > 0:
    use_cols = ['馬名'] + [c for c in ['勝率','連対率','複勝率','ベストタイム秒'] if c in rate.columns]
    df_score = df_score.merge(rate[use_cols], on='馬名', how='left')

# ===== ベストタイム重み =====
bt_min = df_score['ベストタイム秒'].min(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_max = df_score['ベストタイム秒'].max(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_span = (bt_max - bt_min) if pd.notna(bt_min) and pd.notna(bt_max) and (bt_max > bt_min) else 1.0
CLASS_BASE_BT = {"OP": 1.50, "L": 1.38, "G3": 1.19, "G2": 1.00, "G1": 0.80}
def besttime_weight_final(grade: str, surface: str, distance_m: int, user_scale: float) -> float:
    base = CLASS_BASE_BT.get(str(grade), CLASS_BASE_BT["OP"])
    s = 1.10 if str(surface) == "ダ" else 1.00
    try:
        d = int(distance_m)
        if d <= 1400:           dfac = 1.20
        elif d == 1600:         dfac = 1.10
        elif 1800 <= d <= 2200: dfac = 1.00
        elif d >= 2400:         dfac = 0.85
        else:                   dfac = 1.00
    except: dfac = 1.00
    w = base * s * dfac * float(user_scale)
    return float(np.clip(w, 0.0, 2.0))

CLASS_PTS = {'G1':10, 'G2':8, 'G3':6, 'リステッド':5, 'オープン特別':4}
def class_points(row) -> int:
    g = normalize_grade_text(row.get('クラス名')) if 'クラス名' in row else None
    if not g and '競走名' in row:
        g = normalize_grade_text(row.get('競走名'))
    if g in CLASS_PTS: return CLASS_PTS[g]
    name = str(row.get('クラス名','')) + ' ' + str(row.get('競走名',''))
    if re.search(r'3\s*勝', name): return 3
    if re.search(r'2\s*勝', name): return 2
    if re.search(r'1\s*勝', name): return 1
    if re.search(r'新馬|未勝利', name): return 1
    if re.search(r'オープン', name): return 4
    if re.search(r'リステッド|L\b', name, flags=re.I): return 5
    return 1

def wfa_base_for(sex: str, age: int | None, dt: pd.Timestamp) -> float:
    try:
        a = int(age) if age is not None and not pd.isna(age) else None
    except Exception:
        a = None
    m = int(dt.month) if isinstance(dt, pd.Timestamp) else 1
    if a == 2:
        male = wfa_2_early_m if m <= 9 else wfa_2_late_m
        filly = wfa_2_early_f if m <= 9 else wfa_2_late_f
    elif a is not None and a >= 3:
        male, filly = wfa_3p_m, wfa_3p_f
    else:
        male, filly = wfa_3p_m, wfa_3p_f
    return male if sex in ("牡", "セ") else filly

# ===== 血統ボーナス：セッションから読む =====
pedi_bonus_pts: float = float(st.session_state.get('pedi:points', 0.0))
_pedi_map_session = st.session_state.get('pedi:map', {})
pedi_bonus_map = { _trim_name(k): bool(v) for k, v in dict(_pedi_map_session).items() }
def _pedi_bonus_for(name: str) -> float:
    nm = _trim_name(name)
    return float(pedi_bonus_pts) if pedi_bonus_map.get(nm, False) else 0.0

def calc_score(r):
    g = class_points(r)
    raw = g * (r['頭数'] + 1 - r['確定着順']) + lambda_part * g

    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r.get('性別'), 1)
    stw = style_w.get(r.get('脚質'), 1)
    fw  = frame_w.get(str(r.get('枠')), 1)
    aw  = age_w.get(str(r.get('年齢')), 1.0)

    gnorm = normalize_grade_text(r.get('クラス名')) or normalize_grade_text(r.get('競走名'))
    grade_point = grade_bonus if gnorm in ['G1','G2','G3'] else 0

    agari_bonus = 0
    try:
        ao = int(r.get('上3F順位', np.nan))
        if   ao == 1: agari_bonus = agari1_bonus
        elif ao == 2: agari_bonus = agari2_bonus
        elif ao == 3: agari_bonus = agari3_bonus
    except: pass

    body_bonus = 0
    try:
        name = r['馬名']
        now_bw = float(r.get('馬体重', np.nan))
        tekitai = float(best_bw_map.get(name, np.nan))
        if not np.isnan(now_bw) and not np.isnan(tekitai) and abs(now_bw - tekitai) <= 10:
            body_bonus = bw_bonus
    except Exception: pass

    rate_bonus = 0.0
    try:
        if '勝率' in r and pd.notna(r.get('勝率', np.nan)):   rate_bonus += win_w  * (float(r['勝率'])  / 100.0)
        if '連対率' in r and pd.notna(r.get('連対率', np.nan)): rate_bonus += quin_w * (float(r['連対率']) / 100.0)
        if '複勝率' in r and pd.notna(r.get('複勝率', np.nan)): rate_bonus += plc_w  * (float(r['複勝率'])  / 100.0)
    except: pass

    bt_bonus = 0.0
    try:
        if pd.notna(r.get('ベストタイム秒', np.nan)):
            bt_norm = (bt_max - float(r['ベストタイム秒'])) / bt_span
            bt_norm = max(0.0, min(1.0, bt_norm))
            bt_w_final = besttime_weight_final(
                grade=TARGET_GRADE, surface=TARGET_SURFACE, distance_m=int(TARGET_DISTANCE), user_scale=besttime_w
            )
            bt_bonus = bt_w_final * bt_norm
    except Exception:
        pass

    kg_pen = 0.0
    try:
        kg = float(r.get('斤量', np.nan))
        if not np.isnan(kg):
            base = wfa_base_for(r.get('性別',''), int(r.get('年齢')) if pd.notna(r.get('年齢',np.nan)) else None, race_date) if use_wfa_base else 56.0
            delta = kg - float(base)
            kg_pen = (-max(0.0,  delta) * float(weight_coeff)
                      + 0.5 * max(0.0, -delta) * float(weight_coeff))
    except Exception:
        pass

    pedi_bonus = _pedi_bonus_for(r['馬名'])

    total_bonus = grade_point + agari_bonus + body_bonus + rate_bonus + bt_bonus + kg_pen + pedi_bonus
    return raw * sw * gw * stw * fw * aw + total_bonus

# 1走→正規化
if 'レース日' not in df_score.columns:
    st.error("レース日 列が見つかりません。Excelの1枚目に含めてください。")
    st.stop()
df_score['score_raw']  = df_score.apply(calc_score, axis=1)
if df_score['score_raw'].max() == df_score['score_raw'].min():
    df_score['score_norm'] = 50.0
else:
    df_score['score_norm'] = (
        (df_score['score_raw'] - df_score['score_raw'].min()) /
        (df_score['score_raw'].max() - df_score['score_raw'].min()) * 100
    )

# ===== 時系列加重 =====
now = pd.Timestamp.today()
df_score['_days_ago'] = (now - df_score['レース日']).dt.days
df_score['_w'] = 0.5 ** (df_score['_days_ago'] / (half_life_m * 30.4375)) if half_life_m > 0 else 1.0

# ===== 回り情報の付与 =====
turn_df = None
df_score = _attach_turn_to_scores(df_score, turn_df, use_default=use_default_venue_map)

# ===== 右/左回り 適性の集計 =====
g_turn = df_score[['馬名','score_norm','_w','回り']].dropna(subset=['馬名','score_norm','_w'])
g_turn['馬名'] = g_turn['馬名'].map(_trim_name)

def _wavg_row(s: pd.DataFrame) -> float:
    sw = float(pd.to_numeric(s['_w'], errors='coerce').sum())
    if sw <= 0:
        return float('nan')
    num = (pd.to_numeric(s['score_norm'], errors='coerce') * pd.to_numeric(s['_w'], errors='coerce')).sum()
    return float(num / sw)

def _make_weighted(df_sub: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if df_sub.empty:
        return pd.DataFrame(columns=[col_name])
    res = df_sub.groupby('馬名', sort=False).apply(_wavg_row)
    if isinstance(res, pd.Series):
        return res.rename(col_name).to_frame()
    elif isinstance(res, pd.DataFrame):
        if res.shape[1] == 0:
            return pd.DataFrame(columns=[col_name])
        return res.iloc[:, 0].rename(col_name).to_frame()
    else:
        return pd.DataFrame({col_name: pd.Series(res, index=pd.Index([], name='馬名'))})

right = _make_weighted(g_turn[g_turn['回り'].astype(str) == '右'], 'RightZ')
left  = _make_weighted(g_turn[g_turn['回り'].astype(str) == '左'], 'LeftZ')

cnts = (
    g_turn.pivot_table(index='馬名', columns='回り', values='score_norm', aggfunc='count')
         .rename(columns={'右':'nR','左':'nL'})
) if not g_turn.empty else pd.DataFrame()

turn_pref = pd.concat([right,left,cnts], axis=1).reset_index() \
    if len(right)+len(left)>0 else pd.DataFrame(columns=['馬名','RightZ','LeftZ','nR','nL'])

if len(turn_pref) > 0:
    turn_pref['馬名'] = turn_pref['馬名'].map(_trim_name)

for c in ['RightZ','LeftZ','nR','nL']:
    if c not in turn_pref.columns:
        turn_pref[c] = np.nan

def _pref_label(row):
    R, L = row['RightZ'], row['LeftZ']
    nR, nL = row['nR'], row['nL']
    if pd.notna(R) and pd.notna(L):
        if (R - L) >= turn_gap_thr: return '右'
        if (L - R) >= turn_gap_thr: return '左'
        return '中立'
    if pd.notna(R) and (nR >= 1): return '右?'
    if pd.notna(L) and (nL >= 1): return '左?'
    return '不明'

turn_pref['TurnPref'] = turn_pref.apply(_pref_label, axis=1) if len(turn_pref)>0 else []
turn_pref['TurnGap']  = (turn_pref['RightZ'].fillna(0) - turn_pref['LeftZ'].fillna(0)) if len(turn_pref)>0 else []

def _pref_pts(row):
    lab = str(row['TurnPref'])
    if TARGET_TURN == '右':
        if lab == '右':  return 1.0
        if lab == '左':  return -1.0
        if lab == '右?': return 0.5
        if lab == '左?': return -0.5
        return 0.0
    else:
        if lab == '左':  return 1.0
        if lab == '右':  return -1.0
        if lab == '左?': return 0.5
        if lab == '右?': return -0.5
        return 0.0

if len(turn_pref)>0:
    turn_pref['TurnPrefPts'] = turn_pref.apply(_pref_pts, axis=1)
else:
    turn_pref['TurnPrefPts'] = []

# ===== 馬ごとの集計 =====
def w_mean(x, w):
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    s = w.sum()
    return float((x*w).sum()/s) if s>0 else np.nan

agg = []
for name, g in df_score.groupby('馬名'):
    avg  = g['score_norm'].mean()
    std  = g['score_norm'].std(ddof=0)
    wavg = w_mean(g['score_norm'], g['_w'])
    wstd = w_std_unbiased(g['score_norm'], g['_w'], ddof=1) if len(g) >= 2 else np.nan
    agg.append({'馬名':name,'AvgZ':avg,'Stdev':std,'WAvgZ':wavg,'WStd':wstd,'Nrun':len(g)})

df_agg = pd.DataFrame(agg)

# —— WStd の穴埋め
wstd_nontrivial = df_agg.loc[df_agg['Nrun']>=2, 'WStd']
default_std = float(wstd_nontrivial.median()) if (wstd_nontrivial.notna().any()) else 6.0
df_agg['WStd'] = df_agg['WStd'].fillna(default_std)
min_floor = max(1.0, default_std*0.6)
df_agg.loc[df_agg['WStd'] < min_floor, 'WStd'] = min_floor

# ===== 脚質統合（手入力優先→自動） =====
for df in [horses, df_agg]:
    if '馬名' in df.columns: df['馬名'] = df['馬名'].map(_trim_name)

if '脚質' in horses.columns:
    horses['脚質'] = horses['脚質'].map(normalize_style)

name_list = df_agg['馬名'].tolist()
combined_style = pd.Series(index=name_list, dtype=object)
if '脚質' in horses.columns:
    combined_style.update(horses.set_index('馬名')['脚質'])
if not df_style.empty and auto_style_on:
    pred_series = df_style.set_index('馬名')['推定脚質']
    if AUTO_OVERWRITE:
        combined_style.update(pred_series)
    else:
        mask_blank = combined_style.isna() | combined_style.astype(str).str.strip().eq('')
        combined_style.loc[mask_blank] = pred_series.reindex(combined_style.index)[mask_blank]
combined_style = combined_style.fillna('')
df_agg['脚質'] = df_agg['馬名'].map(combined_style)

# ===== P行列（脚質確率）→ ペースMC =====
Hn = len(name_list)
P = np.zeros((Hn, 4), dtype=float)
pmap = None
if not df_style.empty and '馬名' in df_style.columns:
    df_prob = df_style.rename(columns={
        'p_差込': 'p_差し','p_追い込み': 'p_追込','p_追込み': 'p_追込',
        'p_逃げ率': 'p_逃げ','p_先行率': 'p_先行','p_差し率': 'p_差し','p_追込率': 'p_追込',
    })
    need_cols = ['p_逃げ','p_先行','p_差し','p_追込']
    if set(need_cols).issubset(df_prob.columns):
        for c in need_cols:
            df_prob[c] = pd.to_numeric(df_prob[c], errors='coerce').fillna(0.0)
        pmap = (df_prob[['馬名'] + need_cols].set_index('馬名')[need_cols])
        row_sums = pmap.sum(axis=1).replace(0, np.nan)
        pmap = pmap.div(row_sums, axis=0).fillna(0.0)

for i, nm in enumerate(name_list):
    stl = combined_style.get(nm, '')
    if not AUTO_OVERWRITE and (stl in STYLES):
        P[i, :] = 0.0; P[i, STYLES.index(stl)] = 1.0; continue
    if pmap is not None and nm in pmap.index:
        P[i, :] = pmap.loc[nm, ['p_逃げ','p_先行','p_差し','p_追込']].to_numpy(dtype=float)
        if P[i, :].sum() == 0:
            if stl in STYLES: P[i, :] = 0.0; P[i, STYLES.index(stl)] = 1.0
            else: P[i, :] = np.array([0.25,0.25,0.25,0.25])
    elif stl in STYLES:
        P[i, :] = 0.0; P[i, STYLES.index(stl)] = 1.0
    else:
        P[i, :] = np.array([0.25,0.25,0.25,0.25])

mark_rule = {
    'ハイペース':      {'逃げ':'△','先行':'△','差し':'◎','追込':'〇'},
    'ミドルペース':    {'逃げ':'〇','先行':'◎','差し':'〇','追込':'△'},
    'ややスローペース': {'逃げ':'〇','先行':'◎','差し':'△','追込':'×'},
    'スローペース':    {'逃げ':'◎','先行':'〇','差し':'△','追込':'×'},
}
mark_to_pts = {'◎':2, '〇':1, '○':1, '△':0, '×':-1}

rng_pace = np.random.default_rng(int(mc_seed) + 12345)
sum_pts = np.zeros(Hn, dtype=float)
pace_counter = {'ハイペース':0,'ミドルペース':0,'ややスローペース':0,'スローペース':0}
for _ in range(int(pace_mc_draws)):
    sampled = [rng_pace.choice(4, p=P[i]) for i in range(Hn)]
    nige  = sum(1 for s in sampled if s==0)
    sengo = sum(1 for s in sampled if s==1)
    epi = (epi_alpha*nige + epi_beta*sengo) / max(1, Hn)
    if   epi >= thr_hi:   pace_t = "ハイペース"
    elif epi >= thr_mid:  pace_t = "ミドルペース"
    elif epi >= thr_slow: pace_t = "ややスローペース"
    else:                 pace_t = "スローペース"
    pace_counter[pace_t] += 1
    mk = mark_rule[pace_t]
    for i, s in enumerate(sampled):
        sum_pts[i] += mark_to_pts[ mk[STYLES[s]] ]
df_agg['PacePts'] = sum_pts / max(1, int(pace_mc_draws))
pace_type = max(pace_counter, key=lambda k: pace_counter[k]) if sum(pace_counter.values())>0 else "ミドルペース"
if pace_mode == "固定（手動）":
    pace_type = pace_fixed
    v_pts = np.array([mark_to_pts[ mark_rule[pace_type][st] ] for st in STYLES], dtype=float)
    df_agg['PacePts'] = (P @ v_pts)

# ===== 回り適性を df_agg へマージ =====
if len(turn_pref)>0:
    df_agg = df_agg.merge(
        turn_pref[['馬名','RightZ','LeftZ','nR','nL','TurnGap','TurnPref','TurnPrefPts']],
        on='馬名', how='left'
    )
else:
    for c in ['RightZ','LeftZ','nR','nL','TurnGap','TurnPref','TurnPrefPts']:
        df_agg[c] = np.nan
    df_agg['TurnPrefPts'] = df_agg['TurnPrefPts'].fillna(0.0)
df_agg['TurnPrefPts'] = df_agg['TurnPrefPts'].fillna(0.0)

# ===== 距離×回り（精密）を df_agg に付与 =====
if USE_DIST_TURN and {'距離','回り','score_norm','_w'}.issubset(df_score.columns):
    rows = []
    for nm in df_agg['馬名'].astype(str):
        prof = _dist_turn_profile_for_horse(
            name=nm, df_hist=df_score,
            target_distance=int(TARGET_DISTANCE),
            target_turn=str(TARGET_TURN),
            surface=str(TARGET_SURFACE),
            h_m=float(dist_bw_m), opp_turn_w=float(opp_turn_discount),
            prior_mode=str(dist_prior_mode), tau=float(dist_tau),
            grid_step=int(grid_step), grid_span_m=int(grid_span_m),
            df_agg_for_prior=df_agg
        )
        rows.append({'馬名': nm, **prof})
    _disttbl = pd.DataFrame(rows)
    df_agg = df_agg.merge(_disttbl, on='馬名', how='left')
else:
    for c in ['DistTurnZ','n_eff_turn','BestDist_turn','DistTurnZ_best']:
        df_agg[c] = np.nan

# ===== 最終スコア（正規化は内部指標） =====
# ここは RecencyZ / StabZ / PacePts / TurnPrefPts / DistTurnZ が揃う“後”に置く

# 1) 必要列の用意（未生成でも0で埋めて落ちないように）
for c in ("PacePts", "TurnPrefPts", "DistTurnZ"):
    if c not in df_agg.columns:
        df_agg[c] = 0.0
df_agg[["PacePts", "TurnPrefPts", "DistTurnZ"]] = df_agg[["PacePts", "TurnPrefPts", "DistTurnZ"]].fillna(0.0)

# 2) RecencyZ / StabZ をここで生成
#    - RecencyZ は WAvgZ（なければ AvgZ）をZ化
base_for_recency = (
    df_agg.get("WAvgZ", pd.Series(np.nan, index=df_agg.index))
      .fillna(df_agg.get("AvgZ", pd.Series(0.0, index=df_agg.index)))
)
df_agg["RecencyZ"] = z_score(pd.to_numeric(base_for_recency, errors="coerce").fillna(0.0))

#    - StabZ は「小さいほど◎」なので WStd にマイナスを掛けてZ化
if "WStd" not in df_agg.columns:
    df_agg["WStd"] = 6.0
wstd_fill = pd.to_numeric(df_agg["WStd"], errors="coerce")
if not np.isfinite(wstd_fill).any():
    wstd_fill = pd.Series(6.0, index=df_agg.index)
df_agg["StabZ"] = z_score(-(wstd_fill.fillna(wstd_fill.median())))

# 3) 距離×回り 係数（スライダーが無い/未反映でも落ちないようフォールバック）
_dist_gain_from_state = st.session_state.get("dist_turn_gain", None)
if _dist_gain_from_state is None:
    # ローカルに変数があるならそれを使う
    _dist_gain_from_state = float(locals().get("dist_turn_gain", 0.0))
dist_turn_gain_val = float(_dist_gain_from_state)

# 4) 最終スコア
df_agg["FinalRaw"] = (
    df_agg["RecencyZ"]
    + float(stab_weight) * df_agg["StabZ"]
    + float(pace_gain)   * df_agg["PacePts"]
    + float(turn_gain)   * df_agg["TurnPrefPts"]
    + float(dist_turn_gain_val) * df_agg["DistTurnZ"]
)
df_agg["FinalZ"] = z_score(df_agg["FinalRaw"])

# ===== NEW: AR100（B中心化の線形キャリブレーション） =====
S = df_agg['FinalRaw'].to_numpy(dtype=float)
if np.all(~np.isfinite(S)) or len(S)==0:
    med = 0.0; qa = 1.0
else:
    med = float(np.nanmedian(S))
    p = max(0.01, min(0.99, 1.0 - float(band_A_share)/100.0))
    qa = float(np.nanquantile(S, p))
denom = (qa - med)
a = (70.0 - float(band_mid_target)) / (denom if abs(denom) > 1e-8 else 1.0)
df_agg['AR100'] = band_mid_target + a * (df_agg['FinalRaw'] - med)
df_agg['AR100'] = df_agg['AR100'].clip(band_clip_lo, band_clip_hi)

def to_band(x):
    try:
        v = float(x)
    except:
        return ''
    if v >= 90: return 'SS'
    if v >= 80: return 'S'
    if v >= 70: return 'A'
    if v >= 60: return 'B'
    if v >= 50: return 'C'
    return 'E'
df_agg['Band'] = df_agg['AR100'].map(to_band)

# ============================================================
# ========== ここから：勝率は PL（解析解）に置き換え ==========
# ============================================================

def _softmax_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    v = v - np.max(v)
    ev = np.exp(v)
    s = ev.sum()
    return ev / s if (s > 0 and np.isfinite(s)) else np.ones_like(ev)/len(ev)

# --- （任意）過去走からの等温回帰/Platt校正器の学習 ---
def _make_race_id_for_hist(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df['レース日'], errors='coerce').dt.strftime('%Y%m%d').fillna('00000000')
    n = df['競走名'].astype(str).fillna('')
    return d + '_' + n

def fit_isotonic_from_history(df_hist: pd.DataFrame, beta: float = 1.4):
    if not SK_ISO: 
        return None
    need = {'レース日','競走名','score_norm','確定着順'}
    if not need.issubset(df_hist.columns):
        return None
    df = df_hist.dropna(subset=['score_norm','確定着順']).copy()
    df['race_id'] = _make_race_id_for_hist(df)
    X_list, y_list = [], []
    for rid, g in df.groupby('race_id'):
        s = g['score_norm'].astype(float).to_numpy()
        p = _softmax_vec(beta * s)
        y = (pd.to_numeric(g['確定着順'], errors='coerce') == 1).astype(int).to_numpy()
        if len(p) == len(y) and len(p) >= 2:
            X_list.append(p); y_list.append(y)
    if not X_list:
        return None
    X = np.concatenate(X_list)
    Y = np.concatenate(y_list)
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(X, Y)
    return ir

def fit_platt_from_history(df_hist: pd.DataFrame, beta: float = 1.4):
    if not SK_PLATT:
        return None
    need = {'レース日','競走名','score_norm','確定着順'}
    if not need.issubset(df_hist.columns):
        return None
    df = df_hist.dropna(subset=['score_norm','確定着順']).copy()
    df['race_id'] = _make_race_id_for_hist(df)
    X_list, y_list = [], []
    for rid, g in df.groupby('race_id'):
        s = g['score_norm'].astype(float).to_numpy()
        p = _softmax_vec(beta * s)
        y = (pd.to_numeric(g['確定着順'], errors='coerce') == 1).astype(int).to_numpy()
        if len(p) == len(y) and len(p) >= 2:
            X_list.append(p.reshape(-1,1)); y_list.append(y)
    if not X_list:
        return None
    X = np.vstack(X_list)
    Y = np.concatenate(y_list)
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(X, Y)
    return lr

# --- 本レース：PL（softmax）で勝率、Top3は軽量MC ---
abilities = df_agg['FinalRaw'].to_numpy(float)
m = np.nanmean(abilities); s = np.nanstd(abilities) + 1e-9
abilities = (abilities - m) / s  # スケーリングのみ（順位は不変）

beta_pl = float(mc_beta)
x = beta_pl * (abilities - np.max(abilities))
ex = np.exp(x)
p_win_pl = ex / np.sum(ex)

# （任意）校正
calibrator = None
if do_calib:
    if calib_method.startswith("Isotonic") and SK_ISO:
        with st.spinner("校正（Isotonic）を学習中…"):
            calibrator = fit_isotonic_from_history(df_score.copy(), beta=beta_pl)
    elif calib_method.startswith("Platt") and SK_PLATT:
        with st.spinner("校正（Platt）を学習中…"):
            calibrator = fit_platt_from_history(df_score.copy(), beta=beta_pl)

if calibrator is not None:
    try:
        if isinstance(calibrator, IsotonicRegression):
            p_win_cal = calibrator.predict(p_win_pl)
        else:
            p_win_cal = calibrator.predict_proba(p_win_pl.reshape(-1,1))[:,1]
        p_win_pl = np.clip(p_win_cal, 1e-6, 1-1e-6)
    except Exception:
        pass

df_agg['勝率%_PL'] = (100 * p_win_pl).round(2)

# ---- Top3はPLの逐次サンプリングで近似（軽量MC）。勝率はPLそのものなのでMCはTop3用だけ。
draws_top3 = int(mc_iters)
rng_top3 = np.random.default_rng(int(mc_seed) + 24601)
U = beta_pl * abilities[None, :] + rng_top3.gumbel(size=(draws_top3, len(abilities)))
rank_idx = np.argsort(-U, axis=1)
top3_counts = np.zeros(len(abilities), float)
for k in range(3):
    top3_counts += np.bincount(rank_idx[:, k], minlength=len(abilities)).astype(float)
df_agg['複勝率%_PL'] = (100 * (top3_counts / draws_top3)).round(2)

# ------------------------------------------------------------
# （この先：散布図、コメント、ポジション図、タブUI、EV/ケリー、評価タブ等は【後半】で差し替え）
# ------------------------------------------------------------

# ------------------------------------------------------------
# ここから後半：UIタブ／可視化／評価／校正ダッシュボード／買い目（EV>1 + 分数ケリー）
# ------------------------------------------------------------
# --- 追加：最終結果（AR100 など）の分布グラフ ---
def render_result_distribution(df, col="AR100"):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        st.info(f"{col} の値がありません。最終結果を計算してください。")
        return

    mean_v = float(s.mean())
    bins = min(30, max(10, int(len(s) ** 0.5) * 2))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(s, bins=bins)
    ax.axvline(mean_v, linestyle="--")
    ax.set_xlabel(col)
    ax.set_ylabel("件数")
    ax.grid(alpha=.3)
    st.pyplot(fig, use_container_width=True)

    st.caption(f"平均 {col}: {mean_v:.2f}")
  

# ↑ ここが render_result_distribution の最後の行

# ▼▼▼ ここに追記：出走プレビュー用の小関数とレンダラ ▼▼▼
def _pill(mark: str) -> str:
    color = {'◎':'#4f46e5','〇':'#14b8a6','○':'#14b8a6','▲':'#3b82f6','△':'#94a3b8','×':'#ef4444'}.get(str(mark), '#94a3b8')
    return (
        f"<span style='display:inline-flex;align-items:center;gap:6px;padding:3px 10px;"
        f"border-radius:999px;border:1px solid rgba(255,255,255,.12);"
        f"background:rgba(255,255,255,.05);color:{color};font-weight:700;'>{mark}</span>"
    )

def _bar(v, max_v=100):
    try:
        pct = max(0.0, min(100.0, float(v)/float(max_v)*100.0))
    except Exception:
        pct = 0.0
    return (
        "<div style='height:10px;border-radius:999px;background:#0f1830;"
        "border:1px solid rgba(255,255,255,.08);overflow:hidden'>"
        f"<span style='display:block;height:100%;width:{pct:.1f}%;"
        "background:linear-gradient(90deg,#4f46e5,#14b8a6)'></span></div>"
    )

def render_final_preview(df):
    """df は df_disp を渡す想定（列：馬名・印・AR100・枠・番 など）"""
    import pandas as pd, numpy as np
    for _, r in df.iterrows():
        name = str(r.get('馬名',''))
        mark = str(r.get('印',''))
        ar   = r.get('AR100', np.nan)
        ar_s = f"{float(ar):.1f}" if pd.notna(ar) else "-"
        info = []
        if pd.notna(r.get('枠')): info.append(f"枠 {int(r['枠'])}")
        if pd.notna(r.get('番')): info.append(f"馬番 {int(r['番'])}")
        info = " / ".join(info)

        st.markdown(
            f"""
<div style='background:#121a2d;border:1px solid rgba(255,255,255,.06);border-radius:18px;
            padding:14px 16px;margin:8px 0;box-shadow:0 10px 25px rgba(0,0,0,.25)'>
  <div style='display:flex;justify-content:space-between;align-items:center;gap:12px;'>
    <div>
      <div style='display:flex;align-items:center;gap:10px;'>
        {_pill(mark)}
        <strong style='font-size:16px'>{name}</strong>
        <span style='color:#94a3b8;font-size:12px'>{info}</span>
      </div>
    </div>
    <div style='min-width:240px;text-align:right'>
      <div style='margin-bottom:6px'><span style='color:#94a3b8;font-size:12px'>AR100</span>
        <span style='font-variant-numeric:tabular-nums;font-weight:700;'>{ar_s}</span>
      </div>
      {_bar(ar, 100)}
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
# ▲▲▲ ここまで追記 ▲▲▲

st.subheader("本日の見立て")

# 表示用テーブル作成
# 表示用カラム
disp_cols = ['枠','番','馬名','脚質','PacePts','TurnPref','RightZ','LeftZ',
             'RecencyZ','StabZ','FinalRaw','FinalZ','AR100','Band','勝率%_PL','複勝率%_PL',
             # 追加
             'DistTurnZ','n_eff_turn','BestDist_turn'
]

# ...（中略）...

show_cols = ['順位','印','枠','番','馬名','脚質','AR100','Band',
             '勝率%_PL','複勝率%_PL','TurnPref','PacePts','RightZ','LeftZ','FinalZ',
             # 追加
             'DistTurnZ','n_eff_turn','BestDist_turn'
]

# === 表示用 df_disp 構築（show_cols 定義の直後に追加） ===

def _fmt_int(x):
    try:
        return "" if pd.isna(x) else f"{int(x)}"
    except Exception:
        return ""

# 下流で参照する列が欠けていても落ちないように作っておく
for c in ['PacePts','TurnPref','RightZ','LeftZ','RecencyZ','StabZ','FinalRaw','FinalZ',
          'AR100','Band','勝率%_PL','複勝率%_PL','DistTurnZ','n_eff_turn','BestDist_turn']:
    if c not in df_agg.columns:
        df_agg[c] = np.nan

# 今日の枠・馬番・（手入力の）脚質をマージ
df_disp = df_agg.merge(
    horses[['馬名','枠','番','脚質']],
    on='馬名', how='left', suffixes=('', '_today')
)

# もし df_agg 側にも脚質があるなら、手入力（today）を優先
if '脚質_today' in df_disp.columns:
    df_disp['脚質'] = df_disp['脚質_today'].where(
        df_disp['脚質_today'].astype(str).str.strip().ne(''),
        df_disp.get('脚質')
    )
    df_disp.drop(columns=['脚質_today'], inplace=True, errors='ignore')

# 枠・番は数値化（表示時は _fmt_int で整形）
for c in ['枠','番']:
    df_disp[c] = pd.to_numeric(df_disp.get(c), errors='coerce')

# 並べ替え（AR100 → 勝率）して順位を付与
df_disp = df_disp.sort_values(['AR100','勝率%_PL'], ascending=[False, False]).reset_index(drop=True)
df_disp['順位'] = np.arange(1, len(df_disp)+1)

# 印（◎○▲△）を付与：1位=◎、2位=○、3位=▲、4～6位=△
def _assign_mark(rank: int) -> str:
    if rank == 1: return '◎'
    if rank == 2: return '〇'
    if rank == 3: return '▲'
    if 4 <= rank <= 6: return '△'
    return ''
df_disp['印'] = df_disp['順位'].apply(_assign_mark)

# 下流の表示/可視化が想定する列順へ整形（存在しない列は上で補完済み）
df_disp = df_disp[['順位','印','枠','番','馬名','脚質','PacePts','TurnPref','RightZ','LeftZ',
                   'RecencyZ','StabZ','FinalRaw','FinalZ','AR100','Band','勝率%_PL','複勝率%_PL',
                   'DistTurnZ','n_eff_turn','BestDist_turn']].copy()

# 整形（フォーマット）にも追加
styled = (
    df_disp[show_cols]
      .style
      .apply(_style_waku, subset=['枠'])
      .format({
          '枠': _fmt_int, '番': _fmt_int,
          'AR100':'{:.1f}','勝率%_PL':'{:.2f}','複勝率%_PL':'{:.2f}',
          'FinalZ':'{:.2f}','RightZ':'{:.1f}','LeftZ':'{:.1f}','PacePts':'{:.2f}',
          # 追加
          'DistTurnZ':'{:.2f}','n_eff_turn':'{:.1f}','BestDist_turn': _fmt_int
      }, na_rep="")
)

st.dataframe(styled, use_container_width=True, height=H(df_disp, 560))


# ====================== タブ ======================
tab_main, tab_vis, tab_eval, tab_calib, tab_bet = st.tabs(["🏁 本命","📊 可視化","📈 評価","📏 校正","💸 買い目"])

with tab_main:
    st.markdown("#### 本命・対抗（Top候補）")
    st.markdown("---")
    st.markdown("#### 全体スコア分布（結果）")
    # AR100 の分布グラフ（→ 勝率で見たければ col="勝率%_PL"）
    render_result_distribution(df_disp, col="AR100")


    # 表示用：上位を抜粋
    top_cols = ['順位','印','枠','番','馬名','AR100','Band','勝率%_PL','複勝率%_PL','PacePts','TurnPref']
    top_view = df_disp[top_cols].head(6).copy()
    st.dataframe(
        top_view.style.format({'AR100':'{:.1f}','勝率%_PL':'{:.2f}','複勝率%_PL':'{:.2f}','PacePts':'{:.2f}'}),
        use_container_width=True, height=H(top_view, 360)
    )

    # 見送りルールのメモ
    if not (df_disp['AR100'] >= 70).any():
        st.warning("今回のレースは『見送り』：A以上（AR100≥70）が不在。")
    else:
        lead = top_view.iloc[0] if len(top_view)>0 else None
        if lead is not None:
            st.info(f"本命候補：**{int(lead['枠'])}-{int(lead['番'])} {lead['馬名']}** / 勝率{lead['勝率%_PL']:.2f}% / AR100 {lead['AR100']:.1f}")

    # 4コーナー配置図（脚質×枠からの想定）
    try:
        fig = render_corner_positions_nowrace(horses, combined_style, title="4コーナー想定ポジション（本レース）")
        st.pyplot(fig)
    except Exception as e:
        st.caption(f"4角ポジション図は表示できませんでした：{e}")

with tab_vis:
    st.markdown("#### 散布図（AR100 × ペース適性）")
    df_plot = df_disp[['馬名','AR100','PacePts','勝率%_PL','脚質','枠','番']].copy()
    df_plot = df_plot.dropna(subset=['AR100','PacePts'])
    if ALT_AVAILABLE and not df_plot.empty:
        ch = (
            alt.Chart(df_plot)
            .mark_circle()
            .encode(
                x=alt.X('PacePts:Q', title='PacePts（＋が追込/差し有利・−が逃げ/先行有利）'),
                y=alt.Y('AR100:Q', title='AR100（B中心化指数）'),
                size=alt.Size('勝率%_PL:Q', title='勝率%（PL）', scale=alt.Scale(range=[40, 600])),
                tooltip=['枠','番','馬名','脚質','AR100','勝率%_PL','PacePts']
            )
            .properties(height=480)
        )
        st.altair_chart(ch, use_container_width=True)
    elif not df_plot.empty:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        s = 40 + (df_plot['勝率%_PL'].fillna(0).to_numpy())*8
        ax.scatter(df_plot['PacePts'], df_plot['AR100'], s=s, alpha=0.6)
        for _, r in df_plot.iterrows():
            ax.annotate(str(r['番']), (r['PacePts'], r['AR100']), xytext=(3,3), textcoords="offset points", fontsize=8)
        ax.set_xlabel('PacePts')
        ax.set_ylabel('AR100')
        ax.grid(True, ls='--', alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("可視化できるデータがありません。")

with tab_eval:
    st.markdown("#### 過去走での妥当性評価（レース内softmax）")

    def history_probs_and_labels(df_hist: pd.DataFrame, beta: float = 1.4, calibrator_obj=None):
        need = {'レース日','競走名','score_norm','確定着順'}
        if not need.issubset(df_hist.columns): 
            return np.array([]), np.array([])
        dfh = df_hist.dropna(subset=['score_norm','確定着順']).copy()
        dfh['race_id'] = (pd.to_datetime(dfh['レース日'], errors='coerce').dt.strftime('%Y%m%d').fillna('00000000')
                          + '_' + dfh['競走名'].astype(str).fillna(''))
        probs = []; labels = []
        for rid, g in dfh.groupby('race_id'):
            s = g['score_norm'].astype(float).to_numpy()
            p = _softmax_vec(beta * s)
            y = (pd.to_numeric(g['確定着順'], errors='coerce') == 1).astype(int).to_numpy()
            if len(p)==len(y) and len(p)>=2:
                if calibrator_obj is not None:
                    try:
                        if isinstance(calibrator_obj, IsotonicRegression):
                            p = calibrator_obj.predict(p)
                        else:
                            p = calibrator_obj.predict_proba(p.reshape(-1,1))[:,1]
                        p = np.clip(p, 1e-6, 1-1e-6)
                    except Exception:
                        pass
                probs.append(p); labels.append(y)
        if not probs:
            return np.array([]), np.array([])
        return np.concatenate(probs), np.concatenate(labels)

    def brier_score(p, y):
        p = np.clip(p, 1e-9, 1-1e-9); y = np.asarray(y, float)
        return float(np.mean((p - y)**2))

    def log_loss_bin(p, y):
        p = np.clip(p, 1e-9, 1-1e-9); y = np.asarray(y, float)
        return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

    # Raw
    p_raw, y = history_probs_and_labels(df_score.copy(), beta=beta_pl, calibrator_obj=None)
    # Calibrated（If available）
    p_cal, _ = history_probs_and_labels(df_score.copy(), beta=beta_pl, calibrator_obj=calibrator) if (calibrator is not None) else (np.array([]), None)

    if p_raw.size == 0:
        st.info("評価に十分な過去走データがありません。")
    else:
        bs_raw = brier_score(p_raw, y)
        ll_raw = log_loss_bin(p_raw, y)
        st.write(f"**Brier**: {bs_raw:.4f}　/　**LogLoss**: {ll_raw:.4f}（未校正）")
        if p_cal.size > 0:
            bs_cal = brier_score(p_cal, y)
            ll_cal = log_loss_bin(p_cal, y)
            st.write(f"**Brier**: {bs_cal:.4f}　/　**LogLoss**: {ll_cal:.4f}（校正後）")

        # Reliability Diagram
        def reliability_points(p, y, n_bins=10):
            bins = np.linspace(0,1,n_bins+1)
            idx = np.digitize(p, bins)-1
            xs, ys, ns = [], [], []
            for b in range(n_bins):
                m = idx==b
                if m.sum() == 0: 
                    continue
                xs.append(p[m].mean())
                ys.append(y[m].mean())
                ns.append(int(m.sum()))
            return np.array(xs), np.array(ys), np.array(ns)

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot([0,1],[0,1], ls='--', alpha=0.5)
        x1,y1,n1 = reliability_points(p_raw, y, n_bins=12)
        ax.plot(x1, y1, marker='o', label='Raw')
        if p_cal.size > 0:
            x2,y2,n2 = reliability_points(p_cal, y, n_bins=12)
            ax.plot(x2, y2, marker='o', label='Calibrated')
        ax.set_xlabel('平均 予測勝率')
        ax.set_ylabel('実測 勝率（Top1）')
        ax.grid(True, ls=':', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        # NDCG@3（参考）
        try:
            df_tmp = df_score[['レース日','競走名','score_norm','確定着順']].dropna().copy()
            df_tmp['race_id'] = (pd.to_datetime(df_tmp['レース日'], errors='coerce').dt.strftime('%Y%m%d') + '_' + df_tmp['競走名'].astype(str))
            df_tmp['y'] = (pd.to_numeric(df_tmp['確定着順'], errors='coerce') == 1).astype(int)
            ndcg_raw = ndcg_by_race(df_tmp[['race_id','y']], p_raw, k=3)
            st.caption(f"NDCG@3（未校正softmax）：{ndcg_raw:.4f}")
        except Exception:
            pass

with tab_calib:
    st.markdown("#### 確率校正の状況")
    if not do_calib:
        st.info("現在は**未校正**です。サイドバーの「📏 確率校正」をONにすると、等温回帰（推奨）またはPlattで校正します。")
    else:
        if calibrator is None:
            st.warning("校正器の学習に必要なデータが不足しています。sheet0に過去走（レース内の着順・score_norm）が必要です。")
        else:
            method = "IsotonicRegression" if isinstance(calibrator, IsotonicRegression) else "Platt (Logistic)"
            st.success(f"校正器：**{method}** を使用中。")
            st.write("- 評価タブでBrier/LogLossと信頼度曲線（Reliability）を確認できます。")
            st.write("- 現在の勝率列：**勝率%_PL**（校正後があればそれを反映）")

with tab_bet:
    st.markdown("#### オッズ入力（小数＝払い戻し総額倍率）")
    st.caption("例：単勝3.5倍 → 3.5、複勝2.1倍 → 2.1。未入力はスキップ。")
    # オッズ編集テーブル
    if 'odds_df' not in st.session_state:
        st.session_state['odds_df'] = df_disp[['枠','番','馬名']].copy().assign(単オッズ=np.nan, 複オッズ=np.nan)
    # マージで新馬追加/順序同期
    odds_merge = df_disp[['枠','番','馬名']].merge(st.session_state['odds_df'], on=['枠','番','馬名'], how='left')
    odds_merge[['単オッズ','複オッズ']] = odds_merge[['単オッズ','複オッズ']].astype(float)
    odds_edit = st.data_editor(
        odds_merge,
        column_config={
            '単オッズ': st.column_config.NumberColumn('単オッズ', min_value=1.01, step=0.01),
            '複オッズ': st.column_config.NumberColumn('複オッズ', min_value=1.01, step=0.01),
        },
        hide_index=True,
        height=H(odds_merge, 460),
        use_container_width=True,
        key="odds_editor",
    )
    st.session_state['odds_df'] = odds_edit[['枠','番','馬名','単オッズ','複オッズ']].copy()

    st.markdown("---")
    kelly_frac = st.slider("分数ケリー係数", 0.05, 1.0, 0.5, 0.05, help="0.5=ハーフケリー。過積みを防ぐ係数。")
    alloc_mode = st.selectbox("配分対象", ["単勝のみ","複勝のみ","単勝＋複勝"], index=2)
    st.caption(f"総予算: **{total_budget} 円** / 最小単位: **{min_unit} 円** / 最大点数: **{max_lines}**")

    # 期待値 & ケリー計算
    picks = []
    p_win = (df_disp['勝率%_PL'].to_numpy()/100.0)
    p_plc = (df_disp['複勝率%_PL'].to_numpy()/100.0)
    names = df_disp['馬名'].tolist()
    waku  = df_disp['枠'].tolist()
    umaban= df_disp['番'].tolist()

    for i, row in odds_edit.reset_index(drop=True).iterrows():
        name = names[i]; w=waku[i]; n=umaban[i]
        # 単勝
        o_t = row.get('単オッズ', np.nan)
        if pd.notna(o_t) and alloc_mode in ("単勝のみ","単勝＋複勝"):
            p = float(p_win[i]); b = float(o_t) - 1.0; q = 1.0 - p
            ev_roi = p * float(o_t) - 1.0                      # ROI（1なら等価、>0で+EV）
            f = max(0.0, (b*p - q) / max(1e-9, b))             # Kelly fraction
            if ev_roi > 0 and f > 0:
                picks.append({'種別':'単勝','枠':w,'番':n,'馬名':name,'p':p,'odds':float(o_t),'EV_ROI':ev_roi,'kelly_f':f})
        # 複勝（簡易：Top3確率×複勝オッズ）
        o_p = row.get('複オッズ', np.nan)
        if pd.notna(o_p) and alloc_mode in ("複勝のみ","単勝＋複勝"):
            p = float(p_plc[i]); b = float(o_p) - 1.0; q = 1.0 - p
            ev_roi = p * float(o_p) - 1.0
            f = max(0.0, (b*p - q) / max(1e-9, b))
            if ev_roi > 0 and f > 0:
                picks.append({'種別':'複勝','枠':w,'番':n,'馬名':name,'p':p,'odds':float(o_p),'EV_ROI':ev_roi,'kelly_f':f})

    if not picks:
        st.warning("現状、EV>0（= p×オッズ > 1）となる候補がありません。オッズを入力/更新して再評価してください。")
    else:
        df_picks = pd.DataFrame(picks)
        # 上限点数でカット（期待値の高い順）
        df_picks.sort_values(['EV_ROI','p','odds'], ascending=[False, False, True], inplace=True)
        df_picks = df_picks.head(int(max_lines)).copy()

        # 分数ケリーに基づく重み → 予算でスケール
        df_picks['w'] = df_picks['kelly_f'] * float(kelly_frac)
        sw = df_picks['w'].sum()
        if sw <= 0:
            st.warning("ケリー基準の重みがゼロです。ケリー係数やオッズ・確率を見直してください。")
        else:
            budget = float(total_budget)
            raw_alloc = budget * (df_picks['w'] / sw)
            # 最小単位で丸め
            stake = (np.floor(raw_alloc / float(min_unit)) * float(min_unit)).astype(int)
            # 0行を削除しすぎないよう、最低1単位を高EVから順に配布
            deficit = int(budget - stake.sum())
            if deficit >= min_unit:
                # 余りを高EV順に割当
                order = df_picks['EV_ROI'].rank(method='first', ascending=False).astype(int)
                idx_order = np.argsort(order.values)
                for j in idx_order:
                    if deficit < min_unit: break
                    stake.iloc[j] += int(min_unit)
                    deficit -= int(min_unit)
            df_picks['推奨金額(円)'] = stake
            df_picks['想定期待収益(円)'] = (df_picks['推奨金額(円)'] * (df_picks['p'] * df_picks['odds'] - 1.0)).round(1)

            # 出力
            show_bet_cols = ['種別','枠','番','馬名','odds','p','EV_ROI','kelly_f','推奨金額(円)','想定期待収益(円)']
            st.dataframe(
                df_picks[show_bet_cols].rename(columns={'odds':'オッズ','p':'勝率(またはTop3確率)','EV_ROI':'期待ROI','kelly_f':'ケリー率'})
                    .style.format({'オッズ':'{:.2f}','勝率(またはTop3確率)':'{:.3f}','期待ROI':'{:.3f}','ケリー率':'{:.3f}'}),
                use_container_width=True, height=H(df_picks, 480)
            )
            st.write(f"**合計投入額:** {int(df_picks['推奨金額(円)'].sum())} 円 / 予算 {int(budget)} 円")

            bet_csv = df_picks[show_bet_cols].to_csv(index=False).encode('utf-8-sig')
            st.download_button("🧾 買い目CSVを保存", data=bet_csv, file_name="bets_kelly.csv", mime="text/csv")

# ============== 仕上げメモ ==============
st.markdown("""
<small>
- 勝率はレース内 **Plackett–Luce（softmax）** の解析解です。Top3（複勝率）は**軽量Gumbelサンプリング**で近似しています。<br>
- 「📏 確率校正」をONにすると、sheet0（過去走）から**等温回帰/Platt**で校正カーブを学習し、現行レース勝率に反映します。<br>
- 「💸 買い目」は**EV>0**かつ分数ケリーで予算配分。オッズは小数（払戻総額倍率）で入力してください。
</small>
""", unsafe_allow_html=True)
