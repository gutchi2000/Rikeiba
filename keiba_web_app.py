# keiba_web_app_final.py
# サイドバー互換（expander）、縦軸堅牢化、年齢/枠重み・MC・血統HTML 完備
# + 右/左回り 自動判定&加点 + AR100/Bandキャリブレーション（B中心化）+ 見送りロジック
# + ★ 先頭の空行を確実に除去 / 4角ポジション図の安全化
# + ★ 脚質エディタの値をセッションに保存・復元（リランしても消えない）
# + ★ 脚質の表記ゆれを吸収（「追い込み」「差込」など→正規化）

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

# ---- 基本設定とフォント ----
@st.cache_resource
def get_jp_font():
    for path in ["ipaexg.ttf", "C:/Windows/Fonts/meiryo.ttc"]:
        try:
            return font_manager.FontProperties(fname=path)
        except Exception:
            pass
    return None

jp_font = get_jp_font()
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']
st.set_page_config(page_title="競馬予想アプリ（修正版）", layout="wide")

# ---- 便利CSS（sidebar 幅だけ調整）----
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
section[data-testid="stSidebar"] {width: 340px !important;}
div.block-container {padding-top: .6rem; padding-bottom: .8rem; max-width: 1400px;}
</style>
""", unsafe_allow_html=True)

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

# === NEW: バンド校正（B中心化） ===
with st.sidebar.expander("🏷 バンド校正（B中心化）", expanded=True):
    band_mid_target = st.slider("中央値→何点に合わせる？", 40, 80, 65, 1,
                                help="AR100でレースの真ん中を何点に置くか（Bの真ん中=65推奨）")
    band_A_share = st.slider("A以上の目標割合(%)", 1, 60, 25, 1,
                             help="AR100が70以上（A, S, SS）になる頭数の割合ターゲット。Bを厚くしたいなら小さめに。")
    band_clip_lo = st.slider("下限クリップ", 0, 60, 40, 1)
    band_clip_hi = st.slider("上限クリップ", 80, 100, 100, 1)

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

with st.sidebar.expander("🧪 モンテカルロ / 保存", expanded=False):
    mc_iters   = st.slider("勝率MC 反復回数", 1000, 100000, 20000, 1000)
    mc_beta    = st.slider("強さ→勝率 温度β", 0.1, 5.0, 1.5, 0.1)
    mc_tau     = st.slider("安定度ノイズ係数 τ", 0.0, 2.0, 0.6, 0.05)
    mc_seed    = st.number_input("乱数Seed", 0, 999999, 42, 1)
    st.markdown("---")
    total_budget = st.slider("合計予算", 500, 50000, 10000, 100)
    min_unit     = st.selectbox("最小賭け単位", [100, 200, 300, 500], index=0)
    max_lines    = st.slider("最大点数(連系)", 1, 60, 20, 1)
    scenario     = st.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])
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
edited = st.data_editor(
    attrs[['枠','番','馬名','性別','年齢','脚質','斤量','馬体重']].copy(),
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=STYLES),
        '斤量': st.column_config.NumberColumn('斤量', min_value=45, max_value=65, step=0.5),
        '馬体重': st.column_config.NumberColumn('馬体重', min_value=300, max_value=600, step=1)
    },
    use_container_width=True,
    num_rows='static',
    height=H(attrs, 420),
    hide_index=True,  # ★ 行番号も非表示に（任意）
    key="horses_editor",  # ★ セッション保持のため key を付与
)
horses = edited.copy()

# ★ 入力直後に脚質の表記ゆれを正規化しておく
if '脚質' in horses.columns:
    horses['脚質'] = horses['脚質'].map(normalize_style)

# ★ セッションへ保存（タブ移動やスライダー操作で消えない）
st.session_state['horses_df'] = horses.copy()

validate_inputs(df_score, horses)

# --- 脚質 自動推定 ---
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

# ★ 念のためここでも正規化
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

# ===== 最終スコア（正規化は内部指標） =====
df_agg['RecencyZ'] = z_score(df_agg['WAvgZ'])
df_agg['StabZ']    = z_score(-df_agg['WStd'])
df_agg['FinalRaw'] = (
    df_agg['RecencyZ']
    + stab_weight * df_agg['StabZ']
    + pace_gain * df_agg['PacePts']
    + turn_gain * df_agg['TurnPrefPts']
)
df_agg['FinalZ']   = z_score(df_agg['FinalRaw'])

# ===== NEW: AR100（B中心化の線形キャリブレーション） =====
S = df_agg['FinalRaw'].to_numpy(dtype=float)
if np.all(~np.isfinite(S)) or len(S)==0:
    med = 0.0; qa = 1.0
else:
    med = float(np.nanmedian(S))
    # A以上割合p: 上位(1-p)分位をA境界70点に合わせる
    p = max(0.01, min(0.99, 1.0 - float(band_A_share)/100.0))
    qa = float(np.nanquantile(S, p))  # ここより上がA以上にしたい
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

# ===== 勝率MC =====
S2 = df_agg['FinalRaw'].to_numpy(dtype=float)
S2 = (S2 - np.nanmean(S2)) / (np.nanstd(S2) + 1e-9)
W = df_agg['WStd'].to_numpy(dtype=float)
W = (W - W.min()) / (W.max() - W.min() + 1e-9)
n = len(S2)
rng = np.random.default_rng(int(mc_seed))
gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(mc_iters, n))
noise  = (mc_tau * W)[None, :] * rng.standard_normal((mc_iters, n))
U = mc_beta * S2[None, :] + noise + gumbel
rank_idx = np.argsort(-U, axis=1)
win_counts  = np.bincount(rank_idx[:, 0], minlength=n).astype(float)
top3_counts = np.zeros(n, dtype=float)
for k in range(3):
    top3_counts += np.bincount(rank_idx[:, k], minlength=n).astype(float)
p_win  = win_counts  / mc_iters
p_top3 = top3_counts / mc_iters
df_agg['勝率%_MC']   = (p_win  * 100).round(2)
df_agg['複勝率%_MC'] = (p_top3 * 100).round(2)

# 上位抽出と印（内部用）
CUTOFF = 50.0
cand = df_agg[df_agg['FinalZ'] >= CUTOFF].sort_values('FinalZ', ascending=False).copy()
topN = cand.head(6).copy()
marks = ['◎','〇','▲','☆','△','△']
if len(topN)>0: topN['印'] = marks[:len(topN)]

# ===== 散布図（縦軸堅牢化） =====
st.markdown("### 散布図（最終偏差値 × 安定度）")
if ALT_AVAILABLE and len(df_agg)>0:
    w_all = df_agg['WStd'].replace([np.inf,-np.inf], np.nan).dropna()
    if len(w_all) >= 2:
        q10, q25, q75, q90 = np.percentile(w_all, [10,25,75,90])
        iqr = max(0.1, q75 - q25)
        y_lo = max(0.0, q10 - 0.3*iqr)
        y_hi = q90 + 0.3*iqr
        if (y_hi - y_lo) < 4.0:
            mid = (y_hi + y_lo)/2
            y_lo, y_hi = max(0.0, mid-2.5), mid+2.5
    else:
        y_lo, y_hi = (0.0, 5.0)

    x_mid = 50.0
    y_mid = float(df_agg['WStd'].mean())
    rect = alt.Chart(pd.DataFrame([
        {'x1': df_agg['FinalZ'].min(), 'x2': x_mid, 'y1': y_mid, 'y2': y_hi},
        {'x1': x_mid, 'x2': df_agg['FinalZ'].max(), 'y1': y_mid, 'y2': y_hi},
        {'x1': df_agg['FinalZ'].min(), 'x2': x_mid, 'y1': y_lo, 'y2': y_mid},
        {'x1': x_mid, 'x2': df_agg['FinalZ'].max(), 'y1': y_lo, 'y2': y_mid},
    ])).mark_rect(opacity=0.07).encode(x='x1:Q', x2='x2:Q', y='y1:Q', y2='y2:Q')
    points = alt.Chart(df_agg).mark_circle(size=100).encode(
        x=alt.X('FinalZ:Q', title='最終偏差値'),
        y=alt.Y('WStd:Q',  title='加重標準偏差（小さいほど安定）', scale=alt.Scale(domain=(float(y_lo), float(y_hi)))),
        tooltip=['馬名','AR100','Band','WAvgZ','WStd','RecencyZ','StabZ','PacePts','TurnPref','FinalZ','勝率%_MC']
    )
    labels = alt.Chart(df_agg).mark_text(dx=6, dy=-6, fontSize=10, color='#ffffff').encode(
        x='FinalZ:Q', y='WStd:Q', text='馬名:N'
    )
    vline = alt.Chart(pd.DataFrame({'x':[x_mid]})).mark_rule(color='gray').encode(x='x:Q')
    hline = alt.Chart(pd.DataFrame({'y':[y_mid]})).mark_rule(color='gray').encode(y='y:Q')
    chart = (rect + points + labels + vline + hline).properties(width=700, height=420).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.table(df_agg[['馬名','FinalZ','WStd']].sort_values('FinalZ', ascending=False).head(20))

# ===== horses2（短評） =====
印map = dict(zip(topN.get('馬名', pd.Series(dtype=str)), topN.get('印', pd.Series(dtype=str))))
merge_cols = [c for c in ['馬名','WAvgZ','WStd','FinalZ','脚質','PacePts'] if c in df_agg.columns]
horses2 = horses.merge(df_agg[merge_cols], on='馬名', how='left') if merge_cols else horses.copy()
for col, default in [('印',''), ('脚質',''), ('短評',''), ('WAvgZ', np.nan), ('WStd', np.nan), ('FinalZ', np.nan), ('PacePts', np.nan)]:
    if col not in horses2.columns: horses2[col] = default
horses2['印'] = horses2['馬名'].map(印map).fillna('')

def ai_comment(row):
    base = ""
    if row['印'] == '◎':
        base += "本命評価。" + ("高い安定感で信頼度抜群。" if pd.notna(row['WStd']) and row['WStd'] <= 8 else "能力上位もムラあり。")
    elif row['印'] == '〇':
        base += "対抗評価。" + ("近走安定しており軸候補。" if pd.notna(row['WStd']) and row['WStd'] <= 10 else "展開ひとつで逆転も。")
    elif row['印'] in ['▲','☆']:
        base += "上位グループの一角。" + ("ムラがあり一発タイプ。" if pd.notna(row['WStd']) and row['WStd'] > 15 else "安定型で堅実。")
    elif row['印'] == '△':
        base += "押さえ候補。" + ("堅実だが勝ち切るまでは？" if pd.notna(row['WStd']) and row['WStd'] < 12 else "展開次第で浮上も。")
    style = str(row.get('脚質','')).strip()
    base += {
        "逃げ":"ハナを奪えれば粘り込み十分。",
        "先行":"先行力を活かして上位争い。",
        "差し":"展開が向けば末脚強烈。",
        "追込":"直線勝負の一撃に期待。"
    }.get(style, "")
    return base
try:
    horses2['短評'] = horses2.apply(ai_comment, axis=1)
except Exception:
    if '短評' not in horses2: horses2['短評'] = ""

# ===== 4角ポジション配置図の関数（安全化） =====
def _corner__wrap_name(name: str, width: int = 4) -> str:
    s = str(name).strip().replace("\u3000", " ")
    s = s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    return "\n".join([s[i:i+width] for i in range(0, len(s), width)]) if s else s
def _corner__zone_from_style(sty: str) -> str:
    sty = str(sty)
    if '逃' in sty or '先' in sty: return '前'
    if '差' in sty:               return '中'
    if '追' in sty:               return '後'
    return '中'
def _corner__to_int(x):
    try:
        return int(str(x).translate(str.maketrans('０１２３４５６７８９','0123456789')))
    except Exception:
        return None
def _corner__path_from_draw(draw: int | None, field_size: int) -> str:
    if not field_size or field_size <= 0: field_size = 16
    if draw is None:
        return "中"
    lo = max(1, int(np.ceil(field_size * 0.25)))
    hi = int(np.floor(field_size * 0.75))
    if draw <= lo:   return "内"
    if draw >  hi:   return "外"
    return "中"
def _corner__cell_offsets(n: int) -> list[tuple[float, float]]:
    if n <= 0: return []
    out = [(0.0, 0.0)]
    if n == 1: return out
    step_r = 0.018; step_t = 0.10
    i = 1
    while len(out) < n:
        ring = (i-1)//6 + 1
        pos  = (i-1)%6
        dr = ((-1)**ring) * step_r * ring
        dtheta = (pos - 2.5) * step_t
        out.append((dr, dtheta))
        i += 1
    return out[:n]
def _corner__draw_track(ax):
    ax.add_patch(Rectangle((0.05, 0.22), 0.55, 0.48, facecolor="#97b42b", edgecolor="none", alpha=0.92))
    cx, cy = 0.60, 0.46
    radii = [0.40, 0.34, 0.28, 0.22]
    lane_colors = ["#6f8f0c", "#7aa012", "#86b71a"]
    for i in range(3):
        ax.add_patch(Wedge((cx, cy), r=radii[i], theta1=-90, theta2=0,
                           width=radii[i]-radii[i+1], facecolor=lane_colors[i], edgecolor="none", alpha=0.98))
    return (cx, cy, radii)
def render_corner_positions_nowrace(horses_df: pd.DataFrame,
                                    combined_style_series: pd.Series,
                                    title: str = "4コーナー想定ポジション（今レース）",
                                    circle_radius_ax: float | None = None,
                                    name_wrap: int = 4) -> plt.Figure:
    df = horses_df.copy()
    if '馬名' not in df.columns:
        raise ValueError("horses_df に『馬名』列が必要です")
    sty = combined_style_series.reindex(df['馬名']).fillna('')
    df['zone4'] = sty.map(_corner__zone_from_style).fillna('中')
    FS = max(1, len(df))
    # ★ draw（馬番/枠）を安全に算出（pd.NA/Noneでも落ちない）
    if '番' in df.columns:
        draw = df['番'].map(_corner__to_int)
    elif '枠' in df.columns:
        _wk = df['枠'].map(_corner__to_int)
        group = int(np.ceil(FS/8))
        draw = _wk.apply(lambda w: ((w-1)*group + 1) if pd.notna(w) else None)
    else:
        draw = pd.Series([i+1 for i in range(FS)], index=df.index, dtype=object)
    df['path4'] = draw.apply(lambda x: _corner__path_from_draw(int(x) if (x is not None and pd.notna(x)) else None, FS))

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=140)
    ax.set_axis_off()
    cx, cy, radii = _corner__draw_track(ax)

    ax.text(0.5, 0.93, title, ha="center", va="center", fontsize=12, weight="bold", fontproperties=jp_font)
    ax.text(0.5, 0.87, "フラット", ha="center", va="center", fontsize=10, color="#c33", fontproperties=jp_font)
    ax.text(0.06, 0.17, "セル＝進路（内/中/外）×位置（前/中/後）", fontsize=9, fontproperties=jp_font)

    lane_to_r = {"内": radii[2] + 0.025, "中": radii[1] - 0.025, "外": radii[0] - 0.040}
    zone_to_theta = {"後": -60, "中": -40, "前": -20}
    if circle_radius_ax is None:
        circle_radius_ax = 0.035 if FS <= 14 else (0.032 if FS <= 16 else 0.029)

    for (lane, zone), g in df.groupby(['path4','zone4'], dropna=False):
        if lane not in lane_to_r or zone not in zone_to_theta:
            continue
        base_r = lane_to_r[lane]
        base_theta = np.deg2rad(zone_to_theta[zone])
        offsets = _corner__cell_offsets(len(g))
        for (dr, dth), (_, row) in zip(offsets, g.iterrows()):
            r = base_r + dr
            th = base_theta + dth
            x = cx + r * np.cos(th)
            y = cy + r * np.sin(th)
            circle = Circle((x, y), radius=circle_radius_ax, transform=ax.transAxes,
                            facecolor="white", edgecolor="#222", linewidth=1.2, zorder=6)
            ax.add_patch(circle)
            nm = _corner__wrap_name(row['馬名'], name_wrap)
            ax.text(x, y, nm, ha="center", va="center", fontsize=8.5, zorder=7, fontproperties=jp_font)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig

# ======================== 結果タブ ========================
tab_dash, tab_prob, tab_pace, tab_bets, tab_all, tab_pedi = st.tabs(
    ["🏠 ダッシュボード", "📈 勝率", "🧭 展開", "🎫 買い目", "📝 全頭コメント", "🧬 血統HTML"]
)

with tab_dash:
    st.subheader("サマリー")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("想定ペース", locals().get("pace_type","—"))
    c2.metric("出走頭数", len(horses))
    c3.metric("本レース回り", f"{TARGET_TURN}")
    if len(topN) > 0:
        c4.metric("◎ FinalZ", f"{topN.iloc[0]['FinalZ']:.1f}")
        try:
            win_pct = float(df_agg.loc[df_agg['馬名']==topN.iloc[0]['馬名'],'勝率%_MC'].iloc[0])
            c5.metric("◎ 推定勝率", f"{win_pct:.1f}%")
        except Exception:
            c5.metric("◎ 推定勝率", "—")

    # Band分布サマリー
    st.markdown("#### Band分布")
    band_counts = df_agg['Band'].value_counts().reindex(['SS','S','A','B','C','E']).fillna(0).astype(int)
    dd = pd.DataFrame({'Band':band_counts.index, '頭数': band_counts.values})
    st.dataframe(dd, use_container_width=True, height=H(dd, 200))

    st.markdown("#### 上位馬（FinalZ≧50・最大6頭）")
    _top = topN.merge(df_agg[['馬名','勝率%_MC','TurnPref','AR100','Band']], on='馬名', how='left')
    show_cols = [c for c in ['馬名','印','AR100','Band','FinalZ','WAvgZ','WStd','PacePts','TurnPref','勝率%_MC'] if c in _top.columns]
    st.dataframe(_top[show_cols], use_container_width=True, height=H(_top, 260))

    st.markdown("#### 回り適性（今回の回りを基準）")
    if {'RightZ','LeftZ','TurnPref','TurnGap'}.issubset(df_agg.columns):
        tv = df_agg[['馬名','RightZ','LeftZ','TurnGap','TurnPref']].copy()
        tv = tv.sort_values('TurnGap', ascending=(TARGET_TURN=='左')).reset_index(drop=True)
        st.dataframe(tv, use_container_width=True, height=H(tv, 260))
    else:
        st.info("回り適性を算出できるデータが不足しています。")

with tab_prob:
    st.subheader("推定勝率・複勝率（モンテカルロ）")
    prob_view = (
        df_agg[['馬名','AR100','Band','FinalZ','WAvgZ','WStd','PacePts','TurnPref','勝率%_MC','複勝率%_MC']]
        .sort_values(['Band','AR100','勝率%_MC'], ascending=[True,False,False]).reset_index(drop=True)
    )
    _pv = prob_view.copy()
    for c in ['勝率%_MC','複勝率%_MC']:
        if c in _pv: _pv[c] = _pv[c].map(lambda x: f"{x:.2f}%")
    st.dataframe(_pv, use_container_width=True, height=H(prob_view, 420))

with tab_pace:
    st.subheader("展開・脚質サマリー")
    st.caption(f"想定ペース: {locals().get('pace_type','—')}（{'固定' if pace_mode=='固定（手動）' else '自動MC'}）")

    df_map = horses.copy()
    if '脚質' not in df_map.columns:
        df_map['脚質'] = ''
    # 手入力があれば優先、空欄は combined_style で補完
    auto_st = df_map['馬名'].map(combined_style)
    cond_filled = df_map['脚質'].astype(str).str.strip().ne('')
    df_map.loc[~cond_filled, '脚質'] = auto_st.loc[~cond_filled]
    # 正規化して未定義は空へ
    df_map['脚質'] = df_map['脚質'].map(normalize_style)
    df_map['脚質'] = df_map['脚質'].fillna('').where(df_map['脚質'].isin(STYLES), other='')

    style_counts = df_map['脚質'].value_counts().reindex(STYLES).fillna(0).astype(int)
    total_heads = int(style_counts.sum()) if style_counts.sum() > 0 else 1
    style_pct = (style_counts / total_heads * 100).round(1)

    pace_summary = pd.DataFrame([{
        '想定ペース': locals().get('pace_type','—'),
        '逃げ':  f"{style_counts['逃げ']}頭（{style_pct['逃げ']}%）",
        '先行':  f"{style_counts['先行']}頭（{style_pct['先行']}%）",
        '差し':  f"{style_counts['差し']}頭（{style_pct['差し']}%）",
        '追込':  f"{style_counts['追込']}頭（{style_pct['追込']}%）",
    }])
    st.table(pace_summary)

    st.markdown("#### 4角ポジション配置図（今レース・想定）")
    try:
        fig_corner = render_corner_positions_nowrace(
            horses_df=horses,
            combined_style_series=combined_style,
            title=f"4コーナー想定ポジション（{locals().get('pace_type','—')}／{TARGET_TURN}回り）"
        )
        st.pyplot(fig_corner, use_container_width=True)
    except Exception as e:
        st.info(f"4角ポジション配置図はスキップ: {e}")

    # 馬番×脚質のロケーション図（既存）
    def _normalize_ban(x):
        return pd.to_numeric(str(x).translate(str.maketrans('０１２３４５６７８９','0123456789')), errors='coerce')
    if '番' in df_map.columns:
        df_map['_ban'] = _normalize_ban(df_map['番'])
        loc_df = df_map.dropna(subset=['_ban']).copy()
        loc_df = loc_df[loc_df['脚質'].isin(STYLES)]
        if not loc_df.empty:
            loc_df['_ban'] = loc_df['_ban'].astype(int)
            loc_df = loc_df.sort_values('_ban')

            fig, ax = plt.subplots(figsize=(10, 3))
            colors = {'逃げ':'red', '先行':'orange', '差し':'green', '追込':'blue'}
            for _, row in loc_df.iterrows():
                x = int(row['_ban']); y = STYLES.index(row['脚質'])
                ax.scatter(x, y, color=colors[row['脚質']], s=200)
                ax.text(
                    x, y, str(row['馬名']),
                    ha='center', va='center', color='white', fontsize=9, weight='bold',
                    bbox=dict(facecolor=colors[row['脚質']], alpha=0.7, boxstyle='round'),
                    fontproperties=jp_font
                )
            ax.set_yticks([0,1,2,3]); ax.set_yticklabels(STYLES, fontproperties=jp_font)
            xs = sorted(loc_df['_ban'].unique())
            ax.set_xticks(xs); ax.set_xticklabels([f"{i}番" for i in xs], fontproperties=jp_font)
            ax.set_xlabel("馬番", fontproperties=jp_font); ax.set_ylabel("脚質", fontproperties=jp_font)
            ax.set_title(f"展開ロケーション（{locals().get('pace_type','—')}想定）", fontproperties=jp_font)
            st.pyplot(fig)
        else:
            st.info("馬番または脚質が未入力のため、配置図は省略しました。")
    else:
        st.info("出走表に『番』列が見つからないため、配置図は省略しました。")

    st.markdown("#### 回り適性サマリー（時間加重の過去走スコアで推定）")
    if {'RightZ','LeftZ','TurnPref','TurnGap'}.issubset(df_agg.columns):
        tv = df_agg[['馬名','RightZ','LeftZ','TurnGap','TurnPref']].copy()
        tv = tv.sort_values('TurnGap', ascending=(TARGET_TURN=='左')).reset_index(drop=True)
        st.dataframe(tv, use_container_width=True, height=H(tv, 260))
    else:
        st.info("回り適性を算出できるデータが不足しています。")

with tab_bets:
    # === 見送りロジック（あなたの方針） ===
    allow_bet = bool((df_agg['AR100'] >= 70).any())
    if not allow_bet:
        st.subheader("今回のレースは『見送り』")
        st.info("方針：**A以上が1頭でもいたら買う／B以下しかいないなら見送り**。本レースはA以上が0頭のため見送り判定です。")
    else:
        h1 = topN.iloc[0]['馬名'] if len(topN) >= 1 else None
        h2 = topN.iloc[1]['馬名'] if len(topN) >= 2 else None
        symbols = topN.get('印', pd.Series([], dtype=str)).tolist()
        names   = topN['馬名'].tolist()
        others_names   = names[1:] if len(names) > 1 else []
        others_symbols = symbols[1:] if len(symbols) > 1 else []

        three = ['馬連','ワイド','馬単']
        def round_to_unit(x, unit): return int(np.floor(x / unit) * unit)

        main_share = 0.5
        pur1 = round_to_unit(total_budget * main_share * (1/4), int(min_unit))  # 単勝
        pur2 = round_to_unit(total_budget * main_share * (3/4), int(min_unit))  # 複勝
        rem  = total_budget - (pur1 + pur2)

        win_each   = round_to_unit(pur1 / 2, int(min_unit))
        place_each = round_to_unit(pur2 / 2, int(min_unit))

        st.subheader("■ 資金配分 (厳密合計)")
        st.write(f"合計予算：{total_budget:,}円  単勝：{pur1:,}円  複勝：{pur2:,}円  残：{rem:,}円  [単位:{min_unit}円]")

        bets = []
        if h1 is not None:
            bets += [{'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
                     {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each}]
        if h2 is not None:
            bets += [{'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
                     {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each}]

        finalZ_map = df_agg.set_index('馬名')['FinalZ'].to_dict()
        pair_candidates, tri_candidates, tri1_candidates = [], [], []
        if h1 is not None and len(others_names) > 0:
            for nm, mk in zip(others_names, others_symbols):
                score = finalZ_map.get(nm, 0)
                pair_candidates.append(('ワイド', f'◎–{mk}', h1, nm, score))
                pair_candidates.append(('馬連', f'◎–{mk}', h1, nm, score))
                pair_candidates.append(('馬単', f'◎→{mk}', h1, nm, score))
            for a, b in combinations(others_names, 2):
                score = finalZ_map.get(a,0) + finalZ_map.get(b,0)
                tri_candidates.append(('三連複','◎-〇▲☆△△', h1, f"{a}／{b}", score))
            second_opts = others_names[:2]
            for s in second_opts:
                for t in others_names:
                    if t == s: continue
                    score = finalZ_map.get(s,0) + 0.7*finalZ_map.get(t,0)
                    tri1_candidates.append(('三連単フォーメーション','◎-〇▲-〇▲☆△△', h1, f"{s}／{t}", score))

        if scenario == '通常':
            with st.expander("馬連・ワイド・馬単 から１券種を選択", expanded=True):
                choice = st.radio("購入券種", options=three, index=1)
                st.write(f"▶ {choice} に残り {rem:,}円 を充当")
            cand = [c for c in pair_candidates if c[0]==choice]
            cand = sorted(cand, key=lambda x: x[-1], reverse=True)[:int(max_lines)]
            K = len(cand)
            if K > 0 and rem >= int(min_unit):
                base = round_to_unit(rem / K, int(min_unit))
                amounts = [base]*K
                leftover = rem - base*K
                i = 0
                while leftover >= int(min_unit) and i < K:
                    amounts[i] += int(min_unit); leftover -= int(min_unit); i += 1
                for (typ, mks, base_h, pair_h, _), amt in zip(cand, amounts):
                    bets.append({'券種':typ,'印':mks,'馬':base_h,'相手':pair_h,'金額':int(amt)})
            else:
                st.info("連系はスキップ（相手不足 or 予算不足）")

        elif scenario == 'ちょい余裕':
            st.write("▶ 残り予算を ワイド ＋ 三連複 で配分")
            cand_wide = sorted([c for c in pair_candidates if c[0]=='ワイド'], key=lambda x: x[-1], reverse=True)
            cand_tri  = sorted(tri_candidates, key=lambda x: x[-1], reverse=True)
            cut_w = min(len(cand_wide), int(max_lines)//2 if int(max_lines)>1 else 1)
            cut_t = min(len(cand_tri),  int(max_lines) - cut_w)
            cand_wide, cand_tri = cand_wide[:cut_w], cand_tri[:cut_t]
            K = len(cand_wide) + len(cand_tri)
            if K>0 and rem >= int(min_unit):
                base = round_to_unit(rem / K, int(min_unit))
                amounts = [base]*K
                leftover = rem - base*K
                i = 0
                while leftover >= int(min_unit) and i < K:
                    amounts[i] += int(min_unit); leftover -= int(min_unit); i += 1
                all_cand = cand_wide + cand_tri
                for (typ, mks, base_h, pair_h, _), amt in zip(all_cand, amounts):
                    bets.append({'券種':typ,'印':mks,'馬':base_h,'相手':pair_h,'金額':int(amt)})
            else:
                st.info("連系はスキップ（相手不足 or 予算不足）")

        elif scenario == '余裕':
            st.write("▶ 残り予算を ワイド ＋ 三連複 ＋ 三連単フォーメーション で配分")
            cand_wide = sorted([c for c in pair_candidates if c[0]=='ワイド'], key=lambda x: x[-1], reverse=True)
            cand_tri  = sorted(tri_candidates, key=lambda x: x[-1], reverse=True)
            cand_tri1 = sorted(tri1_candidates, key=lambda x: x[-1], reverse=True)
            r_w, r_t, r_t1 = 2, 2, 1
            denom = r_w + r_t + r_t1
            q_w = max(1, (int(max_lines) * r_w)//denom)
            q_t = max(1, (int(max_lines) * r_t)//denom)
            q_t1= max(1, int(max_lines) - q_w - q_t)
            cand_wide, cand_tri, cand_tri1 = cand_wide[:q_w], cand_tri[:q_t], cand_tri1[:q_t1]
            K = len(cand_wide) + len(cand_tri) + len(cand_tri1)
            if K>0 and rem >= int(min_unit):
                base = round_to_unit(rem / K, int(min_unit))
                amounts = [base]*K
                leftover = rem - base*K
                i = 0
                while leftover >= int(min_unit) and i < K:
                    amounts[i] += int(min_unit); leftover -= int(min_unit); i += 1
                all_cand = cand_wide + cand_tri + cand_tri1
                for (typ, mks, base_h, pair_h, _), amt in zip(all_cand, amounts):
                    bets.append({'券種':typ,'印':mks,'馬':base_h,'相手':pair_h,'金額':int(amt)})
            else:
                st.info("連系はスキップ（相手不足 or 予算不足）")

        _df = pd.DataFrame(bets)
        spent = int(_df['金額'].fillna(0).replace('',0).sum()) if len(_df)>0 else 0
        diff = total_budget - spent
        if diff != 0 and len(_df) > 0:
            for idx in _df.index:
                cur = int(_df.at[idx,'金額'])
                new = cur + diff
                if new >= 0 and new % int(min_unit) == 0:
                    _df.at[idx,'金額'] = new
                    break

        _df_disp = _df.copy()
        if '金額' in _df_disp.columns and len(_df_disp) > 0:
            def fmt_money(x):
                try:
                    xv = float(x)
                    if np.isnan(xv) or int(xv) <= 0: return ""
                    return f"{int(xv):,}円"
                except Exception:
                    return ""
            _df_disp['金額'] = _df_disp['金額'].map(fmt_money)
        st.subheader("■ 最終買い目一覧（全券種まとめ）")
        if len(_df_disp) == 0:
            st.info("現在、買い目はありません。")
        else:
            st.table(safe_take(_df_disp, ['券種','印','馬','相手','金額']))

with tab_all:
    st.subheader("全頭AI診断コメント")
    q = st.text_input("馬名フィルタ（部分一致）", "")
    _all = horses2.merge(df_agg[['馬名','TurnPref','AR100','Band']], on='馬名', how='left')
    _all = _all[[c for c in ['馬名','印','脚質','短評','AR100','Band','WAvgZ','WStd','TurnPref'] if c in _all.columns]]
    if q.strip():
        _all = _all[_all['馬名'].astype(str).str.contains(q.strip(), case=False, na=False)]
    if _all.empty:
        st.info("コメント表示対象がありません。")
    else:
        st.dataframe(_all, use_container_width=True, height=H(_all, 420))

# ======================== 血統HTML 抽出ユーティリティ（トップレベル定義） ========================
def _detect_charset_from_head(raw: bytes) -> str | None:
    if raw.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    head_txt = raw[:4096].decode("ascii", "ignore")
    m1 = re.search(r'charset\s*=\s*[\'"]?([\w\-]+)', head_txt, flags=re.I)
    return m1.group(1).lower() if m1 else None

def _decode_html_bytes(raw: bytes, preferred: str | None = None) -> str:
    declared = _detect_charset_from_head(raw)
    cands = [c for c in [preferred, declared,
                         "cp932", "shift_jis", "utf-8", "utf-8-sig",
                         "euc_jp", "iso2022_jp", "utf-16", "utf-16-le", "utf-16-be"] if c]
    seen = set()
    for enc in [c for c in cands if not (c in seen or seen.add(c))]:
        try:
            txt = raw.decode(enc)
            # 文字化けが酷いutf-8を弾く軽いガード
            if enc.startswith("utf-8") and txt.count("�") > 10:
                continue
            return txt
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")

def _to_pedigree_url(u: str) -> str:
    m_id = re.search(r"race_id=(\d{12})", u)
    if m_id and "pedigree" not in u:
        return f"https://race.netkeiba.com/race/shutuba/pedigree.html?race_id={m_id.group(1)}"
    return u

def _fetch_url_html(u: str) -> tuple[str, str | None]:
    try:
        import requests
        used = _to_pedigree_url(u)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Referer": "https://race.netkeiba.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        r = requests.get(used, headers=headers, timeout=15, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 500:
            return _decode_html_bytes(r.content), None
        return "", f"HTTP {r.status_code} / bytes={len(r.content)}"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["".join([str(c) for c in tup if "Unnamed" not in str(c)]).strip()
                      for tup in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df

def _promote_header_row_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    if any(re.search("馬名", c) for c in cols):
        return df
    for i in range(min(6, len(df))):
        row = df.iloc[i].astype(str)
        if row.str.contains("馬名").any():
            new_cols = row.str.replace(r"\s+", "", regex=True).tolist()
            df = df.iloc[i+1:].copy()
            df.columns = new_cols
            break
    return df

def _extract_pedi_tables_from_html(html_text: str) -> list[pd.DataFrame]:
    try:
        tables = pd.read_html(html_text, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html_text, flavor="bs4")
        except Exception:
            tables = []
    fixed = []
    for t in tables:
        t = _flatten_columns(t)
        t = _promote_header_row_if_needed(t)
        if any(re.search(r"(馬名|^馬$)", str(c)) for c in t.columns):
            fixed.append(t.reset_index(drop=True))
    return fixed

# ======================== 血統HTML（ビュー＋キーワード一致→ボーナス） ========================
with tab_pedi:
    st.subheader("血統HTMLビューア + ボーナス付与")
    st.caption("NetKeiba等の血統ページHTMLを表示/解析し、キーワード一致の馬へボーナスを付与します。URLでもOK。")

    m = st.radio("入力方法", ["テキスト貼り付け（HTML/URL）", "HTMLファイルをアップロード"], horizontal=True)

    src_url: str | None = None
    html_text = ""

    if m == "テキスト貼り付け（HTML/URL）":
        html_txt = st.text_area(
            "HTML（ソース全文 or URL）を貼り付け",
            height=220,
            placeholder="<html>...</html> または https://race.netkeiba.com/.../pedigree.html"
        )
        val = (html_txt or "").strip()
        if re.match(r"^https?://", val):
            src_url = val
            with st.spinner("URLから取得中…"):
                html_text, err = _fetch_url_html(val)
            if err:
                st.warning(
                    "URLからの取得に失敗しました。ブラウザで『ページを保存（.html）』して『HTMLファイルをアップロード』に切り替えてください。"
                    f"（詳細: {err}）"
                )
        else:
            html_text = val
    else:
        up = st.file_uploader("血統HTMLファイル", type=["html", "htm"], key="pedi_html_up")
        if up:
            raw  = up.read()
            html_text = _decode_html_bytes(raw)

    if html_text.strip() and COMPONENTS:
        components.html(html_text, height=700, scrolling=True)
    elif html_text.strip() and not COMPONENTS:
        st.code(html_text[:8000], language="html")

    dfs = _extract_pedi_tables_from_html(html_text) if html_text.strip() else []

    if (not dfs) and src_url:
        try:
            used = _to_pedigree_url(src_url)
            raw_tables = pd.read_html(used)
            for t in raw_tables:
                t = _flatten_columns(t)
                t = _promote_header_row_if_needed(t)
                if any(re.search(r"(馬名|^馬$)", str(c)) for c in t.columns):
                    dfs.append(t.reset_index(drop=True))
            if dfs:
                st.info("ページ埋め込みは不可でしたが、URLからテーブル抽出に成功しました。")
        except Exception as e:
            st.error(f"URLからのテーブル抽出にも失敗しました: {e}")

    st.markdown("### 🧬 血統ボーナス設定（下で一致した馬に加点）")
    default_pts = int(st.session_state.get('pedi:points', 3))
    points = st.slider("一致した馬へのボーナス点", 0, 20, default_pts)

    if dfs:
        idx = st.selectbox(
            "対象テーブルを選択",
            options=list(range(len(dfs))),
            format_func=lambda i: f"Table {i+1}（列: {', '.join(map(str, dfs[i].columns[:6]))} …）"
        )
        dfp = dfs[idx]

        name_candidates = [c for c in dfp.columns if re.search(r"(馬名|^馬$)", str(c))]
        name_col = st.selectbox(
            "馬名列",
            options=dfp.columns.tolist(),
            index=dfp.columns.tolist().index(name_candidates[0]) if name_candidates else 0
        )

        known = ["父タイプ名","父名","母父タイプ名","母父名","父系","母父系","父","母父"]
        candidate_cols = [c for c in known if c in dfp.columns] or [c for c in dfp.columns if c != name_col]

        st.caption("※ 下のキーワードが、選択した列のどれかに一致した『馬名』へ加点します。複数行OK。")
        keys_text = st.text_area("血統キーワード（1行1ワード）",
                                 value=st.session_state.get('pedi:keys', ""), height=120)
        match_cols = st.multiselect("照合対象の列", candidate_cols,
                                    default=[c for c in known if c in candidate_cols] or candidate_cols)
        method = st.radio("照合方法", ["部分一致", "完全一致"], index=0, horizontal=True)

        def _norm(s: str) -> str:
            s = str(s)
            s = s.translate(_fwid).replace('\u3000', ' ').strip()
            return re.sub(r'\s+', '', s)

        keys = [k for k in (keys_text.splitlines() if keys_text else []) if k.strip()]
        keys_norm = [_norm(k) for k in keys]

        matched_names: list[str] = []
        if keys and match_cols:
            for _, row in dfp.iterrows():
                nm = _trim_name(row.get(name_col, ""))
                row_texts_norm = [_norm(row.get(c, "")) for c in match_cols]
                if method == "完全一致":
                    hit = any(r == k for r in row_texts_norm for k in keys_norm)
                else:
                    hit = any(k in r for r in row_texts_norm for k in keys_norm)
                if hit and nm:
                    matched_names.append(nm)

        st.session_state['pedi:map'] = { _trim_name(n): True for n in matched_names }
        st.session_state['pedi:points'] = int(points)
        st.session_state['pedi:keys'] = keys_text

        colL, colR = st.columns([2,3])
        with colL:
            st.write("一致した馬（加点対象）")
            if matched_names:
                st.table(pd.DataFrame({'馬名': matched_names}))
            else:
                st.info("現在、一致はありません。キーワード/照合列/照合方法を調整してください。")
        with colR:
            st.info(f"設定：ボーナス {points} 点 / 照合列 {', '.join(map(str, match_cols)) if match_cols else '（未選択）'} / {method}")

    else:
        st.info("馬名列を含むテーブルが見つかりません。URLが取れない環境では、ページを『完全保存（.html）』してアップロードしてください。")
