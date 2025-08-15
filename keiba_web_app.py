# keiba_web_app_complete_safe.py
# コピーしてそのまま貼って動くことを目指した完成版（依存不足に耐性あり）
import streamlit as st
import pandas as pd
import numpy as np
import re
import io, json, time
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations

# 外部ライブラリは存在しない可能性があるため try/except で安全に扱う
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

# 機械学習周りは optional。無ければ学習ボタンを押したときに案内を出す。
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, log_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    # ダミー定義（参照時に使わない）
    def train_test_split(*a, **k): raise RuntimeError("sklearn not available")
    def roc_auc_score(*a, **k): raise RuntimeError("sklearn not available")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# フォント設定（環境依存だが例外は握りつぶして進める）
@st.cache_resource
def get_jp_font():
    try:
        return font_manager.FontProperties(fname="ipaexg.ttf")
    except Exception:
        try:
            return font_manager.FontProperties(fname="C:/Windows/Fonts/meiryo.ttc")
        except Exception:
            return None

jp_font = get_jp_font()
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']

st.set_page_config(page_title="競馬予想アプリ", layout="wide")

# ======================== ユーティリティ ========================
def z_score(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([50]*len(s), index=s.index)
    return 50 + 10 * (s - s.mean()) / std

def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

_fwid = str.maketrans('０１２３４５６７８９','0123456789')  # 全角数字→半角変換

def normalize_grade_text(x: str | None) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    s = str(x)
    s = s.translate(_fwid)
    s = s.replace('Ｇ','G').replace('（','(').replace('）',')')
    s = s.replace('Ⅰ','I').replace('Ⅱ','II').replace('Ⅲ','III')
    s = re.sub(r'G\s*III', 'G3', s, flags=re.I)
    s = re.sub(r'G\s*II',  'G2', s, flags=re.I)
    s = re.sub(r'G\s*I',   'G1', s, flags=re.I)
    m = re.search(r'G\s*([123])', s, flags=re.I)
    return f"G{m.group(1)}" if m else None

@st.cache_data(show_spinner=False)
def load_excel_bytes(content: bytes):
    xls = pd.ExcelFile(io.BytesIO(content))
    s0 = pd.read_excel(xls, sheet_name=0)
    s1 = pd.read_excel(xls, sheet_name=1)
    return s0, s1

@st.cache_data(show_spinner=False)
def parse_blood_html_bytes(content: bytes) -> pd.DataFrame:
    text = content.decode(errors='ignore')
    rows = re.findall(r'<tr[\s\S]*?<\/tr>', text, flags=re.I)
    buff = []
    for r in rows:
        cells = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r, flags=re.I)
        if len(cells) >= 2:
            name = re.sub(r'<.*?>', '', cells[0]).strip()
            blood = re.sub(r'<.*?>', '', cells[1]).strip()
            buff.append((name, blood))
    return pd.DataFrame(buff, columns=['馬名','血統'])

def validate_inputs(df_score: pd.DataFrame, horses: pd.DataFrame):
    problems = []
    req = ['馬名','レース日','競走名','頭数','確定着順']
    for c in req:
        if c not in df_score.columns:
            problems.append(f"sheet0 必須列が見つからない: {c}")
        else:
            null_rate = df_score[c].isna().mean()
            if null_rate > 0.2:
                problems.append(f"sheet0 列 {c} の欠損が多い（{null_rate:.0%}）")
    if '斤量' in horses:
        bad = horses['斤量'].dropna()
        if len(bad)>0 and ((bad<45)|(bad>65)).any():
            problems.append("sheet1 斤量がレンジ外（45–65）")
    if {'通過4角','頭数'}.issubset(df_score.columns):
        tmp = df_score[['通過4角','頭数']].dropna()
        if len(tmp)>0 and ((tmp['通過4角']<1) | (tmp['通過4角']>tmp['頭数'])).any():
            problems.append("sheet0 通過4角が頭数レンジ外")
    if problems:
        st.warning("⚠ 入力チェック：\n- " + "\n- ".join(problems))

# ======================== サイドバー（折り畳み＆説明つき整理版） ========================
st.sidebar.header("基本スコア & ボーナス")
lambda_part  = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
besttime_w   = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0)

with st.sidebar.expander("戦績率の重み（当該馬場）", expanded=False):
    st.caption("勝率・連対率・複勝率（％）を1走スコアの“加点分”として取り込む時の係数です。")
    win_w  = st.slider("勝率の重み",   0.0, 5.0, 1.0, 0.1, key="w_win")
    quin_w = st.slider("連対率の重み", 0.0, 5.0, 0.7, 0.1, key="w_quin")
    plc_w  = st.slider("複勝率の重み", 0.0, 5.0, 0.5, 0.1, key="w_plc")

with st.sidebar.expander("各種ボーナス設定", expanded=False):
    st.caption("G1〜G3実績・上がり順位・当日馬体重の適合などの個別ボーナス。")
    grade_bonus  = st.slider("重賞実績ボーナス", 0, 20, 5)
    agari1_bonus = st.slider("上がり3F 1位ボーナス", 0, 10, 3)
    agari2_bonus = st.slider("上がり3F 2位ボーナス", 0, 5, 2)
    agari3_bonus = st.slider("上がり3F 3位ボーナス", 0, 3, 1)
    bw_bonus     = st.slider("馬体重適正ボーナス(±10kg)", 0, 10, 2)

with st.sidebar.expander("本レース条件（ベストタイム重み用）", expanded=True):
    TARGET_GRADE = st.selectbox("本レースの格", ["G1", "G2", "G3", "L", "OP"], index=4, key="target_grade")
    TARGET_SURFACE = st.selectbox("本レースの馬場", ["芝", "ダ"], index=0, key="target_surface")
    TARGET_DISTANCE_M = st.number_input("本レースの距離 [m]", min_value=1000, max_value=3600, value=1800, step=100, key="target_distance_m")

st.sidebar.markdown("---")
st.sidebar.header("属性重み（1走スコアに掛ける係数）")
with st.sidebar.expander("性別重み", expanded=False):
    st.caption("性別に応じて増減。例：牝馬が得意な舞台なら『牝』を>1に。")
    gender_w = {g: st.slider(f"{g}", 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}

with st.sidebar.expander("脚質重み", expanded=False):
    st.caption("馬の脚質そのものにかける基本係数（ペース適性とは別枠）。")
    style_w  = {s: st.slider(f"{s}", 0.0, 2.0, 1.0) for s in ['逃げ','先行','差し','追込']}

with st.sidebar.expander("季節（四季）重み", expanded=False):
    st.caption("開催月→四季を自動判定。季節要因（暑さ寒さ）をざっくり反映。")
    season_w = {s: st.slider(f"{s}", 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}

with st.sidebar.expander("年齢重み", expanded=False):
    st.caption("年齢別の適性差を任意に調整（3〜10歳）。")
    age_w    = {str(age): st.slider(f"{age}歳", 0.0, 2.0, 1.0, 0.05) for age in range(3, 11)}

with st.sidebar.expander("枠順重み", expanded=False):
    st.caption("コース形状や馬場バイアスを枠単位で調整。")
    frame_w  = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0) for i in range(1,9)}

st.sidebar.markdown("---")
st.sidebar.header("時系列・安定性・補正")
st.sidebar.caption(
    "時系列半減期・安定性・ペース適性・斤量ペナルティなどを調整します。"
)

half_life_m  = st.sidebar.slider("時系列半減期(月)", 0.0, 12.0, 6.0, 0.5)
stab_weight  = st.sidebar.slider("安定性(小さいほど◎)の係数", 0.0, 2.0, 0.7, 0.1)
pace_gain    = st.sidebar.slider("ペース適性係数", 0.0, 3.0, 1.0, 0.1)
weight_coeff = st.sidebar.slider("斤量ペナルティ強度(pts/kg)", 0.0, 4.0, 1.0, 0.1)

with st.sidebar.expander("斤量ベース（WFA/JRA簡略）", expanded=True):
    race_date = pd.to_datetime(st.date_input("開催日", value=pd.Timestamp.today().date()))
    use_wfa_base = st.checkbox("WFA基準を使う（推奨）", value=True)

    wfa_2_early_m = st.number_input("2歳（〜9月） 牡/せん [kg]", 50.0, 60.0, 55.0, 0.5)
    wfa_2_early_f = st.number_input("2歳（〜9月） 牝 [kg]"    , 48.0, 60.0, 54.0, 0.5)
    wfa_2_late_m  = st.number_input("2歳（10-12月） 牡/せん [kg]", 50.0, 60.0, 56.0, 0.5)
    wfa_2_late_f  = st.number_input("2歳（10-12月） 牝 [kg]"    , 48.0, 60.0, 55.0, 0.5)
    wfa_3p_m      = st.number_input("3歳以上 牡/せん [kg]" , 50.0, 62.0, 57.0, 0.5)
    wfa_3p_f      = st.number_input("3歳以上 牝 [kg]"     , 48.0, 60.0, 55.0, 0.5)

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

st.sidebar.markdown("---")
st.sidebar.header("ペース / 脚質")
with st.sidebar.expander("脚質自動推定（強化）", expanded=False):
    auto_style_on   = st.checkbox("自動推定を使う（空欄を自動で埋める）", True)
    AUTO_OVERWRITE  = st.checkbox("手入力より自動を優先して上書き", False)
    NRECENT         = st.slider("直近レース数（脚質推定）", 1, 10, 5)
    HL_DAYS_STYLE   = st.slider("半減期（日・脚質用）", 30, 365, 180, 15)
    pace_mc_draws   = st.slider("ペースMC回数", 500, 30000, 5000, 500)

with st.sidebar.expander("ペース設定（自動MC / 固定）", expanded=False):
    pace_mode = st.radio("ペースの扱い", ["自動（MC）", "固定（手動）"], index=0)
    pace_fixed = st.selectbox("固定ペースを選択", ["ハイペース","ミドルペース","ややスローペース","スローペース"],
                              index=1, disabled=(pace_mode=="自動（MC）"))

with st.sidebar.expander("EPI（前圧）チューニング", expanded=False):
    epi_alpha = st.slider("逃げ係数 α", 0.0, 2.0, 1.0, 0.05)
    epi_beta  = st.slider("先行係数 β", 0.0, 2.0, 0.60, 0.05)
    thr_hi    = st.slider("閾値: ハイペース ≥", 0.30, 1.00, 0.52, 0.01)
    thr_mid   = st.slider("閾値: ミドル ≥",    0.10, 0.99, 0.30, 0.01)
    thr_slow  = st.slider("閾値: ややスロー ≥",0.00, 0.98, 0.18, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("勝率シミュレーション（モンテカルロ）")
with st.sidebar.expander("詳細設定", expanded=False):
    mc_iters   = st.slider("反復回数", 1000, 100000, 20000, 1000)
    mc_beta    = st.slider("強さ→勝率 温度β", 0.1, 5.0, 1.5, 0.1)
    mc_tau     = st.slider("安定度ノイズ係数 τ", 0.0, 2.0, 0.6, 0.05)
    mc_seed    = st.number_input("乱数Seed", 0, 999999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.header("資金・点数（購入戦略）")
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)
min_unit     = st.sidebar.selectbox("最小賭け単位", [100, 200, 300, 500], index=0)
max_lines    = st.sidebar.slider("最大点数(連系)", 1, 60, 20, 1)
scenario     = st.sidebar.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])
show_map_ui  = st.sidebar.checkbox("列マッピングUIを表示", value=False)

with st.sidebar.expander("その他（開発者向け）", expanded=False):
    orig_weight  = st.slider("OrigZ の重み (未使用)", 0.0, 1.0, 0.5, 0.05)

def collect_params():
    return {
        "lambda_part": lambda_part, "orig_weight": orig_weight,
        "gender_w": gender_w, "style_w": style_w, "season_w": season_w,
        "age_w": age_w, "frame_w": frame_w, "besttime_w": besttime_w,
        "win_w": win_w, "quin_w": quin_w, "plc_w": plc_w,
        "grade_bonus": grade_bonus, "agari1_bonus": agari1_bonus,
        "agari2_bonus": agari2_bonus, "agari3_bonus": agari3_bonus, "bw_bonus": bw_bonus,
        "half_life_m": half_life_m, "stab_weight": stab_weight,
        "pace_gain": pace_gain, "weight_coeff": weight_coeff,
        "total_budget": total_budget, "min_unit": int(min_unit),
        "max_lines": int(max_lines), "scenario": scenario,
        "auto_style_on": auto_style_on, "AUTO_OVERWRITE": AUTO_OVERWRITE,
        "NRECENT": int(NRECENT), "HL_DAYS_STYLE": int(HL_DAYS_STYLE),
        "pace_mc_draws": int(pace_mc_draws),
        "pace_mode": pace_mode, "pace_fixed": pace_fixed,
        "epi_alpha": epi_alpha, "epi_beta": epi_beta,
        "thr_hi": thr_hi, "thr_mid": thr_mid, "thr_slow": thr_slow,
        "mc_iters": int(mc_iters), "mc_beta": mc_beta,
        "mc_tau": mc_tau, "mc_seed": int(mc_seed),
        "show_map_ui": show_map_ui,
        "target_grade": TARGET_GRADE,
        "target_surface": TARGET_SURFACE,
        "target_distance_m": int(TARGET_DISTANCE_M),
        "race_date": str(race_date.date()),
        "use_wfa_base": use_wfa_base,
        "wfa_2_early_m": float(wfa_2_early_m),
        "wfa_2_early_f": float(wfa_2_early_f),
        "wfa_2_late_m": float(wfa_2_late_m),
        "wfa_2_late_f": float(wfa_2_late_f),
        "wfa_3p_m": float(wfa_3p_m),
        "wfa_3p_f": float(wfa_3p_f),
    }

def apply_params(cfg: dict):
    for k, v in cfg.items():
        st.session_state[k] = v

col_a, col_b = st.sidebar.columns(2)
if col_a.button("設定を保存"):
    cfg = json.dumps(collect_params(), ensure_ascii=False, indent=2)
    st.sidebar.download_button("JSONをDL", data=cfg, file_name="keiba_config.json", mime="application/json")
cfg_file = col_b.file_uploader("設定読み込み", type=["json"], key="cfg_up")
if cfg_file is not None:
    try:
        cfg = json.loads(cfg_file.read().decode("utf-8"))
        apply_params(cfg)
        st.sidebar.success("設定を読み込みました（必要なら再実行）。")
    except Exception as e:
        st.sidebar.error(f"設定ファイルの読み込みエラー: {e}")

# ======================== ファイルアップロード（ここでゲート） ========================
st.title("競馬予想アプリ（完成系・互換性強化版）")
st.subheader("ファイルアップロード")

excel_file = st.file_uploader("Excel（sheet0=過去走 / sheet1=出走表）", type=['xlsx'], key="excel_up")
html_file  = st.file_uploader("HTML（血統）※任意", type=['html'], key="html_up")

if excel_file is None:
    st.info("まずExcel（.xlsx）をアップロードしてください。血統HTMLは無くても動きます。")
    st.stop()

sheet0, sheet1 = load_excel_bytes(excel_file.getvalue())

def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'\s+', '', s)
    s = s.translate(str.maketrans('０１２３４５６７８９','0123456789'))
    s = s.replace('（','(').replace('）',')').replace('％','%')
    return s

def _parse_time_to_sec(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = str(x).strip()
    m = re.match(r'^(\d+):(\d+)\.(\d+)$', s)
    if m: return int(m.group(1))*60 + int(m.group(2)) + float('0.'+m.group(3))
    m = re.match(r'^(\d+)[\.:](\d+)[\.:](\d+)$', s)
    if m: return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/10
    try:  return float(s)
    except: return np.nan

def _auto_guess(col_map, pats):
    for orig, normed in col_map.items():
        for p in pats:
            if re.search(p, normed, flags=re.I):
                return orig
    return None

def _interactive_map(df, patterns, required_keys, title, state_key, show_ui=False):
    cols = list(df.columns)
    cmap = {c: _norm_col(c) for c in cols}
    auto = {k: st.session_state.get(f"{state_key}:{k}") or _auto_guess(cmap, pats)
            for k, pats in patterns.items()}
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
            mapping[key] = st.selectbox(
                key,
                options=['<未選択>'] + cols,
                index=(['<未選択>']+cols).index(default) if default in cols else 0,
                key=f"map:{state_key}:{key}"
            )
            if mapping[key] != '<未選択>':
                st.session_state[f"{state_key}:{key}"] = mapping[key]
    missing = [k for k in required_keys if mapping.get(k) in (None, '<未選択>')]
    if missing: st.stop()
    return {k: (None if v=='<未選択>' else v) for k, v in mapping.items()}

# === sheet0（過去走データ） ===
PAT_S0 = {
    '馬名'         : [r'馬名|名前|出走馬'],
    'レース日'     : [r'レース日|日付S|日付|年月日'],
    '競走名'       : [r'競走名|レース名|名称'],
    'クラス名'     : [r'クラス名|格|条件|レースグレード'],
    '頭数'         : [r'頭数|出走頭数'],
    '確定着順'     : [r'確定着順|着順(?!率)'],
    '枠'           : [r'枠|枠番'],
    '番'           : [r'馬番|番'],
    '斤量'         : [r'斤量'],
    '馬体重'       : [r'馬体重|体重'],
    '上がり3Fタイム': [r'上がり3Fタイム|上がり3F|上3Fタイム|上3F'],
    '上3F順位'     : [r'上がり3F順位|上3F順位'],
    '通過4角'      : [r'通過.*4角|4角.*通過|第4コーナー順位|4角順位'],
    '性別'         : [r'性別'],
    '年齢'         : [r'年齢|馬齢'],
    '走破タイム秒' : [r'走破タイム.*秒|走破タイム|タイム$'],
    '距離'         : [r'距離'],
    '馬場'         : [r'馬場|馬場状態'],
    '天候'         : [r'天候'],
}
REQ_S0 = ['馬名','レース日','競走名','頭数','確定着順']
MAP_S0 = _interactive_map(sheet0, PAT_S0, REQ_S0, "sheet0（過去走）", "s0", show_ui=show_map_ui)

df_score = pd.DataFrame()
for k, col in MAP_S0.items():
    if col is None: continue
    df_score[k] = sheet0[col]

# 型・パース
df_score['レース日'] = pd.to_datetime(df_score['レース日'], errors='coerce')
for c in ['頭数','確定着順','枠','番','斤量','馬体重','上3F順位','通過4角','距離']:
    if c in df_score: df_score[c] = pd.to_numeric(df_score[c], errors='coerce')
if '走破タイム秒' in df_score: df_score['走破タイム秒'] = df_score['走破タイム秒'].apply(_parse_time_to_sec)
if '上がり3Fタイム' in df_score: df_score['上がり3Fタイム'] = df_score['上がり3Fタイム'].apply(_parse_time_to_sec)

# --- 通過4角 / 頭数のクレンジング ---
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

# === sheet1（当日出走表／プロフィール） ===
PAT_S1 = {
    '馬名'   : [r'馬名|名前|出走馬'],
    '枠'     : [r'枠|枠番'],
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
MAP_S1 = _interactive_map(sheet1, PAT_S1, REQ_S1, "sheet1（出走表）", "s1", show_ui=show_map_ui)

attrs = pd.DataFrame()
for k, col in MAP_S1.items():
    if col is None: continue
    attrs[k] = sheet1[col]
for c in ['枠','番','斤量','馬体重']:
    if c in attrs: attrs[c] = pd.to_numeric(attrs[c], errors='coerce')
if 'ベストタイム' in attrs: attrs['ベストタイム秒'] = attrs['ベストタイム'].apply(_parse_time_to_sec)

# --- 馬一覧の確認・編集 ---
if '脚質' not in attrs.columns: attrs['脚質'] = ''
if '斤量' not in attrs.columns: attrs['斤量'] = np.nan
if '馬体重' not in attrs.columns: attrs['馬体重'] = np.nan

st.subheader("馬一覧・脚質・斤量・当日馬体重入力")
edited = st.data_editor(
    attrs[['枠','番','馬名','性別','年齢','脚質','斤量','馬体重']].copy(),
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込']),
        '斤量': st.column_config.NumberColumn('斤量', min_value=45, max_value=65, step=0.5),
        '馬体重': st.column_config.NumberColumn('馬体重', min_value=300, max_value=600, step=1)
    },
    use_container_width=True,
    num_rows='static'
)
horses = edited.copy()

validate_inputs(df_score, horses)

# --- 脚質 自動推定（強化版） ---
df_style = pd.DataFrame({'馬名': [], 'p_逃げ': [], 'p_先行': [], 'p_差し': [], 'p_追込': [], '推定脚質': []})

need_cols = {'馬名','レース日','頭数','通過4角'}
if need_cols.issubset(df_score.columns):
    tmp = (
        df_score[['馬名','レース日','頭数','通過4角','上3F順位']].copy()
        .dropna(subset=['馬名','レース日','頭数','通過4角'])
        .sort_values(['馬名','レース日'], ascending=[True, False])
    )
    tmp['_rn'] = tmp.groupby('馬名').cumcount()+1
    tmp = tmp[tmp['_rn'] <= int(NRECENT)].copy()

    # 時間減衰重み
    today = pd.Timestamp.today()
    tmp['_days'] = (today - pd.to_datetime(tmp['レース日'], errors='coerce')).dt.days.clip(lower=0).fillna(9999)
    tmp['_w'] = 0.5 ** (tmp['_days'] / float(HL_DAYS_STYLE))

    # 位置取り 0（先頭）〜1（最後方）
    denom = (pd.to_numeric(tmp['頭数'], errors='coerce') - 1).replace(0, np.nan)
    pos_ratio = (pd.to_numeric(tmp['通過4角'], errors='coerce') - 1) / denom
    pos_ratio = pos_ratio.clip(0, 1).fillna(0.5)

    # 上がり補強
    if '上3F順位' in tmp.columns:
        ag = pd.to_numeric(tmp['上3F順位'], errors='coerce')
        close_strength = ((3.5 - ag) / 3.5).clip(lower=0, upper=1).fillna(0.0)
    else:
        close_strength = pd.Series(0.0, index=tmp.index)

    # ロジット
    b_nige, b_sengo, b_sashi, b_oikomi = -1.2, 0.6, 0.3, -0.7
    tmp['L_nige']   = b_nige  + 1.6*(1 - pos_ratio) - 1.2*close_strength
    tmp['L_sengo']  = b_sengo + 1.1*(1 - pos_ratio) - 0.1*close_strength
    tmp['L_sashi']  = b_sashi + 1.1*(pos_ratio)     + 0.9*close_strength
    tmp['L_oikomi'] = b_oikomi+ 1.6*(pos_ratio)     + 0.5*close_strength

    rows = []
    idx2style = ['逃げ','先行','差し','追込']
    for name, g in tmp.groupby('馬名'):
        w  = g['_w'].to_numpy(); sw = w.sum()
        if sw <= 0: continue
        pr = pos_ratio.loc[g.index].to_numpy()

        def wavg(v): return float((v*w).sum()/sw)

        Ln = wavg(g['L_nige']); Lse = wavg(g['L_sengo'])
        Ls = wavg(g['L_sashi']); Lo  = wavg(g['L_oikomi'])

        vec = np.array([Ln, Lse, Ls, Lo], dtype=float)
        vec = vec - vec.max()
        p = np.exp(vec); p = p / p.sum()

        pred = idx2style[int(np.argmax(p))]

        # ガード
        pr_mean = float((pr*w).sum()/sw)
        front_share = float(((pr <= 0.15)*w).sum()/sw)
        back_share  = float(((pr >= 0.85)*w).sum()/sw)
        if pred == '逃げ' and not (pr_mean <= 0.22 or front_share >= 0.25):
            pred = '先行'
        if pred == '追込' and not (pr_mean >= 0.78 or back_share  >= 0.25):
            pred = '差し'

        rows.append([name, *p.tolist(), pred])

    if rows:
        df_style = pd.DataFrame(rows, columns=['馬名','p_逃げ','p_先行','p_差し','p_追込','推定脚質'])

        # 逃げ0頭なら最も前に行きやすい馬を逃げに（既存ロジック）
        manual_has_nige = ('脚質' in horses.columns) and horses['脚質'].eq('逃げ').any()
        if (df_style['推定脚質'].eq('逃げ').sum() == 0) and (not manual_has_nige):
            early = tmp.assign(early=(1 - pos_ratio).clip(0, 1), w=tmp['_w'].values)\
                      .groupby('馬名').apply(lambda g: float((g['early']*g['w']).sum()/g['w'].sum()))
            nige_cand = early.idxmax()
            df_style.loc[df_style['馬名'] == nige_cand, '推定脚質'] = '逃げ'

else:
    st.warning("『通過4角』『頭数』『レース日』が不足。脚質の自動推定をスキップしました。")

# --- 戦績率（%→数値）＆ベストタイム抽出（任意） ---
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
    rate = pd.DataFrame({'馬名':[],'勝率':[],'連対率':[],'複勝率':[],'ベストタイム秒':[]})

# ---- 血統（HTML）パース：任意 ----
if html_file is not None:
    blood_df = parse_blood_html_bytes(html_file.getvalue())
else:
    blood_df = pd.DataFrame({'馬名': [], '血統': []})

# === 過去走から「適正馬体重」先取り（修正済み） ===
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
else:
    best_bw_map = {}

# ===== マージ =====
for dup in ['枠','番','性別','年齢','斤量','馬体重','脚質']:
    df_score.drop(columns=[dup], errors='ignore', inplace=True)

df_score = (
    df_score
    .merge(horses[['馬名','枠','番','性別','年齢','斤量','馬体重','脚質']], on='馬名', how='left')
    .merge(blood_df, on='馬名', how='left')
)
if len(rate) > 0:
    use_cols = ['馬名'] + [c for c in ['勝率','連対率','複勝率','ベストタイム秒'] if c in rate.columns]
    df_score = df_score.merge(rate[use_cols], on='馬名', how='left')

# ===== ベストタイム正規化レンジ =====
bt_min = df_score['ベストタイム秒'].min(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_max = df_score['ベストタイム秒'].max(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_span = (bt_max - bt_min) if pd.notna(bt_min) and pd.notna(bt_max) and (bt_max > bt_min) else 1.0

# ===== スコア算出 (1走ごと) =====
st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

CLASS_BASE_BT = {"OP": 1.50, "L": 1.38, "G3": 1.19, "G2": 1.00, "G1": 0.80}
def surface_factor(surface: str) -> float:
    return 1.10 if str(surface) == "ダ" else 1.00
def distance_factor(distance_m: int) -> float:
    try:
        d = int(distance_m)
    except:
        return 1.0
    if d <= 1400:           return 1.20
    if d == 1600:           return 1.10
    if 1800 <= d <= 2200:   return 1.00
    if d >= 2400:           return 0.85
    return 1.00
def besttime_weight_final(grade: str, surface: str, distance_m: int, user_scale: float) -> float:
    base = CLASS_BASE_BT.get(str(grade), CLASS_BASE_BT["OP"])
    w = base * surface_factor(surface) * distance_factor(distance_m) * float(user_scale)
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

def calc_score(r):
    g = class_points(r)
    raw = g * (r['頭数'] + 1 - r['確定着順']) + lambda_part * g

    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r.get('性別'), 1)
    stw = style_w.get(r.get('脚質'), 1)
    fw  = frame_w.get(str(r.get('枠')), 1)
    aw  = age_w.get(str(r.get('年齢')), 1.0)

    bloodline = str(r.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    blood_bonus = 0
    for k in keys:
        if k.strip() and k.strip().lower() in bloodline:
            blood_bonus = bp
            break

    gnorm = normalize_grade_text(r.get('クラス名'))
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
                grade=st.session_state.get("target_grade", TARGET_GRADE),
                surface=st.session_state.get("target_surface", TARGET_SURFACE),
                distance_m=int(st.session_state.get("target_distance_m", TARGET_DISTANCE_M)),
                user_scale=besttime_w
            )
            bt_bonus = bt_w_final * bt_norm
    except Exception:
        pass

    kg_pen = 0.0
    try:
        kg = float(r.get('斤量', np.nan))
        if not np.isnan(kg):
            if use_wfa_base:
                sex = str(r.get('性別', ''))
                try:
                    age_i = int(r.get('年齢'))
                except Exception:
                    age_i = None
                base = wfa_base_for(sex, age_i, race_date)
            else:
                base = 56.0
            kg_pen = -max(0.0, kg - float(base)) * float(weight_coeff)
    except Exception:
        pass

    total_bonus = blood_bonus + grade_point + agari_bonus + body_bonus + rate_bonus + bt_bonus + kg_pen
    return raw * sw * gw * stw * fw * aw + total_bonus

# 1走スコアを計算
if 'レース日' in df_score.columns:
    df_score['レース日'] = pd.to_datetime(df_score['レース日'], errors='coerce')
else:
    st.error("レース日 列が見つかりません。Excelの1枚目に含めてください。")
    st.stop()

df_score['score_raw']  = df_score.apply(calc_score, axis=1)

# --- デバッグ出力（score_raw 分布） ----------
st.write("=== デバッグ: score_raw の基本統計 ===")
st.write(df_score['score_raw'].describe())
st.write("unique count:", df_score['score_raw'].nunique())

# 正規化
if df_score['score_raw'].max() == df_score['score_raw'].min():
    df_score['score_norm'] = 50.0
else:
    df_score['score_norm'] = (
        (df_score['score_raw'] - df_score['score_raw'].min()) /
        (df_score['score_raw'].max() - df_score['score_raw'].min()) * 100
    )

# ===== 時系列加重 (半減期) =====
now = pd.Timestamp.today()
df_score['_days_ago'] = (now - df_score['レース日']).dt.days
df_score['_w'] = 0.5 ** (df_score['_days_ago'] / (half_life_m * 30.4375)) if half_life_m > 0 else 1.0

def w_mean(x, w):
    w = np.array(w); x = np.array(x); s = w.sum()
    return float((x*w).sum()/s) if s>0 else np.nan

def w_std(x, w):
    w = np.array(w); x = np.array(x); s = w.sum()
    if s <= 0: return np.nan
    m = (x*w).sum()/s
    var = (w*((x-m)**2)).sum()/s
    return float(np.sqrt(var))

agg = []
for name, g in df_score.groupby('馬名'):
    avg  = g['score_norm'].mean()
    std  = g['score_norm'].std(ddof=0)
    wavg = w_mean(g['score_norm'], g['_w'])
    wstd = w_std(g['score_norm'], g['_w'])
    agg.append({'馬名':name,'AvgZ':avg,'Stdev':std,'WAvgZ':wavg,'WStd':wstd})

df_agg = pd.DataFrame(agg)
for c in ['Stdev','WStd']:
    if c in df_agg.columns:
        df_agg[c] = df_agg[c].fillna(df_agg[c].median())

# ===== ペース想定（EPI MC or 固定）＋ 最終スコア =====
df_agg['RecencyZ'] = z_score(df_agg['WAvgZ'])
df_agg['StabZ']    = z_score(-df_agg['WStd'].fillna(df_agg['WStd'].median()))

# ----- combined_style を安全に構築 -----
name_list = df_agg['馬名'].tolist()
combined_style = pd.Series(index=name_list, dtype=object)

# 1) 手入力（horses）を反映
if '脚質' in horses.columns:
    try:
        manual_series = horses.set_index('馬名')['脚質'].reindex(combined_style.index)
        combined_style.update(manual_series)
    except Exception:
        pass

# 2) 自動推定（df_style）を反映（AUTO_OVERWRITE に従う）
if not df_style.empty:
    try:
        pred_series = df_style.set_index('馬名')['推定脚質'].reindex(combined_style.index)
        if AUTO_OVERWRITE:
            combined_style.update(pred_series)
        else:
            mask_blank = combined_style.isna() | combined_style.astype(str).str.strip().eq('')
            combined_style.loc[mask_blank] = pred_series.loc[mask_blank]
    except Exception:
        pass

combined_style = combined_style.fillna('')
df_agg['脚質'] = df_agg['馬名'].map(combined_style)

# build P matrix (脚質確率行列) with safe defaults
idx2style = ['逃げ','先行','差し','追込']
H = len(name_list)
P = np.zeros((H, 4), dtype=float)

if not df_style.empty and {'p_逃げ','p_先行','p_差し','p_追込'}.issubset(df_style.columns):
    pmap = df_style.set_index('馬名')[['p_逃げ','p_先行','p_差し','p_追込']]
    for i, nm in enumerate(name_list):
        if nm in pmap.index:
            vals = pmap.loc[nm].to_numpy(dtype=float)
            s = vals.sum()
            P[i, :] = vals / s if s>0 else np.array([0.25,0.25,0.25,0.25])
        else:
            stl = combined_style.get(nm, '')
            if stl in idx2style:
                P[i, idx2style.index(stl)] = 1.0
            else:
                P[i, :] = 0.25
else:
    for i, nm in enumerate(name_list):
        stl = combined_style.get(nm, '')
        if stl in idx2style:
            P[i, idx2style.index(stl)] = 1.0
        else:
            P[i, :] = 0.25

# mark rules & pace MC
mark_rule = {
    'ハイペース':      {'逃げ':'△','先行':'△','差し':'◎','追込':'〇'},
    'ミドルペース':    {'逃げ':'〇','先行':'◎','差し':'〇','追込':'△'},
    'ややスローペース': {'逃げ':'〇','先行':'◎','差し':'△','追込':'×'},
    'スローペース':    {'逃げ':'◎','先行':'〇','差し':'△','追込':'×'},
}
mark_to_pts = {'◎':2, '〇':1, '○':1, '△':0, '×':-1}

rng_pace = np.random.default_rng(int(mc_seed) + 12345)
sum_pts = np.zeros(H, dtype=float)
pace_counter = {'ハイペース':0,'ミドルペース':0,'ややスローペース':0,'スローペース':0}

for _ in range(int(pace_mc_draws)):
    sampled = [rng_pace.choice(4, p=P[i]) for i in range(H)]
    nige  = sum(1 for s in sampled if s==0)
    sengo = sum(1 for s in sampled if s==1)

    epi = (epi_alpha*nige + epi_beta*sengo) / max(1, H)

    if   epi >= thr_hi:
        pace_t = "ハイペース"
    elif epi >= thr_mid:
        pace_t = "ミドルペース"
    elif epi >= thr_slow:
        pace_t = "ややスローペース"
    else:
        pace_t = "スローペース"
    pace_counter[pace_t] += 1

    mk = mark_rule[pace_t]
    for i, s in enumerate(sampled):
        style_name = idx2style[s]
        sum_pts[i] += mark_to_pts[ mk[style_name] ]

df_agg['PacePts'] = sum_pts / max(1, int(pace_mc_draws))
pace_type = max(pace_counter, key=lambda k: pace_counter[k]) if sum(pace_counter.values())>0 else "ミドルペース"

# 固定ペースなら上書き
if pace_mode == "固定（手動）":
    pace_type = pace_fixed
    v_pts = np.array([mark_to_pts[ mark_rule[pace_type][st] ] for st in idx2style], dtype=float)
    df_agg['PacePts'] = (P @ v_pts)

# 最終スコア
df_agg['FinalRaw'] = df_agg['RecencyZ'] + stab_weight * df_agg['StabZ'] + pace_gain * df_agg['PacePts']

# ===== FinalZ 安定化（平坦対処） =====
if (df_agg['FinalRaw'].max() - df_agg['FinalRaw'].min()) < 1e-9:
    st.warning("FinalRaw がほぼ同値になっています。小さな補正を入れてFinalZを計算します。")
    df_agg['FinalZ'] = 50 + (df_agg['WAvgZ'] - df_agg['WAvgZ'].mean()) * 0.1
else:
    df_agg['FinalZ']   = z_score(df_agg['FinalRaw'])

# デバッグ出力（FinalRaw / FinalZ の分布）
st.write("=== デバッグ: FinalRaw / FinalZ の基本統計 ===")
st.write(df_agg[['FinalRaw','FinalZ']].describe())

# ===== 勝率モンテカルロ =====
S = df_agg['FinalRaw'].to_numpy(dtype=float)
S = (S - np.nanmean(S)) / (np.nanstd(S) + 1e-9)
W = df_agg['WStd'].fillna(df_agg['WStd'].median()).to_numpy(dtype=float)
W = (W - W.min()) / (W.max() - W.min() + 1e-9)

n = len(S)
rng = np.random.default_rng(int(mc_seed))
gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(mc_iters, n))
noise  = (mc_tau * W)[None, :] * rng.standard_normal((mc_iters, n))
U = mc_beta * S[None, :] + noise + gumbel
rank_idx = np.argsort(-U, axis=1)

win_counts  = np.bincount(rank_idx[:, 0], minlength=n).astype(float)
top3_counts = np.zeros(n, dtype=float)
for k in range(3):
    top3_counts += np.bincount(rank_idx[:, k], minlength=n).astype(float)

p_win  = win_counts  / mc_iters
p_top3 = top3_counts / mc_iters

df_agg['勝率%_MC']   = (p_win  * 100).round(2)
df_agg['複勝率%_MC'] = (p_top3 * 100).round(2)

# ===== 可視化 =====
avg_st = float(df_agg['WStd'].mean())
x_mid = 50.0
y_mid = avg_st
x_min, x_max = float(df_agg['FinalZ'].min()), float(df_agg['FinalZ'].max())
y_min, y_max = float(df_agg['WStd'].min()),  float(df_agg['WStd'].max())

# 四象限用矩形（Altair が使えるなら描画、無ければ簡易表で代替）
if ALT_AVAILABLE:
    quad_rect = pd.DataFrame([
        {'x1': x_min, 'x2': x_mid, 'y1': y_mid, 'y2': y_max},
        {'x1': x_mid, 'x2': x_max, 'y1': y_mid, 'y2': y_max},
        {'x1': x_min, 'x2': x_mid, 'y1': y_min, 'y2': y_mid},
        {'x1': x_mid, 'x2': x_max, 'y1': y_min, 'y2': y_mid},
    ])
    rect = alt.Chart(quad_rect).mark_rect(opacity=0.07).encode(x='x1:Q', x2='x2:Q', y='y1:Q', y2='y2:Q')
    points = alt.Chart(df_agg).mark_circle(size=100).encode(
        x=alt.X('FinalZ:Q', title='最終偏差値'),
        y=alt.Y('WStd:Q',  title='加重標準偏差（小さいほど安定）'),
        tooltip=['馬名','WAvgZ','WStd','RecencyZ','StabZ','PacePts','勝率%_MC','複勝率%_MC']
    )
    labels = alt.Chart(df_agg).mark_text(dx=6, dy=-6, fontSize=10, color='#ffffff').encode(x='FinalZ:Q', y='WStd:Q', text='馬名:N')
    vline = alt.Chart(pd.DataFrame({'x':[x_mid]})).mark_rule(color='gray').encode(x='x:Q')
    hline = alt.Chart(pd.DataFrame({'y':[y_mid]})).mark_rule(color='gray').encode(y='y:Q')
    quad_text = alt.Chart(pd.DataFrame([ 
        {'label':'消し・大穴',   'x': (x_min + x_mid)/2, 'y': (y_mid + y_max)/2},
        {'label':'波乱・ムラ馬', 'x': (x_mid + x_max)/2, 'y': (y_mid + y_max)/2},
        {'label':'堅実ヒモ',     'x': (x_min + x_mid)/2, 'y': (y_min + y_mid)/2},
        {'label':'鉄板・本命',   'x': (x_mid + x_max)/2, 'y': (y_min + y_mid)/2},
    ])).mark_text(fontSize=14, fontWeight='bold', color='#ffffff').encode(x='x:Q', y='y:Q', text='label:N')
    chart = (rect + points + labels + vline + hline + quad_text).properties(width=700, height=420).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.write("（Altair が利用できないため、散布図の代替表示は省略しています）")
    st.table(df_agg[['馬名','FinalZ','WStd','勝率%_MC']].sort_values('FinalZ', ascending=False).head(20))

# ===== 上位馬抽出（閾値付き：FinalZ>=50、最大6頭） =====
CUTOFF = 50.0
cand = df_agg[df_agg['FinalZ'] >= CUTOFF].sort_values('FinalZ', ascending=False).copy()
topN = cand.head(6)

marks = ['◎','〇','▲','☆','△','△']
topN['印'] = marks[:len(topN)]

st.subheader("上位馬（FinalZ≧50／最大6頭）")
if len(topN) == 0:
    st.warning(f"FinalZが{CUTOFF}以上の馬がいません。")
else:
    def reason(row):
        base = f"時系列加重平均{row['WAvgZ']:.1f}、安定度{row['WStd']:.1f}、ペース点{row['PacePts']}。"
        if row['FinalZ'] >= 65: base += "非常に高評価。"
        elif row['FinalZ'] >= 55: base += "高水準。"
        if row['WStd'] < df_agg['WStd'].quantile(0.3): base += "安定感抜群。"
        elif row['WStd'] < df_agg['WStd'].median(): base += "比較的安定。"
        style = str(row.get('脚質','')).strip()
        base += {"逃げ":"楽逃げなら粘り込み有力。","先行":"先行力で押し切り濃厚。","差し":"展開が向けば末脚強烈。","追込":"直線勝負の一撃に注意。"}.get(style,"")
        return base

    topN['根拠'] = topN.apply(reason, axis=1)
    st.table(topN[['馬名','印','根拠']])

# ===== 推定勝率・複勝率表示 =====
st.subheader("■ 推定勝率・複勝率（モンテカルロ）")
prob_view = (
    df_agg[['馬名','FinalZ','WAvgZ','WStd','PacePts','勝率%_MC','複勝率%_MC']]
    .sort_values('勝率%_MC', ascending=False)
    .reset_index(drop=True)
)
st.table(prob_view)

with st.expander("この表の見方（クリックで開く）", expanded=False):
    st.markdown("""
**列の意味**
- **FinalZ**：RecencyZ + stab_weight×StabZ + pace_gain×PacePts を偏差値化。
- **WAvgZ**：時系列加重平均スコア。
- **WStd**：加重標準偏差（小さいほど安定）。
- **PacePts**：ペース適性点の期待値。
- **勝率%_MC／複勝率%_MC**：モンテカルロでの推定。
""")

# ===== 展開図 =====
df_map = horses.copy()
df_map['脚質'] = df_map['馬名'].map(combined_style).fillna(df_map.get('脚質', ''))
df_map['番'] = pd.to_numeric(df_map['番'].astype(str).str.translate(str.maketrans('０１２３４５６７８９','0123456789')), errors='coerce')
df_map = df_map.dropna(subset=['番']).astype({'番': int})
df_map['脚質'] = pd.Categorical(df_map['脚質'], categories=['逃げ','先行','差し','追込'], ordered=True)

# ===== 現状サマリー（表） =====
st.subheader("現状サマリー（表）")
st.caption(f"ペース: {pace_type}（{'固定' if pace_mode=='固定（手動）' else '自動MC'}）")

style_order = ['逃げ','先行','差し','追込']
style_counts = df_map['脚質'].value_counts().reindex(style_order).fillna(0).astype(int)
total_heads = int(style_counts.sum()) if style_counts.sum() > 0 else 1
style_pct = (style_counts / total_heads * 100).round(1)
pace_summary = pd.DataFrame([{
    '想定ペース': pace_type,
    '逃げ':  f"{style_counts['逃げ']}頭（{style_pct['逃げ']}%）",
    '先行':  f"{style_counts['先行']}頭（{style_pct['先行']}%）",
    '差し':  f"{style_counts['差し']}頭（{style_pct['差し']}%）",
    '追込':  f"{style_counts['追込']}頭（{style_pct['追込']}%）",
}])
st.table(pace_summary)

# 印つき上位馬
pick_df = (
    topN[['印','馬名']]
    .merge(df_agg[['馬名','FinalZ','WStd','勝率%_MC','複勝率%_MC']], on='馬名', how='left')
    .rename(columns={'勝率%_MC':'勝率(%)','複勝率%_MC':'複勝率(%)'})
)
for c in ['FinalZ','WStd','勝率(%)','複勝率(%)']:
    pick_df[c] = pick_df[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
st.table(pick_df[['印','馬名','FinalZ','WStd','勝率(%)','複勝率(%)']])

# ===== 展開ロケーション（視覚） =====
df_map_show = df_map.sort_values(['番'])
if len(df_map_show) > 0:
    fig, ax = plt.subplots(figsize=(10,3))
    colors = {'逃げ':'red', '先行':'orange', '差し':'green', '追込':'blue'}
    for _, row in df_map_show.iterrows():
        if row['脚質'] in colors:
            x = row['番']; y = ['逃げ','先行','差し','追込'].index(row['脚質'])
            ax.scatter(x, y, color=colors[row['脚質']], s=200)
            lab = row['馬名']
            ax.text(x, y, lab, ha='center', va='center', color='white', fontsize=9, weight='bold',
                    bbox=dict(facecolor=colors[row['脚質']], alpha=0.7, boxstyle='round'),
                    fontproperties=jp_font)
    ax.set_yticks([0,1,2,3]); ax.set_yticklabels(['逃げ','先行','差し','追込'], fontproperties=jp_font)
    ax.set_xticks(sorted(df_map_show['番'].unique())); ax.set_xticklabels([f"{i}番" for i in sorted(df_map_show['番'].unique())], fontproperties=jp_font)
    ax.set_xlabel("馬番", fontproperties=jp_font); ax.set_ylabel("脚質", fontproperties=jp_font)
    ax.set_title(f"展開ロケーション（{pace_type}想定）", fontproperties=jp_font)
    st.pyplot(fig)
else:
    st.info("展開ロケーション：表示対象がありません（馬番が未入力かも）。")

# ===== 重賞ハイライト（好走と出走歴の両方を表示） =====
race_col = next((c for c in ['レース名','競走名','レース','名称'] if c in df_score.columns), None)
ag_col   = next((c for c in ['上がり3Fタイム','上がり3F','上がり３Ｆ','上3Fタイム','上3F'] if c in df_score.columns), None)
finish_col = '確定着順'

def grade_from_row(row):
    g = normalize_grade_text(row.get('クラス名')) if 'クラス名' in row else None
    if not g and race_col:
        g = normalize_grade_text(row.get(race_col))
    return g

def _clean_one_line(v):
    if pd.isna(v): return ''
    return str(v).replace('\r','').replace('\n','').strip()

def _fmt_ag(v):
    if v in [None, '', 'nan']: return ''
    s = str(v).replace('秒','').strip()
    try: return f"{float(s):.1f}"
    except: return s

dfg_all = df_score.copy()
dfg_all['GradeN']   = dfg_all.apply(grade_from_row, axis=1)
dfg_all['着順num']  = pd.to_numeric(dfg_all[finish_col], errors='coerce')
dfg_all['_date']    = pd.to_datetime(dfg_all['レース日'], errors='coerce')
dfg_all['_date_str']= dfg_all['_date'].dt.strftime('%Y.%m.%d').fillna('日付不明')
if race_col: dfg_all[race_col] = dfg_all[race_col].map(_clean_one_line)
if ag_col:   dfg_all[ag_col]   = dfg_all[ag_col].map(_fmt_ag)
dfg_all = dfg_all[dfg_all['GradeN'].isin(['G1','G2','G3'])].copy()

def make_table_all(d: pd.DataFrame) -> pd.DataFrame:
    races = d[race_col].fillna('（不明）') if race_col else pd.Series(['（不明）']*len(d), index=d.index)
    ags   = d[ag_col] if ag_col else pd.Series(['']*len(d), index=d.index)
    out = pd.DataFrame({
        'レース名': races,
        '格': d['GradeN'].values,
        '着': d['着順num'].fillna('').map(lambda x: '' if x=='' else int(x)).values,
        '日付': d['_date_str'].values,
        '上がり3F': ags.values,
    })
    return out

tables_all = {name: make_table_all(d) for name, d in dfg_all.sort_values(['馬名','_date'], ascending=[True, False]).groupby('馬名')}

thr_map = {'G1':5, 'G2':4, 'G3':3}
dfg_good = dfg_all[dfg_all.apply(lambda r: r['着順num'] <= thr_map.get(r['GradeN'], 0), axis=1)]
def make_table_good(d: pd.DataFrame) -> pd.DataFrame:
    return make_table_all(d)
tables_good = {name: make_table_good(d) for name, d in dfg_good.groupby('馬名')}

st.subheader("■ 重賞ハイライト（好走＋出走歴）")
for name in topN['馬名'].tolist():
    st.markdown(f"**{name}**")
    t_good = tables_good.get(name)
    t_all  = tables_all.get(name)
    if t_good is not None and not t_good.empty:
        st.write("好走（規定着内）")
        st.table(t_good)
    elif t_all is not None and not t_all.empty:
        st.write("好走は該当なし（出走歴あり・直近）")
        st.table(t_all.head(5))
    else:
        st.write("重賞出走なし")

# ===== horses 情報付与（短評） =====
印map = dict(zip(topN['馬名'], topN['印']))
horses2 = horses.merge(df_agg[['馬名','WAvgZ','WStd','FinalZ','脚質','PacePts']], on='馬名', how='left')
horses2['印'] = horses2['馬名'].map(印map).fillna('')
horses2 = horses2.merge(blood_df, on='馬名', how='left', suffixes=('', '_血統'))

def ai_comment(row):
    base = ""
    if row['印'] == '◎':
        base += "本命評価。" + ("高い安定感で信頼度抜群。" if row['WStd'] <= 8 else "能力上位もムラあり。")
    elif row['印'] == '〇':
        base += "対抗評価。" + ("近走安定しており軸候補。" if row['WStd'] <= 10 else "展開ひとつで逆転も。")
    elif row['印'] in ['▲','☆']:
        base += "上位グループの一角。" + ("ムラがあり一発タイプ。" if row['WStd'] > 15 else "安定型で堅実。")
    elif row['印'] == '△':
        base += "押さえ候補。" + ("堅実だが勝ち切るまでは？" if row['WStd'] < 12 else "展開次第で浮上も。")
    else:
        if pd.notna(row['WAvgZ']) and pd.notna(row['WStd']):
            base += ("実力十分。ヒモ穴候補。" if (row['WAvgZ'] >= 55 and row['WStd'] < 13)
                     else ("実績からは厳しい。" if row['WAvgZ'] < 45 else "決定打に欠ける。"))
    if pd.notna(row['WStd']) and row['WStd'] >= 18:
        base += "波乱含み。"
    elif pd.notna(row['WStd']) and row['WStd'] <= 8:
        base += "非常に安定。"

    bloodtxt = str(row.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    for k in keys:
        if k.strip() and k.strip().lower() in bloodtxt:
            base += f"血統的にも注目（{k.strip()}系統）。"
            break

    style = str(row.get('脚質','')).strip()
    base += {
        "逃げ":"ハナを奪えれば粘り込み十分。",
        "先行":"先行力を活かして上位争い。",
        "差し":"展開が向けば末脚強烈。",
        "追込":"直線勝負の一撃に期待。"
    }.get(style, "")
    return base

horses2['短評'] = horses2.apply(ai_comment, axis=1)

st.subheader("■ 全頭AI診断コメント")
# --- デバッグ & 安全ガード（horses2 表示直前に挿入） ---
# 馬名の正規化（merge 前にやっておくと安全）
try:
    if '馬名' in horses2.columns:
        horses2['馬名'] = horses2['馬名'].astype(str).str.strip()
except Exception:
    pass

# df_agg / blood_df 側も念のため正規化（存在すれば）
try:
    if '馬名' in df_agg.columns:
        df_agg['馬名'] = df_agg['馬名'].astype(str).str.strip()
except Exception:
    pass
try:
    if '馬名' in blood_df.columns:
        blood_df['馬名'] = blood_df['馬名'].astype(str).str.strip()
except Exception:
    pass

# 期待列をチェックして無ければ作る（デフォルト値）
expected_cols = ['馬名','印','脚質','血統','短評','WAvgZ','WStd']
missing = [c for c in expected_cols if c not in horses2.columns]

if missing:
    st.warning(f"表示に必要な列が不足しています： {missing}。自動補完します（後で原因を確認してください）。")
    # 血統はblood_dfから取れるなら優先して埋める
    if '血統' in missing:
        if '馬名' in horses2.columns and not blood_df.empty and '馬名' in blood_df.columns and '血統' in blood_df.columns:
            horses2 = horses2.merge(blood_df[['馬名','血統']], on='馬名', how='left')
        # まだ無ければ空文字で埋める
        if '血統' not in horses2.columns:
            horses2['血統'] = ''

    # WAvgZ/WStdはdf_aggから埋める
    for col in ['WAvgZ','WStd']:
        if col in missing:
            if '馬名' in horses2.columns and '馬名' in df_agg.columns and col in df_agg.columns:
                horses2 = horses2.merge(df_agg[['馬名',col]], on='馬名', how='left')
            if col not in horses2.columns:
                horses2[col] = np.nan

    # 印（短評、脚質）は基本空白で埋める（後処理で上書きされる）
    if '印' not in horses2.columns:
        horses2['印'] = ''
    if '短評' not in horses2.columns:
        horses2['短評'] = ''
    if '脚質' not in horses2.columns:
        horses2['脚質'] = horses2.get('脚質', '').fillna('')

# 最終確認：欠けている列が残っていれば表示して停止ではなく補完
still_missing = [c for c in expected_cols if c not in horses2.columns]
if still_missing:
    st.error(f"依然として列が足りません：{still_missing}。処理を継続しますが、出力が不完全になる可能性があります。")
    for c in still_missing:
        if c in ['WAvgZ','WStd']:
            horses2[c] = np.nan
        else:
            horses2[c] = ''

# （任意）短評を再計算（ai_comment が定義済みなら）
try:
    if '短評' in horses2.columns:
        horses2['短評'] = horses2.apply(ai_comment, axis=1)
except Exception as e:
    st.warning(f"短評の再計算で例外が発生しました: {e}")

# デバッグ出力（列一覧）
st.write(">>> horses2 の列一覧（表示直前）:", horses2.columns.tolist())

st.dataframe(horses2[['馬名','印','脚質','血統','短評','WAvgZ','WStd']])

# ======================== 買い目生成＆資金配分 ========================
h1 = topN.iloc[0]['馬名'] if len(topN) >= 1 else None
h2 = topN.iloc[1]['馬名'] if len(topN) >= 2 else None

symbols = topN['印'].tolist()
names   = topN['馬名'].tolist()
others_names   = names[1:] if len(names) > 1 else []
others_symbols = symbols[1:] if len(symbols) > 1 else []

three = ['馬連','ワイド','馬単']

def round_to_unit(x, unit):
    return int(np.floor(x / unit) * unit)

# まずは従来ロジックで買い目候補と初期配分（金額：整数のまま）
main_share = 0.5
pur1 = round_to_unit(total_budget * main_share * (1/4), min_unit)  # 単勝
pur2 = round_to_unit(total_budget * main_share * (3/4), min_unit)  # 複勝
rem  = total_budget - (pur1 + pur2)

win_each   = round_to_unit(pur1 / 2, min_unit)
place_each = round_to_unit(pur2 / 2, min_unit)

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
pair_candidates = []   # (券種, 表示印, 本命, 相手, 優先度)
tri_candidates  = []   # 三連複 (◎-x-y)
tri1_candidates = []   # 三連単フォーメーション (◎-s-t)

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
    cand = sorted(cand, key=lambda x: x[-1], reverse=True)[:max_lines]
    K = len(cand)
    if K > 0 and rem >= min_unit:
        base = round_to_unit(rem / K, min_unit)
        amounts = [base]*K
        leftover = rem - base*K
        i = 0
        while leftover >= min_unit and i < K:
            amounts[i] += min_unit
            leftover -= min_unit
            i += 1
        for (typ, mks, base_h, pair_h, _), amt in zip(cand, amounts):
            bets.append({'券種':typ,'印':mks,'馬':base_h,'相手':pair_h,'金額':amt})
    else:
        st.info("連系はスキップ（相手不足 or 予算不足）")

elif scenario == 'ちょい余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 で配分")
    cand_wide = sorted([c for c in pair_candidates if c[0]=='ワイド'], key=lambda x: x[-1], reverse=True)
    cand_tri  = sorted(tri_candidates, key=lambda x: x[-1], reverse=True)
    cut_w = min(len(cand_wide), max_lines//2 if max_lines>1 else 1)
    cut_t = min(len(cand_tri),  max_lines - cut_w)
    cand_wide, cand_tri = cand_wide[:cut_w], cand_tri[:cut_t]
    K = len(cand_wide) + len(cand_tri)
    if K>0 and rem >= min_unit:
        base = round_to_unit(rem / K, min_unit)
        amounts = [base]*K
        leftover = rem - base*K
        i = 0
        while leftover >= min_unit and i < K:
            amounts[i] += min_unit
            leftover -= min_unit
            i += 1
        all_cand = cand_wide + cand_tri
        for (typ, mks, base_h, pair_h, _), amt in zip(all_cand, amounts):
            bets.append({'券種':typ,'印':mks,'馬':base_h,'相手':pair_h,'金額':amt})
    else:
        st.info("連系はスキップ（相手不足 or 予算不足）")

elif scenario == '余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 ＋ 三連単フォーメーション で配分")
    cand_wide = sorted([c for c in pair_candidates if c[0]=='ワイド'], key=lambda x: x[-1], reverse=True)
    cand_tri  = sorted(tri_candidates, key=lambda x: x[-1], reverse=True)
    cand_tri1 = sorted(tri1_candidates, key=lambda x: x[-1], reverse=True)
    r_w, r_t, r_t1 = 2, 2, 1
    denom = r_w + r_t + r_t1
    q_w = max(1, (max_lines * r_w)//denom)
    q_t = max(1, (max_lines * r_t)//denom)
    q_t1= max(1, max_lines - q_w - q_t)
    cand_wide, cand_tri, cand_tri1 = cand_wide[:q_w], cand_tri[:q_t], cand_tri1[:q_t1]
    K = len(cand_wide) + len(cand_tri) + len(cand_tri1)
    if K>0 and rem >= min_unit:
        base = round_to_unit(rem / K, min_unit)
        amounts = [base]*K
        leftover = rem - base*K
        i = 0
        while leftover >= min_unit and i < K:
            amounts[i] += min_unit
            leftover -= min_unit
            i += 1
        all_cand = cand_wide + cand_tri + cand_tri1
        for (typ, mks, base_h, pair_h, _), amt in zip(all_cand, amounts):
            bets.append({'券種':typ,'印':mks,'馬':base_h,'相手':pair_h,'金額':amt})
    else:
        st.info("連系はスキップ（相手不足 or 予算不足）")

_df = pd.DataFrame(bets)

# === ハーフケリー（任意・全券種対応） ===
st.subheader("ハーフケリー配分（任意）")
use_kelly = st.checkbox("ハーフケリーで全買い目を再配分する（オッズ入力が必要）", value=False)

if use_kelly and len(_df) > 0:
    if not SKLEARN_AVAILABLE:
        st.error("学習/確率推定に必要なライブラリ（scikit-learn 等）が見つかりません。\n"
                 "ハーフケリー再配分は利用できません。インストールする場合のコマンド例：\n\n"
                 "`pip install scikit-learn lightgbm shap`")
    else:
        idx_of = {name: i for i, name in enumerate(name_list)}

        def pair_prob(i, j, kind: str) -> float:
            if kind == 'ワイド':
                top3 = rank_idx[:, :3]
                ci = np.any(top3 == i, axis=1); cj = np.any(top3 == j, axis=1)
                return (ci & cj).mean()
            if kind == '馬連':
                return (((rank_idx[:,0]==i) & (rank_idx[:,1]==j)) |
                        ((rank_idx[:,0]==j) & (rank_idx[:,1]==i))).mean()
            if kind == '馬単':
                return ((rank_idx[:,0]==i) & (rank_idx[:,1]==j)).mean()
            return np.nan

        def trio_prob(i, j, k) -> float:
            top3 = rank_idx[:, :3]
            ci = np.any(top3 == i, axis=1)
            cj = np.any(top3 == j, axis=1)
            ck = np.any(top3 == k, axis=1)
            return (ci & cj & ck).mean()

        def trifecta_prob(i, j, k) -> float:
            return ((rank_idx[:,0]==i) & (rank_idx[:,1]==j) & (rank_idx[:,2]==k)).mean()

        q_list = []
        for _, r in _df.iterrows():
            typ, h, a = r['券種'], str(r['馬']), str(r['相手'])
            q = np.nan
            try:
                if typ == '単勝':
                    q = float(p_win[idx_of[h]])
                elif typ == '複勝':
                    q = float(p_top3[idx_of[h]])
                elif typ in ('ワイド','馬連','馬単') and a:
                    i = idx_of[h]; j = idx_of[a]
                    q = pair_prob(i, j, typ)
                elif typ == '三連複' and a:
                    parts = [p.strip() for p in a.split('／') if p.strip()]
                    if len(parts) >= 2:
                        i = idx_of[h]; j = idx_of[parts[0]]; k = idx_of[parts[1]]
                        q = trio_prob(i, j, k)
                elif typ in ('三連単','三連単フォーメーション') and a:
                    parts = [p.strip() for p in a.split('／') if p.strip()]
                    if len(parts) >= 2:
                        i = idx_of[h]; j = idx_of[parts[0]]; k = idx_of[parts[1]]
                        q = trifecta_prob(i, j, k)
            except Exception:
                q = np.nan
            q_list.append(q)

        _df['的中確率q'] = q_list

        # 2) オッズ入力UI
        if '想定オッズ(倍)' not in _df.columns:
            _df['想定オッズ(倍)'] = np.nan

        st.caption("※ 例）期待払戻 420円/100円 → 4.2 と入力。未入力は0扱いで賭けません。")
        _df = st.data_editor(
            _df,
            column_config={
                '想定オッズ(倍)': st.column_config.NumberColumn('想定オッズ(倍)', min_value=1.01, step=0.01),
            },
            use_container_width=True,
            num_rows='dynamic',
            disabled=['券種','印','馬','相手','金額','的中確率q']
        )

        # 3) f* を計算し、ハーフケリーで賭け額を算出
        o = pd.to_numeric(_df['想定オッズ(倍)'], errors='coerce')
        q = pd.to_numeric(_df['的中確率q'], errors='coerce')

        f_star = (q * o - 1) / (o - 1)
        f_star = f_star.where((o > 1.01) & q.notna(), np.nan)
        f_star = f_star.clip(lower=0).fillna(0.0)

        bankroll = float(total_budget)
        stake_raw = bankroll * 0.5 * f_star  # ハーフケリー

        total_stake = float(stake_raw.sum())
        if total_stake > bankroll and total_stake > 0:
            stake_raw *= bankroll / total_stake

        stake_rounded = (np.floor(stake_raw / int(min_unit)) * int(min_unit)).astype(int)
        rem_k = int(bankroll - stake_rounded.sum())
        i = 0
        while rem_k >= int(min_unit) and i < len(stake_rounded):
            stake_rounded[i] += int(min_unit)
            rem_k -= int(min_unit)
            i += 1

        _df['金額'] = stake_rounded.astype(int)

else:
    spent = int(_df['金額'].fillna(0).replace('',0).sum()) if len(_df)>0 else 0
    diff = total_budget - spent
    if diff != 0 and len(_df) > 0:
        for idx in _df.index:
            cur = int(_df.at[idx,'金額'])
            new = cur + diff
            if new >= 0 and new % min_unit == 0:
                _df.at[idx,'金額'] = new
                break

# 表示用に“円”フォーマット
_df_disp = _df.copy()
if len(_df_disp)>0:
    _df_disp['金額'] = _df_disp['金額'].map(lambda x: f"{int(x):,}円" if pd.notna(x) and int(x)>0 else "")

unique_types = __df_disp['券種'].unique().tolist() if len(_df_disp)>0 else []
tabs = st.tabs(['サマリー'] + unique_types) if len(unique_types)>0 else st.tabs(['サマリー'])
with tabs[0]:
    st.subheader("■ 最終買い目一覧（全券種まとめ）")
    if len(_df_disp)==0: st.info("現在、買い目はありません。")
    else:
        st.table(_df_disp[['券種','印','馬','相手','金額']])

for i, typ in enumerate(unique_types, start=1):
    with tabs[i]:
        df_this = _df_disp[_df_disp['券種'] == typ]
        st.subheader(f"{typ} 買い目一覧")
        st.table(df_this[['券種','印','馬','相手','金額']]) if len(df_this)>0 else st.info(f"{typ} の買い目はありません。")

# ---------------------------
# LightGBM + 特徴量生成 + 学習 + 予測（安全化）
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("★ LightGBM モデル学習 (特徴量 自動生成)")
target_choice = st.sidebar.selectbox("目的変数", ['win','top3'], index=0)
split_mode = st.sidebar.radio("学習/検証 分割方法", ['日付分割','割合分割'], index=0)
if split_mode == '日付分割':
    cutoff_date = st.sidebar.date_input("学習終了日（学習:それ以前 / 検証:それ以降）",
                                        value=pd.to_datetime(df_score['レース日'].min()).date() if 'レース日' in df_score else pd.Timestamp.today().date())
else:
    test_frac = st.sidebar.slider("検証データ割合 (fraction)", 0.05, 0.5, 0.2, 0.01)

use_lgb = st.sidebar.checkbox("LightGBM を使う（利用可能なら）", value=True)
n_estimators = int(st.sidebar.number_input("num_boost_round / n_estimators", min_value=10, max_value=5000, value=200, step=10))
learning_rate = float(st.sidebar.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01))
max_depth = int(st.sidebar.slider("max_depth (0=none)", 0, 16, 6))
num_leaves = int(st.sidebar.slider("num_leaves", 2, 512, 31))
random_state = int(st.sidebar.number_input("乱数シード", min_value=0, max_value=999999, value=42))

st.sidebar.markdown("---")
do_train = st.sidebar.button("学習 & 評価 実行")

def class_point_guess(x):
    try:
        x = str(x)
        if 'G1' in x or 'GⅠ' in x or 'G 1' in x: return 10
        if 'G2' in x or 'GⅡ' in x or 'G 2' in x: return 8
        if 'G3' in x or 'GⅢ' in x or 'G 3' in x: return 6
        if 'リステッド' in x or re.search(r'\bL\b', x): return 5
        if 'オープン' in x or 'OP' in x: return 4
        if re.search(r'3\s*勝', x): return 3
        if re.search(r'2\s*勝', x): return 2
        if re.search(r'1\s*勝', x): return 1
    except Exception:
        pass
    return 1

@st.cache_data(show_spinner=False)
def build_features_time_aware(df_score, default_half_months=6.0):
    s = df_score.copy()
    if 'レース日' in s.columns:
        s['レース日'] = pd.to_datetime(s['レース日'], errors='coerce')
    else:
        raise ValueError("df_score に 'レース日' 列が必要です。")
    if 'クラス名' in s.columns:
        s['_class_pt'] = s['クラス名'].apply(class_point_guess)
    else:
        s['_class_pt'] = 1
    s = s.sort_values('レース日').reset_index(drop=True)
    if '競走名' in s.columns:
        s['_race_key'] = s['レース日'].dt.strftime('%Y%m%d') + '||' + s['競走名'].astype(str)
    else:
        s['_race_key'] = s['レース日'].dt.strftime('%Y%m%d') + '||' + s.get('レース名', 'unknown').astype(str)
    race_keys = s['_race_key'].unique().tolist()

    feats = []
    horse_history = {}
    for rk in race_keys:
        race_rows = s[s['_race_key'] == rk]
        if race_rows.empty: continue
        race_date = pd.to_datetime(race_rows['レース日'].iloc[0])
        for _, row in race_rows.iterrows():
            name = row.get('馬名')
            if pd.isna(name): continue
            past = horse_history.get(name, [])
            n_start = len(past)
            wins = sum(1 for p in past if p.get('確定着順') == 1)
            top3 = sum(1 for p in past if (p.get('確定着順') is not None and p.get('確定着順') <= 3))
            avg_finish = np.nan
            avg_classpt = np.nan
            avg_3frank = np.nan
            avg_weight = np.nan
            days_since_last = np.nan
            recency_wavg = np.nan
            recency_wstd = np.nan
            best_time = np.nan
            if n_start > 0:
                finishes = np.array([p.get('確定着順') if p.get('確定着順') is not None else np.nan for p in past], dtype=float)
                classpts = np.array([p.get('_class_pt', 1) for p in past], dtype=float)
                ag3 = np.array([p.get('上3F順位') if p.get('上3F順位') is not None else np.nan for p in past], dtype=float)
                weights = np.array([p.get('斤量') if p.get('斤量') is not None else np.nan for p in past], dtype=float)
                times = np.array([p.get('走破タイム秒') if p.get('走破タイム秒') is not None else np.nan for p in past], dtype=float)
                if len(finishes) > 0:
                    avg_finish = np.nanmean(finishes)
                if len(classpts) > 0:
                    avg_classpt = np.nanmean(classpts)
                if len(ag3) > 0:
                    avg_3frank = np.nanmean(ag3)
                if len(weights) > 0:
                    avg_weight = np.nanmean(weights)
                if np.any(~np.isnan(times)):
                    best_time = np.nanmin(times)
                days = np.array([(race_date - pd.to_datetime(p.get('レース日'))).days if p.get('レース日') is not None else 9999 for p in past], dtype=float)
                hl_days = 30.4375 * max(0.1, default_half_months)
                w = 0.5 ** (days / hl_days)
                values_for_recency = np.array([ (p.get('score_raw') if p.get('score_raw') is not None else ((p.get('頭数',1)+1 - (p.get('確定着順') or 1)) * p.get('_class_pt',1)) ) for p in past ], dtype=float)
                if np.sum(w) > 0:
                    recency_wavg = float((values_for_recency * w).sum() / w.sum())
                    m = recency_wavg
                    recency_wstd = float(np.sqrt(((w * ((values_for_recency - m) ** 2)).sum()) / (w.sum()))) if w.sum() > 0 else 0.0
                last_dates = [p.get('レース日') for p in past if p.get('レース日') is not None]
                if len(last_dates) > 0:
                    days_since_last = (race_date - pd.to_datetime(last_dates[-1])).days
            rec = {
                'race_key': rk,
                'race_date': race_date,
                '馬名': name,
                'n_start': n_start,
                'wins': wins,
                'top3': top3,
                'win_rate': (wins / n_start) if n_start > 0 else np.nan,
                'top3_rate': (top3 / n_start) if n_start > 0 else np.nan,
                'avg_finish': avg_finish,
                'avg_classpt': avg_classpt,
                'avg_3frank': avg_3frank,
                'avg_weight': avg_weight,
                'best_time': best_time,
                'days_since_last': days_since_last,
                'recency_wavg': recency_wavg,
                'recency_wstd': recency_wstd,
                'cur_class_pt': row.get('_class_pt', 1),
                'cur_headcount': row.get('頭数', np.nan),
                'cur_rank': row.get('確定着順', np.nan),
                'cur_上3F順位': row.get('上3F順位', np.nan),
                'cur_通過4角': row.get('通過4角', np.nan),
                'cur_斤量': row.get('斤量', np.nan),
                'cur_馬体重': row.get('馬体重', np.nan),
                'cur_枠': row.get('枠', np.nan),
                'cur_性別': row.get('性別', np.nan),
                'cur_年齢': row.get('年齢', np.nan),
                'score_raw_row': row.get('score_raw', np.nan)
            }
            feats.append(rec)
        # append current race rows to history
        for _, row in race_rows.iterrows():
            name = row.get('馬名')
            if pd.isna(name): continue
            horse_history.setdefault(name, []).append(row.to_dict())

    feats_df = pd.DataFrame(feats)
    return feats_df

# training flow
if do_train:
    # sklearn 未検出時の挙動を親切に案内
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn が見つかりません。学習機能を実行するには以下を実行して下さい（環境に合わせて）。")
        st.code("pip install scikit-learn lightgbm shap", language='bash')
        st.info("代替として、FinalZ による簡易ランクを表示します。")
        simple_pred = df_agg.sort_values('FinalZ', ascending=False).reset_index(drop=True)
        marks = ['◎','〇','▲','☆','△','△']
        simple_pred['印'] = [marks[i] if i < len(marks) else '' for i in range(len(simple_pred))]
        st.table(simple_pred[['印','馬名','FinalZ','WAvgZ','WStd','勝率%_MC']].head(12))
    else:
        t0 = time.time()
        st.info("特徴量生成中...")
        with st.spinner("特徴量を作成しています..."):
            feats_df = build_features_time_aware(df_score)

        feats_df['target_win'] = feats_df['cur_rank'].apply(lambda x: 1 if (pd.notna(x) and float(x)==1.0) else 0)
        feats_df['target_top3'] = feats_df['cur_rank'].apply(lambda x: 1 if (pd.notna(x) and float(x) <= 3.0) else 0)
        target_col = 'target_win' if target_choice == 'win' else 'target_top3'

        feats_df['n_start'] = feats_df['n_start'].fillna(0)
        feats_trainable = feats_df[feats_df['n_start'] >= 1].copy()
        if len(feats_trainable) < 50:
            st.warning("学習可能な行数が少なすぎます（<50）。少ないデータでの学習は過学習に注意してください。")

        use_features = [
            'n_start','win_rate','top3_rate','avg_finish','avg_classpt','avg_3frank','avg_weight',
            'best_time','days_since_last','recency_wavg','recency_wstd',
            'cur_class_pt','cur_headcount','cur_上3F順位','cur_通過4角','cur_斤量','cur_馬体重','cur_枠','cur_年齢'
        ]
        use_features = [f for f in use_features if f in feats_trainable.columns]
        feats_trainable[use_features] = feats_trainable[use_features].fillna(feats_trainable[use_features].median())

        cat_cols = []
        if 'cur_枠' in use_features:
            feats_trainable['cur_枠'] = feats_trainable['cur_枠'].astype(float).astype(int).astype(str)
            cat_cols.append('cur_枠')

        X = feats_trainable[use_features].copy()
        y = feats_trainable[target_col].astype(int)

        if split_mode == '日付分割':
            cutoff = pd.to_datetime(cutoff_date)
            mask_train = feats_trainable['race_date'] <= cutoff
            X_train, X_val = X[mask_train].copy(), X[~mask_train].copy()
            y_train, y_val = y[mask_train].copy(), y[~mask_train].copy()
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_frac, random_state=random_state, stratify=y if y.nunique()>1 else None)

        numeric_cols = [c for c in X_train.columns if c not in cat_cols]
        scaler = StandardScaler()
        if len(numeric_cols) > 0:
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

        model = None
        if use_lgb and LGB_AVAILABLE:
            lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols if len(cat_cols)>0 else 'auto', free_raw_data=False)
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': learning_rate,
                'num_leaves': num_leaves,
                'max_depth': (max_depth if max_depth>0 else -1),
                'verbosity': -1,
                'seed': random_state
            }
            model = lgb.train(params, lgb_train, num_boost_round=n_estimators)
            y_pred_prob = model.predict(X_val)
            y_pred = (y_pred_prob >= 0.5).astype(int)
        else:
            model = HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=learning_rate, max_leaf_nodes=max(2, num_leaves), random_state=random_state)
            model.fit(X_train, y_train)
            try:
                y_pred_prob = model.predict_proba(X_val)[:,1]
            except:
                y_pred_prob = model.predict(X_val)
            y_pred = (y_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_prob) if len(np.unique(y_val))>1 else float('nan')
        acc = accuracy_score(y_val, y_pred)
        ll = log_loss(y_val, y_pred_prob, labels=[0,1]) if len(np.unique(y_val))>1 else float('nan')

        st.subheader("モデル評価")
        st.write(f"学習: {len(X_train)} 行 / 検証: {len(X_val)} 行")
        st.write(f"AUC: {auc:.4f}  Accuracy: {acc:.3f}  LogLoss: {ll:.3f}")
        st.text("分類レポート（検証）:")
        st.text(classification_report(y_val, y_pred, zero_division=0))

        # SHAP / FI
        st.subheader("説明性（Feature Importance / SHAP）")
        try:
            if SHAP_AVAILABLE and LGB_AVAILABLE and isinstance(model, lgb.basic.Booster):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_val)
                st.write("SHAP Summary (検証データ)")
                fig_shap = shap.summary_plot(shap_vals, X_val, show=False)
                st.pyplot(fig_shap, bbox_inches='tight')
            elif SHAP_AVAILABLE:
                explainer = shap.Explainer(model, X_train)
                shap_vals = explainer(X_val)
                fig_shap = shap.summary_plot(shap_vals, X_val, show=False)
                st.pyplot(fig_shap, bbox_inches='tight')
            else:
                st.info("shap が利用できないため、feature_importances_ を表示します。 (pip install shap で詳細表示可能)")
                if hasattr(model, 'feature_importances_'):
                    fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    fi.head(20).plot.bar(ax=ax)
                    ax.set_title("Feature importances")
                    st.pyplot(fig)
                else:
                    st.write("モデルに feature_importances_ がありません。")
        except Exception as e:
            st.error(f"説明性算出中にエラー: {e}")

        # 当日出走馬への予測
        st.subheader("当日出走馬への予測（出走表が存在すれば）")
        try:
            last_date = pd.to_datetime(df_score['レース日']).max()
        except:
            last_date = pd.Timestamp.today()

        cur_horses = horses.copy() if 'horses' in globals() else pd.DataFrame()
        if cur_horses is None or cur_horses.empty:
            st.info("出走表（horses）が見つかりません。")
        else:
            def make_feat_for_horse(hn):
                past = df_score[(df_score['馬名'] == hn) & (pd.to_datetime(df_score['レース日']) < last_date)].sort_values('レース日')
                n_start = len(past)
                wins = int((past['確定着順'] == 1).sum()) if '確定着順' in past else 0
                top3 = int((past['確定着順'] <= 3).sum()) if '確定着順' in past else 0
                win_rate = wins / n_start if n_start>0 else np.nan
                top3_rate = top3 / n_start if n_start>0 else np.nan
                avg_finish = past['確定着順'].dropna().astype(float).mean() if '確定着順' in past and len(past.dropna(subset=['確定着順']))>0 else np.nan
                avg_classpt = past['_class_pt'].mean() if '_class_pt' in past else np.nan
                avg_3fr = past['上3F順位'].dropna().astype(float).mean() if '上3F順位' in past and len(past.dropna(subset=['上3F順位']))>0 else np.nan
                avg_weight = past['斤量'].dropna().astype(float).mean() if '斤量' in past and len(past.dropna(subset=['斤量']))>0 else np.nan
                best_time = past['走破タイム秒'].dropna().astype(float).min() if '走破タイム秒' in past and len(past.dropna(subset=['走破タイム秒']))>0 else np.nan
                if n_start>0:
                    days = (last_date - pd.to_datetime(past['レース日'])).dt.days.fillna(9999).clip(lower=0).values
                    hl_days = 30.4375 * 6.0
                    w = 0.5 ** (days / hl_days)
                    vals = past['score_raw'].fillna(0).astype(float).values if 'score_raw' in past else np.ones(len(past))
                    recency_wavg = float((vals * w).sum() / w.sum()) if w.sum()>0 else np.nan
                    m = recency_wavg
                    recency_wstd = float(np.sqrt(((w * ((vals - m) ** 2)).sum()) / (w.sum()))) if w.sum()>0 else 0.0
                else:
                    recency_wavg, recency_wstd = np.nan, np.nan
                cur_row = cur_horses[cur_horses['馬名'] == hn]
                cur_class_pt = class_point_guess(cur_row.iloc[0].get('クラス名')) if ('クラス名' in cur_row.columns and len(cur_row)>0) else np.nan
                cur_headcount = cur_row.iloc[0].get('頭数') if '頭数' in cur_row.columns and len(cur_row)>0 else np.nan
                cur_上3F = cur_row.iloc[0].get('上3F順位') if '上3F順位' in cur_row.columns and len(cur_row)>0 else np.nan
                cur_通過4 = cur_row.iloc[0].get('通過4角') if '通過4角' in cur_row.columns and len(cur_row)>0 else np.nan
                cur_斤量 = cur_row.iloc[0].get('斤量') if '斤量' in cur_row.columns and len(cur_row)>0 else np.nan
                cur_馬体重 = cur_row.iloc[0].get('馬体重') if '馬体重' in cur_row.columns and len(cur_row)>0 else np.nan
                cur_枠 = cur_row.iloc[0].get('枠') if '枠' in cur_row.columns and len(cur_row)>0 else np.nan
                cur_年齢 = cur_row.iloc[0].get('年齢') if '年齢' in cur_row.columns and len(cur_row)>0 else np.nan

                return {
                    '馬名': hn, 'n_start': n_start, 'win_rate': win_rate, 'top3_rate': top3_rate, 'avg_finish': avg_finish,
                    'avg_classpt': avg_classpt, 'avg_3frank': avg_3fr, 'avg_weight': avg_weight, 'best_time': best_time,
                    'days_since_last': (last_date - pd.to_datetime(past['レース日']).max()).days if n_start>0 else np.nan,
                    'recency_wavg': recency_wavg, 'recency_wstd': recency_wstd,
                    'cur_class_pt': cur_class_pt, 'cur_headcount': cur_headcount, 'cur_上3F順位': cur_上3F,
                    'cur_通過4角': cur_通過4, 'cur_斤量': cur_斤量, 'cur_馬体重': cur_馬体重, 'cur_枠': cur_枠, 'cur_年齢': cur_年齢
                }

            rows = [make_feat_for_horse(hn) for hn in cur_horses['馬名'].tolist()]
            pred_df = pd.DataFrame(rows)

            # prepare X_pred using same use_features and scaler
            X_pred = pred_df[[c for c in use_features if c in pred_df.columns]].copy()
            X_pred = X_pred.fillna(X_pred.median())
            if 'cur_枠' in X_pred.columns:
                X_pred['cur_枠'] = X_pred['cur_枠'].astype(float).astype(int).astype(str)

            # scale numeric columns consistently
            if len(numeric_cols) > 0:
                numeric_present = [c for c in numeric_cols if c in X_pred.columns]
                if len(numeric_present) > 0:
                    X_pred[numeric_present] = scaler.transform(X_pred[numeric_present])

            if LGB_AVAILABLE and use_lgb and isinstance(model, lgb.basic.Booster):
                pred_prob = model.predict(X_pred)
            else:
                try:
                    pred_prob = model.predict_proba(X_pred)[:,1]
                except:
                    pred_prob = model.predict(X_pred)
            pred_df['pred_prob'] = pred_prob
            pred_df = pred_df.sort_values('pred_prob', ascending=False).reset_index(drop=True)
            marks = ['◎','〇','▲','☆','△','△']
            pred_df['印'] = [marks[i] if i < len(marks) else '' for i in range(len(pred_df))]
            st.table(pred_df[['印','馬名','pred_prob','n_start','win_rate','recency_wavg']].head(12).rename(columns={'pred_prob':'勝率推定'}))

        t1 = time.time()
        st.success(f"学習と評価が完了しました（所要 {t1-t0:.1f} 秒）。")
