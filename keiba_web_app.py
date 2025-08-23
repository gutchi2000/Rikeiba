# keiba_web_app_minified_clean.py
# 縦軸(WStd)修正／重複削除バグ修正／UI復元（購入券種ラジオ）版
import streamlit as st
import pandas as pd
import numpy as np
import re, io, json
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations

# Altair は任意
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

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
st.set_page_config(page_title="競馬予想アプリ（軽量版）", layout="wide")

def _inject_base_css():
    st.markdown("""
    <style>
    #MainMenu, header, footer {visibility:hidden;}
    section[data-testid="stSidebar"] {width: 340px !important;}
    div.block-container {padding-top: .8rem; padding-bottom: 1.5rem; max-width: 1400px;}
    button[role="tab"] {border-radius: 10px !important; padding: .35rem .8rem;}
    .smallcaps{font-variant:all-small-caps; opacity:.9}
    .badge{display:inline-block; padding:.2rem .5rem; border-radius:999px; background:#223; color:#cfe; font-size:.8rem; margin-right:.25rem}
    </style>
    """, unsafe_allow_html=True)
_inject_base_css()

with st.sidebar.expander("🧭 クイックスタート", expanded=True):
    st.markdown("""
1) **Excel**（sheet0=過去走 / sheet1=出走表）をアップロード  
2) 足りない列があれば **列マッピングUI** をON  
3) 左のスライダーを少し調整 → 下に**勝率**・**上位馬**・**買い目**が出ます
""")

STYLES = ['逃げ','先行','差し','追込']
_fwid = str.maketrans('０１２３４５６７８９％','0123456789%')

# ===== util =====
def z_score(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([50]*len(s), index=s.index)
    return 50 + 10 * (s - s.mean()) / std

def season_of(m:int)->str:
    return '春' if 3<=m<=5 else '夏' if 6<=m<=8 else '秋' if 9<=m<=11 else '冬'

def normalize_grade_text(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    s = str(x).translate(_fwid)
    s = (s.replace('Ｇ','G').replace('（','(').replace('）',')')
           .replace('Ⅰ','I').replace('Ⅱ','II').replace('Ⅲ','III'))
    s = re.sub(r'G\s*III','G3',s,flags=re.I); s = re.sub(r'G\s*II','G2',s,flags=re.I); s = re.sub(r'G\s*I','G1',s,flags=re.I)
    s = re.sub(r'ＪＰＮ','Jpn',s,flags=re.I); s = re.sub(r'JPN','Jpn',s,flags=re.I)
    s = re.sub(r'Jpn\s*III','Jpn3',s,flags=re.I); s = re.sub(r'Jpn\s*II','Jpn2',s,flags=re.I); s = re.sub(r'Jpn\s*I','Jpn1',s,flags=re.I)
    m = re.search(r'(?:G|Jpn)\s*([123])',s,flags=re.I)
    return f"G{m.group(1)}" if m else None

@st.cache_data(show_spinner=False)
def load_excel_bytes(content: bytes):
    xls = pd.ExcelFile(io.BytesIO(content))
    return pd.read_excel(xls,0), pd.read_excel(xls,1)

def validate_inputs(df_score, horses):
    problems=[]
    for c in ['馬名','レース日','競走名','頭数','確定着順']:
        if c not in df_score: problems.append(f"sheet0 必須列なし: {c}")
    if '斤量' in horses:
        bad = horses['斤量'].dropna()
        if len(bad)>0 and ((bad<45)|(bad>65)).any(): problems.append("sheet1 斤量がレンジ外（45–65）")
    if {'通過4角','頭数'}.issubset(df_score.columns):
        tmp = df_score[['通過4角','頭数']].dropna()
        if len(tmp)>0 and ((tmp['通過4角']<1)|(tmp['通過4角']>tmp['頭数'])).any(): problems.append("sheet0 通過4角が頭数レンジ外")
    if problems: st.warning("⚠ 入力チェック：\n- "+"\n- ".join(problems))

def _parse_time_to_sec(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return np.nan
    s=str(x).strip()
    m=re.match(r'^(\d+):(\d+)\.(\d+)$',s)
    if m: return int(m.group(1))*60+int(m.group(2))+float('0.'+m.group(3))
    m=re.match(r'^(\d+)[\.:](\d+)[\.:](\d+)$',s)
    if m: return int(m.group(1))*60+int(m.group(2))+int(m.group(3))/10
    try: return float(s)
    except: return np.nan

def _norm_col(s:str)->str:
    s=str(s).strip(); s=re.sub(r'\s+','',s)
    s=s.translate(str.maketrans('０１２３４５６７８９','0123456789')).replace('（','(').replace('）',')').replace('％','%')
    return s

def _auto_guess(col_map,pats):
    for orig,normed in col_map.items():
        for p in pats:
            if re.search(p,normed,flags=re.I): return orig
    return None

def _interactive_map(df, patterns, required_keys, title, state_key, show_ui=False):
    cols=list(df.columns); cmap={c:_norm_col(c) for c in cols}
    auto={k: st.session_state.get(f"{state_key}:{k}") or _auto_guess(cmap,pats) for k,pats in patterns.items()}
    if not show_ui:
        miss=[k for k in required_keys if not auto.get(k)]
        if not miss:
            for k,v in auto.items():
                if v: st.session_state[f"{state_key}:{k}"]=v
            return auto
        else:
            st.warning(f"{title} の必須列が自動認識できません: "+", ".join(miss))
            show_ui=True
    with st.expander(f"列マッピング：{title}", expanded=True):
        mapping={}
        for key,pats in patterns.items():
            default=st.session_state.get(f"{state_key}:{key}") or auto.get(key)
            mapping[key]=st.selectbox(key, ['<未選択>']+cols,
                                      index=(['<未選択>']+cols).index(default) if default in cols else 0,
                                      key=f"map:{state_key}:{key}")
            if mapping[key] != '<未選択>': st.session_state[f"{state_key}:{key}"]=mapping[key]
    miss=[k for k in required_keys if mapping.get(k) in (None,'<未選択>')]
    if miss: st.stop()
    return {k:(None if v=='<未選択>' else v) for k,v in mapping.items()}

# ===== サイドバー =====
st.sidebar.markdown("## ⚙️ パラメタ設定")
tab_basic, tab_detail = st.sidebar.tabs(["🔰 よく使う", "🛠 詳細"])
with tab_basic:
    st.sidebar.header("基本スコア & ボーナス")
    lambda_part  = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
    besttime_w   = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0)
    with st.sidebar.expander("戦績率の重み（当該馬場）", expanded=False):
        win_w  = st.slider("勝率の重み",0.0,5.0,1.0,0.1,key="w_win")
        quin_w = st.slider("連対率の重み",0.0,5.0,0.7,0.1,key="w_quin")
        plc_w  = st.slider("複勝率の重み",0.0,5.0,0.5,0.1,key="w_plc")
    with st.sidebar.expander("各種ボーナス設定", expanded=False):
        grade_bonus  = st.slider("重賞実績ボーナス",0,20,5)
        agari1_bonus = st.slider("上がり3F 1位ボーナス",0,10,3)
        agari2_bonus = st.slider("上がり3F 2位ボーナス",0,5,2)
        agari3_bonus = st.slider("上がり3F 3位ボーナス",0,3,1)
        bw_bonus     = st.slider("馬体重適正ボーナス(±10kg)",0,10,2)
    with st.sidebar.expander("本レース条件（ベストタイム重み用）", expanded=True):
        TARGET_GRADE = st.selectbox("本レースの格", ["G1","G2","G3","L","OP"], index=4, key="target_grade")
        TARGET_SURFACE = st.selectbox("本レースの馬場", ["芝","ダ"], index=0, key="target_surface")
        TARGET_DISTANCE_M = st.number_input("本レースの距離 [m]", 1000, 3600, 1800, 100, key="target_distance_m")
    st.sidebar.markdown("---")
    st.sidebar.header("時系列・安定性・補正")
    half_life_m  = st.sidebar.slider("時系列半減期(月)", 0.0, 12.0, 6.0, 0.5)
    stab_weight  = st.sidebar.slider("安定性(小さいほど◎)の係数", 0.0, 2.0, 0.7, 0.1)
    pace_gain    = st.sidebar.slider("ペース適性係数", 0.0, 3.0, 1.0, 0.1)
    weight_coeff = st.sidebar.slider("斤量ペナルティ強度(pts/kg)", 0.0, 4.0, 1.0, 0.1)
    with st.sidebar.expander("斤量ベース（WFA/JRA簡略）", expanded=False):
        race_date = pd.to_datetime(st.date_input("開催日", value=pd.Timestamp.today().date()))
        use_wfa_base = st.checkbox("WFA基準を使う（推奨）", value=True)
        wfa_2_early_m = st.number_input("2歳（〜9月） 牡/せん [kg]", 50.0, 60.0, 55.0, 0.5)
        wfa_2_early_f = st.number_input("2歳（〜9月） 牝 [kg]"    , 48.0, 60.0, 54.0, 0.5)
        wfa_2_late_m  = st.number_input("2歳（10-12月） 牡/せん [kg]", 50.0, 60.0, 56.0, 0.5)
        wfa_2_late_f  = st.number_input("2歳（10-12月） 牝 [kg]"    , 48.0, 60.0, 55.0, 0.5)
        wfa_3p_m      = st.number_input("3歳以上 牡/せん [kg]" , 50.0, 62.0, 57.0, 0.5)
        wfa_3p_f      = st.number_input("3歳以上 牝 [kg]"     , 48.0, 60.0, 55.0, 0.5)
    st.sidebar.markdown("---")
    st.sidebar.header("資金・点数（購入戦略）")
    total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)
    min_unit     = st.sidebar.selectbox("最小賭け単位", [100,200,300,500], index=0)
    max_lines    = st.sidebar.slider("最大点数(連系)", 1, 60, 20, 1)
    scenario     = st.sidebar.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])
    show_map_ui  = st.sidebar.checkbox("列マッピングUIを表示", value=False)

with tab_detail:
    st.sidebar.header("属性重み（1走スコア係数）")
    gender_w = {g: st.slider(f"{g}", 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
    style_w  = {s: st.slider(f"{s}", 0.0, 2.0, 1.0) for s in STYLES}
    season_w = {s: st.slider(f"{s}", 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
    age_w    = {str(a): st.slider(f"{a}歳", 0.0, 2.0, 1.0, 0.05) for a in range(3,11)}
    frame_w  = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0) for i in range(1,9)}
    st.sidebar.markdown("---")
    st.sidebar.header("ペース / 脚質")
    with st.sidebar.expander("脚質自動推定（強化）", expanded=False):
        auto_style_on   = st.checkbox("自動推定を使う", True)
        AUTO_OVERWRITE  = st.checkbox("手入力より自動を優先", False)
        NRECENT         = st.slider("直近レース数", 1, 10, 5)
        HL_DAYS_STYLE   = st.slider("半減期（日・脚質）", 30, 365, 180, 15)
        pace_mc_draws   = st.slider("ペースMC回数", 500, 30000, 5000, 500)
    with st.sidebar.expander("ペース設定（自動MC / 固定）", expanded=False):
        pace_mode = st.radio("ペースの扱い", ["自動（MC）","固定（手動）"], index=0)
        pace_fixed = st.selectbox("固定ペース", ["ハイペース","ミドルペース","ややスローペース","スローペース"],
                                  1, disabled=(pace_mode=="自動（MC）"))
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
        mc_beta    = st.slider("温度β", 0.1, 5.0, 1.5, 0.1)
        mc_tau     = st.slider("安定度ノイズ τ", 0.0, 2.0, 0.6, 0.05)
        mc_seed    = st.number_input("乱数Seed", 0, 999999, 42, 1)
    with st.sidebar.expander("その他（開発者向け）", expanded=False):
        orig_weight  = st.slider("OrigZ の重み (未使用)", 0.0, 1.0, 0.5, 0.05)

# ===== ファイル =====
st.title("競馬予想アプリ（軽量版・互換性強化）")
st.subheader("ファイルアップロード")
excel_file = st.file_uploader("Excel（sheet0=過去走 / sheet1=出走表）", type=['xlsx'], key="excel_up")
if excel_file is None:
    st.info("まずExcel（.xlsx）をアップロードしてください。"); st.stop()
sheet0, sheet1 = load_excel_bytes(excel_file.getvalue())

# === sheet0 ===
PAT_S0 = {
    '馬名':[r'馬名|名前|出走馬'],'レース日':[r'レース日|日付S|日付|年月日'],
    '競走名':[r'競走名|レース名|名称'],'クラス名':[r'クラス名|格|条件|レースグレード'],
    '頭数':[r'頭数|出走頭数'],'確定着順':[r'確定着順|着順(?!率)'],
    '枠':[r'枠|枠番'],'番':[r'馬番|番'],'斤量':[r'斤量'],'馬体重':[r'馬体重|体重'],
    '上がり3Fタイム':[r'上がり3Fタイム|上がり3F|上3Fタイム|上3F'],'上3F順位':[r'上がり3F順位|上3F順位'],
    '通過4角':[r'通過.*4角|4角.*通過|第4コーナー順位|4角順位'],'性別':[r'性別'],'年齢':[r'年齢|馬齢'],
    '走破タイム秒':[r'走破タイム.*秒|走破タイム|タイム$'],'距離':[r'距離'],'馬場':[r'馬場|馬場状態'],'天候':[r'天候'],
}
REQ_S0 = ['馬名','レース日','競走名','頭数','確定着順']
MAP_S0 = _interactive_map(sheet0, PAT_S0, REQ_S0, "sheet0（過去走）", "s0", show_ui=show_map_ui)

df_score = pd.DataFrame()
for k, col in MAP_S0.items():
    if col is None: continue
    df_score[k] = sheet0[col]

df_score['レース日'] = pd.to_datetime(df_score['レース日'], errors='coerce')
for c in ['頭数','確定着順','枠','番','斤量','馬体重','上3F順位','通過4角','距離']:
    if c in df_score: df_score[c] = pd.to_numeric(df_score[c], errors='coerce')
if '走破タイム秒' in df_score: df_score['走破タイム秒'] = df_score['走破タイム秒'].apply(_parse_time_to_sec)
if '上がり3Fタイム' in df_score: df_score['上がり3Fタイム'] = df_score['上がり3Fタイム'].apply(_parse_time_to_sec)

if '頭数' in df_score:
    df_score['頭数'] = df_score['頭数'].astype(str).str.extract(r'(\d+)')[0].apply(pd.to_numeric, errors='coerce')
if '通過4角' in df_score:
    s = df_score['通過4角']
    if s.dtype.kind not in 'iu':
        last_num = s.astype(str).str.extract(r'(\d+)(?!.*\d)')[0]
        df_score['通過4角'] = pd.to_numeric(last_num, errors='coerce')
    ok = df_score['頭数'].notna() & df_score['通過4角'].notna()
    bad = ok & ((df_score['通過4角'] < 1) | (df_score['通過4角'] > df_score['頭数']))
    df_score.loc[df_score['通過4角'].eq(0), '通過4角'] = np.nan
    df_score.loc[bad, '通過4角'] = np.nan

# === sheet1 ===
PAT_S1 = {
    '馬名':[r'馬名|名前|出走馬'],'枠':[r'枠|枠番'],'番':[r'馬番|番'],'性別':[r'性別'],'年齢':[r'年齢|馬齢'],
    '斤量':[r'斤量'],'馬体重':[r'馬体重|体重'],'脚質':[r'脚質'],
    '勝率':[r'勝率(?!.*率)|\b勝率\b'],'連対率':[r'連対率|連対'],'複勝率':[r'複勝率|複勝'],
    'ベストタイム':[r'ベスト.*タイム|Best.*Time|ﾍﾞｽﾄ.*ﾀｲﾑ|タイム.*(最速|ベスト)'],
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

# 入力UI
if '脚質' not in attrs: attrs['脚質'] = ''
if '斤量' not in attrs: attrs['斤量'] = np.nan
if '馬体重' not in attrs: attrs['馬体重'] = np.nan

st.subheader("馬一覧・脚質・斤量・当日馬体重入力")
edited = st.data_editor(
    attrs[['枠','番','馬名','性別','年齢','脚質','斤量','馬体重']].copy(),
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=STYLES),
        '斤量': st.column_config.NumberColumn('斤量', min_value=45, max_value=65, step=0.5),
        '馬体重': st.column_config.NumberColumn('馬体重', min_value=300, max_value=600, step=1)
    },
    use_container_width=True, num_rows='static'
)
horses = edited.copy()
validate_inputs(df_score, horses)

# --- 脚質自動推定（省略：元コードと同じ） ---
df_style = pd.DataFrame({'馬名': [], 'p_逃げ': [], 'p_先行': [], 'p_差し': [], 'p_追込': [], '推定脚質': []})
need_cols = {'馬名','レース日','頭数','通過4角'}
if auto_style_on and need_cols.issubset(df_score.columns):
    tmp = (df_score[['馬名','レース日','頭数','通過4角','上3F順位']].copy()
            .dropna(subset=['馬名','レース日','頭数','通過4角'])
            .sort_values(['馬名','レース日'], ascending=[True, False]))
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
        close_strength = ((3.5 - ag) / 3.5).clip(0,1).fillna(0.0)
    else:
        close_strength = pd.Series(0.0, index=tmp.index)
    b_nige,b_sengo,b_sashi,b_oikomi = -1.2,0.6,0.3,-0.7
    tmp['L_nige']= b_nige+1.6*(1-pos_ratio)-1.2*close_strength
    tmp['L_sengo']=b_sengo+1.1*(1-pos_ratio)-0.1*close_strength
    tmp['L_sashi']=b_sashi+1.1*(pos_ratio)+0.9*close_strength
    tmp['L_oikomi']=b_oikomi+1.6*(pos_ratio)+0.5*close_strength
    rows=[]
    for name,g in tmp.groupby('馬名'):
        w=g['_w'].to_numpy(); sw=w.sum()
        if sw<=0: continue
        wavg=lambda v: float((v*w).sum()/sw)
        vec=np.array([wavg(g['L_nige']),wavg(g['L_sengo']),wavg(g['L_sashi']),wavg(g['L_oikomi'])],dtype=float)
        vec=vec-vec.max(); p=np.exp(vec); p=p/p.sum(); pred=STYLES[int(np.argmax(p))]
        pr=(pd.to_numeric(g['通過4角'],errors='coerce')-1)/(pd.to_numeric(g['頭数'],errors='coerce')-1)
        pr=pr.clip(0,1).fillna(0.5); wpr=float((pr*w).sum()/sw)
        if pred=='逃げ' and not (wpr<=0.22 or ((pr<=0.15)*w).sum()/sw>=0.25): pred='先行'
        if pred=='追込' and not (wpr>=0.78 or ((pr>=0.85)*w).sum()/sw>=0.25): pred='差し'
        rows.append([name,*p.tolist(),pred])
    if rows:
        df_style=pd.DataFrame(rows,columns=['馬名','p_逃げ','p_先行','p_差し','p_追込','推定脚質'])

# --- 戦績率(%→数値)＆ベストタイム ---
rate_cols=[c for c in ['勝率','連対率','複勝率'] if c in attrs.columns]
if rate_cols:
    rate = attrs[['馬名']+rate_cols].copy()
    for c in rate_cols:
        rate[c]=rate[c].astype(str).str.replace('%','',regex=False).str.replace('％','',regex=False)
        rate[c]=pd.to_numeric(rate[c], errors='coerce')
    mx = pd.concat([rate[c] for c in rate_cols], axis=1).max().max()
    if pd.notna(mx) and mx <= 1.0:
        for c in rate_cols: rate[c]*=100.0
    if 'ベストタイム秒' in attrs: rate=rate.merge(attrs[['馬名','ベストタイム秒']], on='馬名', how='left')
else:
    rate=pd.DataFrame({'馬名':[],'勝率':[],'連対率':[],'複勝率':[],'ベストタイム秒':[]})

# === 重複ガード（←ここがバグ修正ポイント） ===
try:
    if '馬名' in horses.columns:
        horses.drop_duplicates('馬名', keep='first', inplace=True)   # horsesのみ
except Exception: pass
try:
    df_score = df_score.drop_duplicates(subset=['馬名','レース日','競走名'], keep='first')  # 過去走は完全重複のみ除去
except Exception: pass

# ===== マージ =====
for dup in ['枠','番','性別','年齢','斤量','馬体重','脚質']:
    df_score.drop(columns=[dup], errors='ignore', inplace=True)
df_score = df_score.merge(horses[['馬名','枠','番','性別','年齢','斤量','馬体重','脚質']], on='馬名', how='left')
if len(rate)>0:
    use_cols=['馬名']+[c for c in ['勝率','連対率','複勝率','ベストタイム秒'] if c in rate.columns]
    df_score=df_score.merge(rate[use_cols], on='馬名', how='left')

# ===== ベストタイム重み =====
bt_min = df_score['ベストタイム秒'].min(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_max = df_score['ベストタイム秒'].max(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_span = (bt_max-bt_min) if pd.notna(bt_min) and pd.notna(bt_max) and (bt_max>bt_min) else 1.0
CLASS_BASE_BT={"OP":1.50,"L":1.38,"G3":1.19,"G2":1.00,"G1":0.80}
def besttime_weight_final(grade,surface,distance_m,user_scale):
    base=CLASS_BASE_BT.get(str(grade),CLASS_BASE_BT["OP"])
    s=1.10 if str(surface)=="ダ" else 1.00
    try:
        d=int(distance_m)
        if d<=1400: dfac=1.20
        elif d==1600: dfac=1.10
        elif 1800<=d<=2200: dfac=1.00
        elif d>=2400: dfac=0.85
        else: dfac=1.00
    except: dfac=1.00
    return float(np.clip(base*s*dfac*float(user_scale),0.0,2.0))

CLASS_PTS={'G1':10,'G2':8,'G3':6,'リステッド':5,'オープン特別':4}
def class_points(row)->int:
    g=normalize_grade_text(row.get('クラス名')) if 'クラス名' in row else None
    if not g and '競走名' in row: g=normalize_grade_text(row.get('競走名'))
    if g in CLASS_PTS: return CLASS_PTS[g]
    name=str(row.get('クラス名',''))+' '+str(row.get('競走名',''))
    if re.search(r'3\s*勝',name): return 3
    if re.search(r'2\s*勝',name): return 2
    if re.search(r'1\s*勝',name): return 1
    if re.search(r'新馬|未勝利',name): return 1
    if re.search(r'オープン',name): return 4
    if re.search(r'リステッド|L\b',name,flags=re.I): return 5
    return 1

def wfa_base_for(sex:str, age, dt:pd.Timestamp)->float:
    try: a=int(age) if age is not None and not pd.isna(age) else None
    except: a=None
    m=int(dt.month) if isinstance(dt,pd.Timestamp) else 1
    if a==2:
        male = wfa_2_early_m if m<=9 else wfa_2_late_m
        filly= wfa_2_early_f if m<=9 else wfa_2_late_f
    elif a is not None and a>=3:
        male,filly=wfa_3p_m,wfa_3p_f
    else:
        male,filly=wfa_3p_m,wfa_3p_f
    return male if sex in ("牡","セ") else filly

def calc_score(r):
    g = class_points(r)
    raw = g*(r['頭数']+1-r['確定着順']) + lambda_part*g
    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r.get('性別'),1)
    stw = style_w.get(r.get('脚質'),1)
    fw  = frame_w.get(str(r.get('枠')),1)
    aw  = age_w.get(str(r.get('年齢')),1.0)
    gnorm = normalize_grade_text(r.get('クラス名'))
    grade_point = grade_bonus if gnorm in ['G1','G2','G3'] else 0
    agari_bonus=0
    try:
        ao=int(r.get('上3F順位',np.nan))
        if ao==1: agari_bonus=agari1_bonus
        elif ao==2: agari_bonus=agari2_bonus
        elif ao==3: agari_bonus=agari3_bonus
    except: pass
    rate_bonus=0.0
    try:
        if '勝率' in r and pd.notna(r.get('勝率',np.nan)):   rate_bonus+=win_w  * (float(r['勝率'])/100.0)
        if '連対率' in r and pd.notna(r.get('連対率',np.nan)): rate_bonus+=quin_w * (float(r['連対率'])/100.0)
        if '複勝率' in r and pd.notna(r.get('複勝率',np.nan)): rate_bonus+=plc_w  * (float(r['複勝率'])/100.0)
    except: pass
    bt_bonus=0.0
    try:
        if pd.notna(r.get('ベストタイム秒',np.nan)):
            bt_norm=(bt_max-float(r['ベストタイム秒']))/bt_span
            bt_norm=max(0.0,min(1.0,bt_norm))
            bt_bonus = besttime_weight_final(st.session_state.get("target_grade",TARGET_GRADE),
                                             st.session_state.get("target_surface",TARGET_SURFACE),
                                             int(st.session_state.get("target_distance_m",TARGET_DISTANCE_M)),
                                             besttime_w) * bt_norm
    except: pass
    kg_pen=0.0
    try:
        kg=float(r.get('斤量',np.nan))
        if not np.isnan(kg):
            base = wfa_base_for(str(r.get('性別','')), int(r.get('年齢',3)), race_date) if use_wfa_base else 56.0
            delta=kg-float(base)
            kg_pen = (-max(0.0,delta)*float(weight_coeff) + 0.5*max(0.0,-delta)*float(weight_coeff))
    except: pass
    return raw*sw*gw*stw*fw*aw + (grade_point+agari_bonus+rate_bonus+bt_bonus+kg_pen)

if 'レース日' not in df_score:
    st.error("レース日 列が見つかりません。"); st.stop()
df_score['score_raw']=df_score.apply(calc_score, axis=1)
if df_score['score_raw'].max()==df_score['score_raw'].min():
    df_score['score_norm']=50.0
else:
    rng=(df_score['score_raw']-df_score['score_raw'].min())/(df_score['score_raw'].max()-df_score['score_raw'].min())
    df_score['score_norm']=rng*100

# ===== 時系列加重 & 不偏加重標準偏差 =====
now=pd.Timestamp.today()
df_score['_days_ago']=(now-df_score['レース日']).dt.days
df_score['_w']=0.5 ** (df_score['_days_ago']/(half_life_m*30.4375)) if half_life_m>0 else 1.0

def w_mean(x,w):
    w=np.asarray(w,dtype=float); x=np.asarray(x,dtype=float); s=w.sum()
    return float((x*w).sum()/s) if s>0 else np.nan

def w_std(x,w,ddof=1):
    w=np.asarray(w,dtype=float); x=np.asarray(x,dtype=float); s=w.sum()
    if s<=0: return np.nan
    m=(x*w).sum()/s
    num=(w*((x-m)**2)).sum()
    if ddof==0:
        denom=s
    else:
        w2=(w**2).sum()
        denom=s - w2/s  # 不偏（頻度重み）
        if denom<=0: denom=s
    return float(np.sqrt(num/denom))

agg=[]
for name,g in df_score.groupby('馬名'):
    avg=g['score_norm'].mean()
    std=g['score_norm'].std(ddof=0)
    wavg=w_mean(g['score_norm'], g['_w'])
    wstd=w_std(g['score_norm'], g['_w'], ddof=1)
    agg.append({'馬名':name,'AvgZ':avg,'Stdev':std,'WAvgZ':wavg,'WStd':wstd})
df_agg=pd.DataFrame(agg)
for c in ['Stdev','WStd']:
    if c in df_agg: df_agg[c]=df_agg[c].fillna(df_agg[c].median())

df_agg['RecencyZ']=z_score(df_agg['WAvgZ'])
df_agg['StabZ']=z_score(-df_agg['WStd'].fillna(df_agg['WStd'].median()))

# 脚質統合
def _trim_name(x): 
    try: return str(x).replace('\u3000',' ').strip()
    except: return x
for df in [horses,df_agg]:
    if '馬名' in df: df['馬名']=df['馬名'].map(_trim_name)

combined_style=pd.Series(index=df_agg['馬名'], dtype=object)
if '脚質' in horses.columns:
    combined_style.update(horses.set_index('馬名')['脚質'])
if not df_style.empty and auto_style_on:
    pred_series=df_style.set_index('馬名')['推定脚質'].reindex(combined_style.index)
    mask=combined_style.isna()|combined_style.astype(str).str.strip().eq('')
    combined_style.loc[mask]=pred_series.loc[mask]
combined_style=combined_style.fillna('')
df_agg['脚質']=df_agg['馬名'].map(combined_style)

# ペース想定
H=len(df_agg)
P=np.zeros((H,4),dtype=float)
for i,nm in enumerate(df_agg['馬名']):
    stl=combined_style.get(nm,'')
    if stl in STYLES:
        P[i, STYLES.index(stl)]=1.0
    else:
        P[i,:]=0.25
mark_rule={
    'ハイペース':{'逃げ':'△','先行':'△','差し':'◎','追込':'〇'},
    'ミドルペース':{'逃げ':'〇','先行':'◎','差し':'〇','追込':'△'},
    'ややスローペース':{'逃げ':'〇','先行':'◎','差し':'△','追込':'×'},
    'スローペース':{'逃げ':'◎','先行':'〇','差し':'△','追込':'×'},
}
mark_to_pts={'◎':2,'〇':1,'○':1,'△':0,'×':-1}
rng_pace=np.random.default_rng(int(mc_seed)+12345)
sum_pts=np.zeros(H); pace_counter={'ハイペース':0,'ミドルペース':0,'ややスローペース':0,'スローペース':0}
for _ in range(int(pace_mc_draws)):
    sampled=[rng_pace.choice(4, p=P[i]/P[i].sum() if P[i].sum()>0 else np.array([.25,.25,.25,.25])) for i in range(H)]
    nige=sum(1 for s in sampled if s==0); sengo=sum(1 for s in sampled if s==1)
    epi=(epi_alpha*nige + epi_beta*sengo)/max(1,H)
    pace_t="ハイペース" if epi>=thr_hi else "ミドルペース" if epi>=thr_mid else "ややスローペース" if epi>=thr_slow else "スローペース"
    pace_counter[pace_t]+=1; mk=mark_rule[pace_t]
    for i,s in enumerate(sampled): sum_pts[i]+=mark_to_pts[ mk[STYLES[s]] ]
df_agg['PacePts']=sum_pts/max(1,int(pace_mc_draws))
pace_type=max(pace_counter,key=lambda k: pace_counter[k]) if sum(pace_counter.values())>0 else "ミドルペース"
if pace_mode=="固定（手動）":
    pace_type=pace_fixed
    v_pts=np.array([mark_to_pts[mark_rule[pace_type][st]] for st in STYLES])
    df_agg['PacePts']=P@v_pts

df_agg['FinalRaw']=df_agg['RecencyZ'] + stab_weight*df_agg['StabZ'] + pace_gain*df_agg['PacePts']
df_agg['FinalZ']=z_score(df_agg['FinalRaw']) if (df_agg['FinalRaw'].max()-df_agg['FinalRaw'].min())>1e-9 \
                 else 50 + (df_agg['WAvgZ']-df_agg['WAvgZ'].mean())*0.1

# ===== 散布図（縦軸の見え方を安定化） =====
if ALT_AVAILABLE and not df_agg.empty:
    x_min,float_x_max = float(df_agg['FinalZ'].min()), float(df_agg['FinalZ'].max())
    y_min_raw,y_max_raw = float(df_agg['WStd'].min()), float(df_agg['WStd'].max())
    y_lo = max(0.0, y_min_raw - 0.5)
    y_hi = y_max_raw + 0.5
    quad_rect = pd.DataFrame([
        {'x1': x_min, 'x2': 50.0, 'y1': (y_lo+y_hi)/2, 'y2': y_hi},
        {'x1': 50.0, 'x2': float_x_max, 'y1': (y_lo+y_hi)/2, 'y2': y_hi},
        {'x1': x_min, 'x2': 50.0, 'y1': y_lo, 'y2': (y_lo+y_hi)/2},
        {'x1': 50.0, 'x2': float_x_max, 'y1': y_lo, 'y2': (y_lo+y_hi)/2},
    ])
    rect = alt.Chart(quad_rect).mark_rect(opacity=0.07).encode(
        x='x1:Q', x2='x2:Q', y='y1:Q', y2='y2:Q')
    points = alt.Chart(df_agg).mark_circle(size=100).encode(
        x=alt.X('FinalZ:Q', title='最終偏差値',
                scale=alt.Scale(domain=(x_min-1, float_x_max+1), nice=False)),
        y=alt.Y('WStd:Q', title='加重標準偏差（小さいほど安定）',
                scale=alt.Scale(domain=(y_lo, y_hi), nice=True, clamp=True),
                axis=alt.Axis(format='.1f')),
        tooltip=['馬名','WAvgZ','WStd','RecencyZ','StabZ','PacePts','FinalZ']
    )
    labels = alt.Chart(df_agg).mark_text(dx=6, dy=-6, fontSize=10, color='#ffffff').encode(
        x='FinalZ:Q', y='WStd:Q', text='馬名:N'
    )
    vline = alt.Chart(pd.DataFrame({'x':[50.0]})).mark_rule(color='gray').encode(x='x:Q')
    hline = alt.Chart(pd.DataFrame({'y':[(y_lo+y_hi)/2]})).mark_rule(color='gray').encode(y='y:Q')
    quad_text = alt.Chart(pd.DataFrame([
        {'label':'消し・大穴',   'x': (x_min+50)/2,       'y': (y_lo+y_hi*1.0)/2},
        {'label':'波乱・ムラ馬', 'x': (50+float_x_max)/2, 'y': (y_lo+y_hi*1.0)/2},
        {'label':'堅実ヒモ',     'x': (x_min+50)/2,       'y': (y_lo+(y_lo+y_hi)/2)/2},
        {'label':'鉄板・本命',   'x': (50+float_x_max)/2, 'y': (y_lo+(y_lo+y_hi)/2)/2},
    ])).mark_text(fontSize=14, fontWeight='bold', color='#ffffff').encode(x='x:Q', y='y:Q', text='label:N')
    st.altair_chart((rect+points+labels+vline+hline+quad_text).properties(height=420), use_container_width=True)

# ===== 勝率MC（先に計算） =====
S=df_agg['FinalRaw'].to_numpy(float)
S=(S-np.nanmean(S))/(np.nanstd(S)+1e-9)
W=df_agg['WStd'].fillna(df_agg['WStd'].median()).to_numpy(float)
W=(W-W.min())/(W.max()-W.min()+1e-9)
n=len(S); rng=np.random.default_rng(int(mc_seed))
gumbel=rng.gumbel(0.0,1.0,size=(mc_iters,n))
noise=(mc_tau*W)[None,:]*rng.standard_normal((mc_iters,n))
U=mc_beta*S[None,:]+noise+gumbel
rank_idx=np.argsort(-U,axis=1)
win_counts=np.bincount(rank_idx[:,0], minlength=n).astype(float)
top3_counts=np.zeros(n); 
for k in range(3): top3_counts += np.bincount(rank_idx[:,k], minlength=n).astype(float)
df_agg['勝率%_MC']=(win_counts/mc_iters*100).round(2)
df_agg['複勝率%_MC']=(top3_counts/mc_iters*100).round(2)
prob_view=(df_agg[['馬名','FinalZ','WAvgZ','WStd','PacePts','勝率%_MC','複勝率%_MC']]
           .sort_values('勝率%_MC',ascending=False).reset_index(drop=True))

# ===== 上位馬 =====
CUTOFF=50.0
topN=df_agg[df_agg['FinalZ']>=CUTOFF].sort_values('FinalZ',ascending=False).head(6).copy()
topN['印']=['◎','〇','▲','☆','△','△'][:len(topN)]

# ===== 展開表用 =====
def _normalize_ban(x): return pd.to_numeric(str(x).translate(str.maketrans('０１２３４５６７８９','0123456789')), errors='coerce')
df_map=horses.copy(); df_map['脚質']=df_map['馬名'].map(combined_style).fillna(df_map.get('脚質',''))
df_map['番']=_normalize_ban(df_map['番']); df_map=df_map.dropna(subset=['番']).astype({'番':int})
df_map['脚質']=pd.Categorical(df_map['脚質'], categories=STYLES, ordered=True)

# ===== horses2（短評） =====
印map = dict(zip(topN['馬名'], topN['印']))

# df_agg が空でも落ちないように、ある列だけ取る
merge_cols = [c for c in ['馬名','WAvgZ','WStd','FinalZ','脚質','PacePts'] if c in df_agg.columns]
horses2 = horses.merge(df_agg[merge_cols], on='馬名', how='left') if merge_cols else horses.copy()

# 欠損ガード：必要列を必ず作る（KeyError対策）
for col, default in [('印',''), ('脚質',''), ('短評',''), ('WAvgZ', np.nan), ('WStd', np.nan), ('FinalZ', np.nan), ('PacePts', np.nan)]:
    if col not in horses2.columns:
        horses2[col] = default

# 印の付与
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

# 短評（安全に生成）
try:
    horses2['短評'] = horses2.apply(ai_comment, axis=1)
except Exception:
    # 何かあっても落とさない
    if '短評' not in horses2:
        horses2['短評'] = ""


# ===== 資金配分・買い目 =====
st.subheader("■ 資金配分 (厳密合計)")
main_share=0.5
def round_to_unit(x,unit): return int(np.floor(x/unit)*unit)
pur1=round_to_unit(total_budget*main_share*(1/4), int(min_unit))
pur2=round_to_unit(total_budget*main_share*(3/4), int(min_unit))
rem=total_budget-(pur1+pur2)
win_each=round_to_unit(pur1/2, int(min_unit)); place_each=round_to_unit(pur2/2, int(min_unit))
st.write(f"合計予算：{total_budget:,}円  単勝：{pur1:,}円  複勝：{pur2:,}円  残：{rem:,}円  [単位:{min_unit}円]")

h1=topN.iloc[0]['馬名'] if len(topN)>=1 else None
h2=topN.iloc[1]['馬名'] if len(topN)>=2 else None
bets=[]
if h1: bets += [{'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
                {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each}]
if h2: bets += [{'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
                {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each}]

finalZ_map=df_agg.set_index('馬名')['FinalZ'].to_dict()
names=topN['馬名'].tolist(); symbols=topN['印'].tolist()
others=names[1:]; others_sym=symbols[1:]
pair_candidates=[]; tri_candidates=[]; tri1_candidates=[]
if h1 and others:
    for nm,mk in zip(others,others_sym):
        score=finalZ_map.get(nm,0)
        pair_candidates += [('ワイド', f'◎–{mk}', h1, nm, score),
                            ('馬連' , f'◎–{mk}', h1, nm, score),
                            ('馬単' , f'◎→{mk}', h1, nm, score)]
    from itertools import combinations as comb
    for a,b in comb(others,2):
        tri_candidates.append(('三連複','◎-〇▲☆△△', h1, f"{a}／{b}", finalZ_map.get(a,0)+finalZ_map.get(b,0)))
    second_opts=others[:2]
    for s in second_opts:
        for t in others:
            if t==s: continue
            tri1_candidates.append(('三連単フォーメーション','◎-〇▲-〇▲☆△△', h1, f"{s}／{t}", finalZ_map.get(s,0)+0.7*finalZ_map.get(t,0)))

three=['馬連','ワイド','馬単']
if scenario=='通常':
    with st.expander("馬連・ワイド・馬単 から１券種を選択", expanded=True):
        choice=st.radio("購入券種", options=three, index=1, horizontal=True)
        st.write(f"▶ {choice} に残り {rem:,}円 を充当")
    cand=[c for c in pair_candidates if c[0]==choice]
    cand=sorted(cand,key=lambda x:x[-1],reverse=True)[:int(max_lines)]
    K=len(cand)
    if K>0 and rem>=int(min_unit):
        base=round_to_unit(rem/K, int(min_unit)); amts=[base]*K; leftover=rem-base*K; i=0
        while leftover>=int(min_unit) and i<K: amts[i]+=int(min_unit); leftover-=int(min_unit); i+=1
        for (typ,mks,bh,ph,_),amt in zip(cand,amts): bets.append({'券種':typ,'印':mks,'馬':bh,'相手':ph,'金額':int(amt)})
elif scenario=='ちょい余裕':
    cand_wide=sorted([c for c in pair_candidates if c[0]=='ワイド'], key=lambda x:x[-1], reverse=True)
    cand_tri =sorted(tri_candidates, key=lambda x:x[-1], reverse=True)
    cut_w=min(len(cand_wide), int(max_lines)//2 if int(max_lines)>1 else 1)
    cut_t=min(len(cand_tri), int(max_lines)-cut_w)
    allc=cand_wide[:cut_w]+cand_tri[:cut_t]
    K=len(allc)
    if K>0 and rem>=int(min_unit):
        base=round_to_unit(rem/K, int(min_unit)); amts=[base]*K; leftover=rem-base*K; i=0
        while leftover>=int(min_unit) and i<K: amts[i]+=int(min_unit); leftover-=int(min_unit); i+=1
        for (typ,mks,bh,ph,_),amt in zip(allc,amts): bets.append({'券種':typ,'印':mks,'馬':bh,'相手':ph,'金額':int(amt)})
elif scenario=='余裕':
    cand_wide=sorted([c for c in pair_candidates if c[0]=='ワイド'], key=lambda x:x[-1], reverse=True)
    cand_tri =sorted(tri_candidates, key=lambda x:x[-1], reverse=True)
    cand_tri1=sorted(tri1_candidates, key=lambda x:x[-1], reverse=True)
    r_w,r_t,r_t1=2,2,1; denom=r_w+r_t+r_t1
    q_w=max(1,(int(max_lines)*r_w)//denom); q_t=max(1,(int(max_lines)*r_t)//denom); q_t1=max(1,int(max_lines)-q_w-q_t)
    allc=cand_wide[:q_w]+cand_tri[:q_t]+cand_tri1[:q_t1]; K=len(allc)
    if K>0 and rem>=int(min_unit):
        base=round_to_unit(rem/K, int(min_unit)); amts=[base]*K; leftover=rem-base*K; i=0
        while leftover>=int(min_unit) and i<K: amts[i]+=int(min_unit); leftover-=int(min_unit); i+=1
        for (typ,mks,bh,ph,_),amt in zip(allc,amts): bets.append({'券種':typ,'印':mks,'馬':bh,'相手':ph,'金額':int(amt)})

_df=pd.DataFrame(bets)
spent=int(_df['金額'].fillna(0).replace('',0).sum()) if len(_df)>0 else 0
diff=total_budget-spent
if diff!=0 and len(_df)>0:
    for idx in _df.index:
        cur=int(_df.at[idx,'金額']); new=cur+diff
        if new>=0 and new%int(min_unit)==0: _df.at[idx,'金額']=new; break
_df_disp=_df.copy()
if '金額' in _df_disp and len(_df_disp)>0:
    _df_disp['金額']=_df_disp['金額'].map(lambda x: "" if (pd.isna(x) or int(x)<=0) else f"{int(x):,}円")

# ===== タブ =====
tab_dash, tab_prob, tab_pace, tab_bets, tab_all = st.tabs(["🏠 ダッシュボード","📈 勝率","🧭 展開","🎫 買い目","📝 全頭コメント"])

with tab_dash:
    st.subheader("サマリー")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("想定ペース", locals().get("pace_type","—"))
    c2.metric("出走頭数", len(horses))
    if len(topN)>0:
        c3.metric("◎ FinalZ", f"{topN.iloc[0]['FinalZ']:.1f}")
        try:
            win_pct=float(prob_view.loc[prob_view['馬名']==topN.iloc[0]['馬名'],'勝率%_MC'].iloc[0])
            c4.metric("◎ 推定勝率", f"{win_pct:.1f}%")
        except Exception: c4.metric("◎ 推定勝率", "—")
    st.markdown("#### 上位馬（FinalZ≧50・最大6頭）")
    if len(topN)==0:
        st.info("該当なし")
    else:
        cols=[c for c in ['馬名','印','FinalZ','WAvgZ','WStd','PacePts','勝率%_MC'] if c in topN.columns]
        _top_view=topN[cols].copy()
        st.dataframe(_top_view, use_container_width=True, height=240)
        st.download_button("⬇ 上位馬CSV", data=_top_view.to_csv(index=False).encode("utf-8-sig"), file_name="topN.csv", mime="text/csv")

with tab_prob:
    st.subheader("推定勝率・複勝率（モンテカルロ）")
    _pv=prob_view.copy()
    for c in ['勝率%_MC','複勝率%_MC']:
        if c in _pv: _pv[c]=_pv[c].map(lambda x: f"{x:.2f}%")
    st.dataframe(_pv, use_container_width=True, height=380)
    st.download_button("⬇ 勝率テーブルCSV", data=prob_view.to_csv(index=False).encode("utf-8-sig"), file_name="probability_table.csv", mime="text/csv")

with tab_pace:
    st.subheader("展開・脚質サマリー")
    st.caption(f"想定ペース: {locals().get('pace_type','—')}（{'固定' if st.session_state.get('pace_mode')=='固定（手動）' else '自動MC'}）")
    _sc=df_map['脚質'].value_counts().reindex(STYLES).fillna(0).astype(int)
    st.table(pd.DataFrame(_sc, columns=['頭数']).T)

with tab_bets:
    st.subheader("最終買い目一覧")
    if _df_disp.empty:
        st.info("現在、買い目はありません。")
    else:
        show=[c for c in ['券種','印','馬','相手','金額'] if c in _df_disp.columns]
        st.dataframe(_df_disp[show], use_container_width=True, height=320)
        st.download_button("⬇ 買い目CSV", data=_df_disp[show].to_csv(index=False).encode("utf-8-sig"), file_name="bets.csv", mime="text/csv")

with tab_all:
    st.subheader("全頭AI診断コメント")
    q = st.text_input("馬名フィルタ（部分一致）", "")

    # ← 安全な列選択（存在する列だけ）
    show_cols = [c for c in ['馬名','印','脚質','短評','WAvgZ','WStd'] if c in horses2.columns]
    _all = horses2[show_cols].copy()

    if q.strip():
        _all = _all[_all['馬名'].astype(str).str.contains(q.strip(), case=False, na=False)]

    if _all.empty:
        st.info("コメント表示対象がありません。上部の入力と計算結果をご確認ください。")
    else:
        st.dataframe(_all, use_container_width=True, height=420)
        st.download_button("⬇ 全頭コメントCSV",
            data=_all.to_csv(index=False).encode("utf-8-sig"),
            file_name="all_comments.csv", mime="text/csv")
