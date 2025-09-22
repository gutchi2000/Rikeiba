# -*- coding: utf-8 -*-
# 競馬予想アプリ（AUTO統合版 / 2025-09-22）
# - “増やす”より“畳む/自動化”。一括反映パッチ。
# - 主要変更点:
#   A) レース内デフレート(score_adj)を標準化
#   B) 勝率PL温度βを自動チューニング（手動可）
#   C) 右/左回り加点は連続値化 (gap×有効本数ゲート)
#   D) ベストタイム重みは履歴から自己学習(相関)
#   E) 距離×回りの帯域(h)は自動（Silverman風）
#   F) 率ボーナスはベータ縮約のみ（デフォルトOFF）
#   G) 複勝(Top3)は反対称Gumbelで分散低減
#   H) AR100は分位写像（単調変換＝順位不変）
#   I) Isotonic校正のみ（Platt除外）
#   J) 表示はコンパクト化 / 診断は別タブ

import os, io, re, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from itertools import combinations

# ---- optional ----
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

# ===== 日本語フォント =====
from matplotlib import font_manager
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'IPAexGothic','IPAGothic','Noto Sans CJK JP','Yu Gothic UI','Meiryo','Hiragino Sans','MS Gothic'
]

st.set_page_config(page_title="競馬予想アプリ（AUTO版）", layout="wide")

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

jp_font = get_jp_font()
if jp_font is not None:
    try:
        plt.rcParams['font.family'] = jp_font.get_name()
    except Exception:
        pass

# ===== スタイル関係 =====
WAKU_COL = {1:"#ffffff",2:"#000000",3:"#e6002b",4:"#1560bd",5:"#ffd700",6:"#00a04b",7:"#ff7f27",8:"#f19ec2"}

def _style_waku(s: pd.Series):
    out=[]
    for v in s:
        if pd.isna(v):
            out.append("")
        else:
            v=int(v); bg=WAKU_COL.get(v,"#fff"); fg="#000" if v==1 else "#fff"
            out.append(f"background-color:{bg}; color:{fg}; font-weight:700; text-align:center;")
    return out

# ===== 関数群 =====

def season_of(m: int) -> str:
    if 3<=m<=5: return '春'
    if 6<=m<=8: return '夏'
    if 9<=m<=11: return '秋'
    return '冬'

def z_score(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    std = s.std(ddof=0)
    if not np.isfinite(std) or std==0: return pd.Series([50]*len(s), index=s.index)
    return 50 + 10*(s - s.mean())/std

def _parse_time_to_sec(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return np.nan
    s = str(x).strip()
    m = re.match(r'^(\d+):(\d+)\.(\d+)$', s)
    if m: return int(m.group(1))*60 + int(m.group(2)) + float('0.'+m.group(3))
    m = re.match(r'^(\d+)[\.:](\d+)[\.:](\d+)$', s)
    if m: return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/10
    try: return float(s)
    except: return np.nan

def _trim_name(x):
    try: return str(x).replace('\u3000',' ').strip()
    except: return str(x)

# 安定した重み付き標準偏差

def w_std_unbiased(x, w, ddof=1):
    x=np.asarray(x,float); w=np.asarray(w,float)
    sw=w.sum()
    if not np.isfinite(sw) or sw<=0: return np.nan
    m=np.sum(w*x)/sw
    var=np.sum(w*(x-m)**2)/sw
    n_eff=(sw**2)/np.sum(w**2) if np.sum(w**2)>0 else 0
    if ddof and n_eff>ddof: var*= n_eff/(n_eff-ddof)
    return float(np.sqrt(max(var,0.0)))

# NDCG@k（安全）


def ndcg_by_race(frame: pd.DataFrame, scores, k: int=3) -> float:
    f = frame[['race_id','y']].copy().reset_index(drop=True)
    s = np.asarray(scores,float)
    if len(s)!=len(f): s=s[:len(f)]
    vals=[]
    for _, idx in f.groupby('race_id').groups.items():
        idx=np.asarray(list(idx),int)
        y_true=np.nan_to_num(f.loc[idx,'y'].astype(float).to_numpy(), nan=0.0)
        y_pred=np.nan_to_num(s[idx].astype(float), nan=0.0)
        m=len(idx)
        if m==0: continue
        if m==1:
            vals.append(1.0 if y_true[0]>0 else 0.0); continue
        kk=int(min(max(1,k),m))
        order=np.argsort(-y_pred)
        gains=(2.0**y_true[order]-1.0)
        discounts=1.0/np.log2(np.arange(2,m+2))
        dcg=float(np.sum(gains[:kk]*discounts[:kk]))
        order_best=np.argsort(-y_true)
        gains_best=(2.0**y_true[order_best]-1.0)
        idcg=float(np.sum(gains_best[:kk]*discounts[:kk]))
        vals.append(dcg/idcg if idcg>0 else 0.0)
    return float(np.mean(vals)) if vals else float('nan')

# === 安全な等温回帰適用 ===
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

# ===== サイドバー =====
st.sidebar.title("⚙️ パラメタ設定（AUTO統合）")
MODE = st.sidebar.radio("モード", ["AUTO（推奨）","手動（上級者）"], index=0, horizontal=True)

with st.sidebar.expander("🔰 基本", expanded=True):
    lambda_part  = st.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
    grade_bonus  = st.slider("重賞実績ボーナス", 0, 20, 5)
    agari1_bonus = st.slider("上がり3F 1位ボーナス", 0, 10, 3)
    agari2_bonus = st.slider("上がり3F 2位ボーナス", 0, 5, 2)
    agari3_bonus = st.slider("上がり3F 3位ボーナス", 0, 3, 1)

with st.sidebar.expander("本レース条件", expanded=True):
    TARGET_GRADE    = st.selectbox("本レースの格", ["G1","G2","G3","L","OP"], index=4)
    TARGET_SURFACE  = st.selectbox("本レースの馬場", ["芝","ダ"], index=0)
    TARGET_DISTANCE = st.number_input("本レースの距離 [m]", 1000, 3600, 1800, 100)
    TARGET_TURN     = st.radio("回り", ["右","左"], index=0, horizontal=True)

with st.sidebar.expander("🛠 安定化/補正", expanded=True):
    half_life_m  = st.slider("時系列半減期(月)", 0.0, 12.0, 6.0, 0.5)
    stab_weight  = st.slider("安定性(小さいほど◎)の係数", 0.0, 2.0, 0.7, 0.1)
    pace_gain    = st.slider("ペース適性係数", 0.0, 3.0, 1.0, 0.1)
    weight_coeff = st.slider("斤量ペナルティ強度(pts/kg)", 0.0, 4.0, 1.0, 0.1)

with st.sidebar.expander("📏 確率校正", expanded=False):
    do_calib = st.checkbox("等温回帰で勝率を校正", value=False)

with st.sidebar.expander("🎛 手動（上級者向け）", expanded=(MODE=="手動（上級者）")):
    # 手動時のみ使う。AUTOでは内部で決定。
    besttime_w_manual = st.slider("ベストタイム重み(手動)", 0.0, 2.0, 1.0)
    dist_bw_m_manual  = st.slider("距離帯の幅[手動]", 50, 600, 200, 25)
    mc_beta_manual    = st.slider("PL温度β(手動)", 0.3, 5.0, 1.4, 0.1)

with st.sidebar.expander("🖥 表示", expanded=False):
    FULL_TABLE_VIEW = st.checkbox("全頭表示（スクロール無し）", True)
    MAX_TABLE_HEIGHT = st.slider("最大高さ(px)", 800, 10000, 5000, 200)
    SHOW_CORNER = st.checkbox("4角ポジション図を表示", False)

# ===== ファイルアップロード =====
st.title("競馬予想アプリ（AUTO版）")
st.subheader("Excelアップロード（sheet0=過去走 / sheet1=出走表）")
excel_file = st.file_uploader("Excel（.xlsx）", type=['xlsx'], key="excel_up")
if excel_file is None:
    st.info("まずExcelをアップロードしてください。")
    st.stop()

@st.cache_data(show_spinner=False)
def load_excel_bytes(content: bytes):
    xls = pd.ExcelFile(io.BytesIO(content))
    s0 = pd.read_excel(xls, sheet_name=0)
    s1 = pd.read_excel(xls, sheet_name=1)
    return s0, s1

sheet0, sheet1 = load_excel_bytes(excel_file.getvalue())

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
}
REQ_S0 = ['馬名','レース日','競走名','頭数','確定着順']

# 自動で拾い、足りない時だけUI
MAP_S0 = {k: _auto_pick(sheet0, v) for k,v in PAT_S0.items()}
missing = [k for k in REQ_S0 if MAP_S0.get(k) is None]
if missing:
    MAP_S0 = _map_ui(sheet0, PAT_S0, REQ_S0, 'sheet0（過去走）', 's0')

# 取り出し & 前処理
s0 = pd.DataFrame()
for k, col in MAP_S0.items():
    if col and col in sheet0.columns:
        s0[k]=sheet0[col]

s0['レース日']=pd.to_datetime(s0['レース日'], errors='coerce')
for c in ['頭数','確定着順','枠','番','斤量','馬体重','上3F順位','通過4角','距離']:
    if c in s0: s0[c]=pd.to_numeric(s0[c], errors='coerce')
if '走破タイム秒' in s0: s0['走破タイム秒']=s0['走破タイム秒'].apply(_parse_time_to_sec)
if '上がり3Fタイム' in s0: s0['上がり3Fタイム']=s0['上がり3Fタイム'].apply(_parse_time_to_sec)
# === タイム/指標の厳密数値化 ===
for col in ['PCI', 'PCI3', 'Ave-3F']:
    if col in s0.columns:
        s0[col] = pd.to_numeric(s0[col], errors='coerce')

if '距離' in s0.columns:
    s0['距離'] = pd.to_numeric(s0['距離'], errors='coerce')

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
REQ_S1=['馬名','枠','番','性別','年齢']
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

# 脚質エディタ（AUTOで全補完→足りないとこ手直し可）
if '脚質' not in s1.columns: s1['脚質']=''
if '斤量' not in s1.columns: s1['斤量']=np.nan
if '馬体重' not in s1.columns: s1['馬体重']=np.nan

st.subheader("馬一覧（必要なら脚質/斤量/体重を調整）")

# --- 脚質の自動推定（軽量/既存ロジック簡略版） ---

def auto_style_from_history(df: pd.DataFrame, n_recent=5, hl_days=180):
    need={'馬名','レース日','頭数','通過4角'}
    if not need.issubset(df.columns):
        return pd.DataFrame({'馬名':[],'推定脚質':[]})
    t=df[['馬名','レース日','頭数','通過4角','上3F順位']].dropna(subset=['馬名','レース日','頭数','通過4角']).copy()
    t=t.sort_values(['馬名','レース日'], ascending=[True, False])
    t['_rn']=t.groupby('馬名').cumcount()+1
    t=t[t['_rn']<=n_recent].copy()
    today=pd.Timestamp.today()
    t['_days']=(today-pd.to_datetime(t['レース日'], errors='coerce')).dt.days.clip(lower=0).fillna(9999)
    t['_w']=0.5 ** (t['_days']/float(hl_days))
    denom=(pd.to_numeric(t['頭数'], errors='coerce')-1).replace(0,np.nan)
    pos_ratio=(pd.to_numeric(t['通過4角'], errors='coerce')-1)/denom
    pos_ratio=pos_ratio.clip(0,1).fillna(0.5)
    if '上3F順位' in t.columns:
        ag=pd.to_numeric(t['上3F順位'], errors='coerce')
        close=((3.5-ag)/3.5).clip(0,1).fillna(0.0)
    else:
        close=pd.Series(0.0, index=t.index)
    b={'逃げ':-1.2,'先行':0.6,'差し':0.3,'追込':-0.7}
    t['L_逃げ']= b['逃げ'] + 1.6*(1-pos_ratio) - 1.2*close
    t['L_先行']= b['先行']+ 1.1*(1-pos_ratio) - 0.1*close
    t['L_差し']= b['差し']+ 1.1*(pos_ratio)   + 0.9*close
    t['L_追込']= b['追込']+ 1.6*(pos_ratio)   + 0.5*close
    rows=[]
    for name,g in t.groupby('馬名'):
        w=g['_w'].to_numpy(); sw=w.sum()
        if sw<=0: continue
        vec=np.array([
            float((g['L_逃げ']*w).sum()/sw),
            float((g['L_先行']*w).sum()/sw),
            float((g['L_差し']*w).sum()/sw),
            float((g['L_追込']*w).sum()/sw)
        ])
        vec=vec-vec.max(); p=np.exp(vec); p/=p.sum()
        rows.append([name, STYLES[int(np.argmax(p))]])
    return pd.DataFrame(rows, columns=['馬名','推定脚質'])

pred_style = auto_style_from_history(s0.copy())

# マージ（手入力優先。空欄は自動で埋める）
s1['脚質']=s1['脚質'].map(normalize_style)
if not pred_style.empty:
    s1=s1.merge(pred_style, on='馬名', how='left')
    s1['脚質']=s1['脚質'].where(s1['脚質'].astype(str).str.strip().ne(''), s1['推定脚質'])
    s1.drop(columns=['推定脚質'], inplace=True)

# 編集UI（ただし必須入力は撤廃）
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

# ===== 検証軽チェック =====
problems=[]
for c in ['馬名','レース日','競走名','頭数','確定着順']:
    if c not in s0.columns: problems.append(f"sheet0 必須列が不足: {c}")
if '通過4角' in s0.columns and '頭数' in s0.columns:
    tmp=s0[['通過4角','頭数']].dropna()
    if len(tmp)>0 and ((tmp['通過4角']<1)|(tmp['通過4角']>tmp['頭数'])).any():
        problems.append('sheet0 通過4角が頭数レンジ外')
if problems:
    st.warning("入力チェック:\n- "+"\n- ".join(problems))

# ===== マージ（過去走×当日情報） =====
for dup in ['枠','番','性別','年齢','斤量','馬体重','脚質']:
    s0.drop(columns=[dup], errors='ignore', inplace=True)

df = s0.merge(horses[['馬名','枠','番','性別','年齢','斤量','馬体重','脚質']], on='馬名', how='left')

# ===== 1走スコア（従来＋軽量） =====

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

race_date = pd.Timestamp.today()

# 適正馬体重（最良着順時）
best_bw_map={}
if {'馬名','馬体重','確定着順'}.issubset(df.columns):
    _bw=df[['馬名','馬体重','確定着順']].dropna()
    _bw['確定着順']=pd.to_numeric(_bw['確定着順'], errors='coerce')
    _bw=_bw[_bw['確定着順'].notna()]
    try:
        best_idx=_bw.groupby('馬名')['確定着順'].idxmin()
        best_bw_map=_bw.loc[best_idx].set_index('馬名')['馬体重'].astype(float).to_dict()
    except Exception:
        best_bw_map={}

# 1走スコア

def calc_score(r):
    g=class_points(r)
    raw = g*(r['頭数'] + 1 - r['確定着順']) + lambda_part*g
    # 軽ボーナス
    grade_point = grade_bonus if (normalize_grade_text(r.get('クラス名')) or normalize_grade_text(r.get('競走名'))) in ['G1','G2','G3'] else 0
    agari_bonus = 0
    try:
        ao=int(r.get('上3F順位', np.nan))
        if ao==1: agari_bonus=agari1_bonus
        elif ao==2: agari_bonus=agari2_bonus
        elif ao==3: agari_bonus=agari3_bonus
    except: pass
    # 馬体重±10kg
    body_bonus=0
    try:
        name=r['馬名']; now_bw=float(r.get('馬体重', np.nan))
        tekitai=float(best_bw_map.get(name,np.nan))
        if np.isfinite(now_bw) and np.isfinite(tekitai) and abs(now_bw-tekitai)<=10:
            body_bonus=2
    except: pass
    return raw + grade_point + agari_bonus + body_bonus

if 'レース日' not in df.columns:
    st.error('レース日が見つかりません。'); st.stop()

# 一旦正規化
_df = df.copy()
_df['score_raw'] = _df.apply(calc_score, axis=1)
if _df['score_raw'].max()==_df['score_raw'].min():
    _df['score_norm']=50.0
else:
    _df['score_norm'] = (_df['score_raw'] - _df['score_raw'].min()) / (_df['score_raw'].max()-_df['score_raw'].min())*100

# 時系列重み
now = pd.Timestamp.today()
_df['_days_ago']=(now - _df['レース日']).dt.days
_df['_w'] = 0.5 ** (_df['_days_ago'] / (half_life_m*30.4375)) if half_life_m>0 else 1.0

# ===== A) レース内デフレート（標準） =====

def _make_race_id_for_hist(dfh: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dfh['レース日'], errors='coerce').dt.strftime('%Y%m%d').fillna('00000000') + '_' + dfh['競走名'].astype(str).fillna('')

_df['rid_hist'] = _make_race_id_for_hist(_df)
med = _df.groupby('rid_hist')['score_norm'].transform('median')
_df['score_adj'] = _df['score_norm'] - med

# ===== 右/左回り（場名簡易） =====
DEFAULT_VENUE_TURN = {'札幌':'右','函館':'右','福島':'右','新潟':'左','東京':'左','中山':'右','中京':'左','京都':'右','阪神':'右','小倉':'右'}

def infer_turn_row(row):
    name=str(row.get('競走名',''))
    for v,t in DEFAULT_VENUE_TURN.items():
        if v in name: return t
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

    # ===== ここからが安全CVブロック（すべて関数内にインデント） =====
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
        # データが少ない/グループが1種のときは既定本数にフォールバック
        best_param = sorted(param_grid)[len(param_grid)//2]

    # 念のための二重フォールバック
    if best_param is None:
        best_param = sorted(param_grid)[len(param_grid)//2]

    # 学習（分位 q ごと）
    models = {}
    for q in q_list:
        m = GradientBoostingRegressor(loss='quantile', alpha=q,
                                      n_estimators=best_param, max_depth=3,
                                      random_state=random_state)
        m.fit(Xs, y, sample_weight=w)
        models[q] = m

    # 残差分散の推定（メディアン予測基準）
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
                       dist_turn_today_df: pd.DataFrame, feats: list[str]):
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
    for _, r in H.iterrows():
        name = str(r['馬名'])
        x = {}
        x['距離'] = float(target_distance)
        x['斤量'] = float(r.get('斤量', np.nan))
        x['is_dirt'] = 1.0 if str(target_surface).startswith('ダ') else 0.0

        pci_field = _field_pci_from_pace(globals().get('pace_type', 'ミドルペース'))
        if 'PCI' in feats:
            x['PCI'] = float(pci_wmean.get(name, np.nan))
            if not np.isfinite(x['PCI']):
                x['PCI'] = pci_field
        if 'PCI3' in feats:
            x['PCI3'] = float(pci3_wmean.get(name, np.nan))
            if not np.isfinite(x['PCI3']):
                x['PCI3'] = (x.get('PCI', pci_field) + 1.0)
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

df_agg = df_agg.merge(horses[['馬名','枠','番','脚質']], on='馬名', how='left')

# 右/左集計（score_adjの重み平均）
turn_base = _df[['馬名','回り','score_adj','_w']].dropna()
if not turn_base.empty:
    right = (turn_base[turn_base['回り'].astype(str)=='右']
             .groupby('馬名').apply(lambda s: np.average(s['score_adj'], weights=s['_w']) if s['_w'].sum()>0 else np.nan)
             .rename('RightZ'))
    left  = (turn_base[turn_base['回り'].astype(str)=='左']
             .groupby('馬名').apply(lambda s: np.average(s['score_adj'], weights=s['_w']) if s['_w'].sum()>0 else np.nan)
             .rename('LeftZ'))
    counts = (turn_base.pivot_table(index='馬名', columns='回り', values='score_adj', aggfunc='count')
              .rename(columns={'右':'nR','左':'nL'}))
    turn_pref = pd.concat([right,left,counts], axis=1).reset_index()
else:
    turn_pref = pd.DataFrame({'馬名':df_agg['馬名']})

for c in ['RightZ','LeftZ','nR','nL']:
    if c not in turn_pref.columns: turn_pref[c]=np.nan

# C) 連続TurnPrefPts：gap×有効本数ゲート
turn_pref['TurnGap'] = (turn_pref['RightZ'].fillna(0) - turn_pref['LeftZ'].fillna(0))
# 有効本数の近似：nR/nLの合算をKish風に（単純）
turn_pref['n_eff_turn'] = (turn_pref['nR'].fillna(0) + turn_pref['nL'].fillna(0)).clip(lower=0)
conf = np.clip(turn_pref['n_eff_turn'] / 3.0, 0.0, 1.0)
turn_pref['TurnPrefPts'] = np.clip(turn_pref['TurnGap']/1.5, -1.0, 1.0) * conf

df_agg = df_agg.merge(turn_pref[['馬名','RightZ','LeftZ','TurnGap','n_eff_turn','TurnPrefPts']], on='馬名', how='left')

# 距離×回り（自動h）
rows=[]
for nm in df_agg['馬名'].astype(str):
    prof=dist_turn_profile(nm, hist_for_turn, int(TARGET_DISTANCE), str(TARGET_TURN), h_auto, opp_turn_w=0.5)
    rows.append({'馬名':nm, **prof})
_dfturn = pd.DataFrame(rows)
df_agg = df_agg.merge(_dfturn, on='馬名', how='left')

# RecencyZ / StabZ
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

# 最終スコア（未キャリブレーション指標）
turn_gain = 1.0
pace_gain = float(pace_gain)
stab_weight = float(stab_weight)
dist_gain = 1.0


# === 欠損クレンジング（FinalRawの直前で一度だけ） ===
df_agg['RecencyZ']    = pd.to_numeric(df_agg['RecencyZ'], errors='coerce').fillna(df_agg['RecencyZ'].median())
df_agg['StabZ']       = pd.to_numeric(df_agg['StabZ'],    errors='coerce').fillna(df_agg['StabZ'].median())
df_agg['TurnPrefPts'] = pd.to_numeric(df_agg['TurnPrefPts'], errors='coerce').fillna(0.0)
df_agg['DistTurnZ']   = pd.to_numeric(df_agg['DistTurnZ'],   errors='coerce').fillna(0.0)

df_agg['PacePts']=0.0  # 後でMCから
# FinalRaw（基礎：Recency/Stab/Turn/Dist を合算。BT・Paceは後で加点）
df_agg['FinalRaw'] = (
    df_agg['RecencyZ']
    + float(stab_weight) * df_agg['StabZ']
    + 1.0 * df_agg['TurnPrefPts']
    + 1.0 * df_agg['DistTurnZ'].fillna(0.0)
)


# BTを加点（自己学習係数）
if 'ベストタイム秒' in s1.columns:
    btmap = s1.set_index('馬名')['ベストタイム秒'].to_dict()
    btvals = df_agg['馬名'].map(btmap)
    if pd.Series(btvals).notna().any():
        bts = pd.Series(btvals)
        bt_min=bts.min(skipna=True); bt_max=bts.max(skipna=True)
        span=(bt_max-bt_min) if (pd.notna(bt_min) and pd.notna(bt_max) and bt_max>bt_min) else 1.0
        BT_norm = ((bt_max - bts)/span).clip(0,1).fillna(0.0)
        df_agg['FinalRaw'] += w_bt * BT_norm

# ===== ペースMC（反対称Gumbelで分散低減） =====
# 簡易: 脚質からペース指数を作り、印の期待点を加算
# ここでは既存の“期待点マップ”を用いた平均化
mark_rule={
    'ハイペース':      {'逃げ':'△','先行':'△','差し':'◎','追込':'〇'},
    'ミドルペース':    {'逃げ':'〇','先行':'◎','差し':'〇','追込':'△'},
    'ややスローペース': {'逃げ':'〇','先行':'◎','差し':'△','追込':'×'},
    'スローペース':    {'逃げ':'◎','先行':'〇','差し':'△','追込':'×'},
}
mark_to_pts={'◎':2,'〇':1,'○':1,'△':0,'×':-1}

# 脚質確率（今回は確定値→one-hot）
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

# β：AUTO or 手動
beta_pl = tune_beta(_df.copy()) if MODE=="AUTO（推奨）" else float(mc_beta_manual)

# ペースMC（反対称）
rng = np.random.default_rng(24601)
draws = 4000  # 実用十分（必要ならUI化）
Hn=len(name_list)
sum_pts=np.zeros(Hn,float); pace_counter={'ハイペース':0,'ミドルペース':0,'ややスローペース':0,'スローペース':0}
# 逃げ/先行カウント → ペース分類
for _ in range(draws//2):
    sampled = [np.argmax(P[i]) for i in range(Hn)]
    nige  = sum(1 for s in sampled if s==0)
    sengo = sum(1 for s in sampled if s==1)
    epi=(epi_alpha*nige + epi_beta*sengo)/max(1,Hn)
    if   epi>=thr_hi:   pace_t='ハイペース'
    elif epi>=thr_mid:  pace_t='ミドルペース'
    elif epi>=thr_slow: pace_t='ややスローペース'
    else:               pace_t='スローペース'
    pace_counter[pace_t]+=2  # 対称ペア分まとめて
    mk=mark_rule[pace_t]
    for i,s in enumerate(sampled):
        sum_pts[i]+=2*mark_to_pts[ mk[STYLES[s]] ]

df_agg['PacePts']=sum_pts/max(1,draws)
pace_type=max(pace_counter, key=lambda k: pace_counter[k]) if sum(pace_counter.values())>0 else 'ミドルペース'

# === タイム分布 → 着順MC ===
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
        feats=feats
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

    draws_mc = 12000
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

# PacePts反映
# Pace を後乗せ（基礎＋BTを保持したまま）
df_agg['PacePts'] = pd.to_numeric(df_agg['PacePts'], errors='coerce').fillna(0.0)

df_agg['FinalRaw'] += float(pace_gain) * df_agg['PacePts']

# ===== 勝率（PL解析解）＆ Top3（Gumbel反対称） =====
# -- 校正器の学習（履歴から） --
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

# -- PL勝率（NaNセーフ） --
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

# -- Top3近似（Gumbel反対称、NaNセーフ） --
#   abilities は PLと同じ標準化スコアを全頭に作る（欠損は0）
if finite.any():
    A = S.copy()
    A[~finite] = A[finite].mean()
    m = A.mean(); s = A.std() + 1e-9
    abilities = ((A - m) / s).to_numpy(float)
else:
    abilities = np.zeros(len(S), float)

draws_top3 = 8000
rng3 = np.random.default_rng(13579)
G = rng3.gumbel(size=(draws_top3//2, len(abilities)))
U = np.vstack([beta_pl*abilities[None,:] + G, beta_pl*abilities[None,:] - G])
rank_idx = np.argsort(-U, axis=1)
counts = np.zeros(len(abilities), float)
for k in range(3):
    counts += np.bincount(rank_idx[:,k], minlength=len(abilities)).astype(float)
df_agg['複勝率%_PL'] = (100*(counts / draws_top3)).round(2)

# ===== H) AR100: 分位写像（NaN→中央値扱い） =====
ranks = S.rank(method='average', pct=True).fillna(0.5)
qx = np.array([0.00,0.10,0.25,0.50,0.75,0.90,0.97,1.00])
qy = np.array([40 , 45 , 55 , 65 , 72 , 80 , 90 , 98 ])
df_agg['AR100'] = np.interp(ranks.to_numpy(float), qx, qy)

def to_band(v):
    if not np.isfinite(v): return 'E'
    if v>=90: return 'SS'
    if v>=80: return 'S'
    if v>=70: return 'A'
    if v>=60: return 'B'
    if v>=50: return 'C'
    return 'E'

df_agg['Band'] = df_agg['AR100'].map(to_band)

# ===== テーブル整形（日本語ラベル付き） =====
_dfdisp = df_agg.copy().sort_values(['AR100','勝率%_PL'], ascending=[False, False]).reset_index(drop=True)
_dfdisp['順位'] = np.arange(1, len(_dfdisp)+1)

def _fmt_int(x):
    try:
        return '' if pd.isna(x) else f"{int(x)}"
    except:
        return ''

show_cols = [
    '順位','枠','番','馬名','脚質',
    'AR100','Band',
    '勝率%_PL','複勝率%_PL',
    '勝率%_TIME','複勝率%_TIME','期待着順_TIME',
    'PredTime_s','PredTime_p20','PredTime_p80','PredSigma_s',
    'RecencyZ','StabZ','PacePts','TurnPrefPts','DistTurnZ'
]

JP = {
    '順位':'順位','枠':'枠','番':'馬番','馬名':'馬名','脚質':'脚質',
    'AR100':'AR100','Band':'評価帯',
    '勝率%_PL':'勝率%（PL）','複勝率%_PL':'複勝率%（PL）',
    '勝率%_TIME':'勝率%（タイム）','複勝率%_TIME':'複勝率%（タイム）','期待着順_TIME':'期待着順（タイム）',
    'PredTime_s':'予測タイム中央値[s]','PredTime_p20':'20%速い側[s]','PredTime_p80':'80%遅い側[s]','PredSigma_s':'タイム分散σ[s]',
    'RecencyZ':'近走Z','StabZ':'安定性Z','PacePts':'ペースPts','TurnPrefPts':'回り加点','DistTurnZ':'距離×回りZ'
}

_dfdisp_view = _dfdisp[show_cols].rename(columns=JP)

# JPは「英名→日本語名」の辞書。参照は英名キーで統一して値（日本語名）を使う
fmt = {
    JP['AR100']:'{:.1f}',
    JP['勝率%_PL']:'{:.2f}',
    JP['複勝率%_PL']:'{:.2f}',
    JP['RecencyZ']:'{:.2f}',
    JP['StabZ']:'{:.2f}',
    JP['PacePts']:'{:.2f}',
    JP['TurnPrefPts']:'{:.2f}',
    JP['DistTurnZ']:'{:.2f}',
    JP['PredTime_s']:'{:.3f}',
    JP['PredTime_p20']:'{:.3f}',
    JP['PredTime_p80']:'{:.3f}',
    JP['PredSigma_s']:'{:.3f}',
}

# 枠・馬番は整数表示に
num_fmt = {
    JP['枠']: _fmt_int,
    JP['番']: _fmt_int,   # ← リネーム後は「馬番」
}
num_fmt.update(fmt)

styled = (
    _dfdisp_view
      .style
      .apply(_style_waku, subset=[JP['枠']])
      .format(num_fmt, na_rep="")
)


st.markdown("### 本命リスト（AUTO統合）")
st.dataframe(styled, use_container_width=True, height=H(len(_dfdisp_view)))

# 上位抜粋（6頭）
# _dfdisp が未定義でも動くようにフォールバック
if '_dfdisp' in globals():
    base = _dfdisp.rename(columns=JP)   # 英名→日本語に変換
else:
    base = _dfdisp_view                 # 既に日本語列

# head_cols は英名の並びなので、日本語名にマップして抽出
cols_jp = [JP[c] if c in JP else c for c in head_cols]
head_view = base[cols_jp].head(6).copy()

st.markdown("#### 上位抜粋")
st.dataframe(
    head_view.style.format({
        JP['枠']: _fmt_int,
        JP['番']: _fmt_int,
        JP['AR100']:'{:.1f}',
        JP['勝率%_PL']:'{:.2f}',
        JP['勝率%_TIME']:'{:.2f}',
        JP['PacePts']:'{:.2f}',
        JP['PredTime_s']:'{:.3f}',
        JP['PredSigma_s']:'{:.3f}',
    }),
    use_container_width=True, height=H(len(head_view))
)


# 見送り目安
if not (_dfdisp['AR100']>=70).any():
    st.warning('今回のレースは「見送り」：A以上（AR100≥70）が不在。')
else:
    lead = _dfdisp.iloc[0]
    st.info(f"本命候補：**{int(lead['枠'])}-{int(lead['番'])} {lead['馬名']}** / 勝率{lead['勝率%_PL']:.2f}% / AR100 {lead['AR100']:.1f}")


# 4角図（任意）
if SHOW_CORNER:
    try:
        # 既存のレンダラがプロジェクトにある前提。無ければスキップ。
        from matplotlib.patches import Wedge, Rectangle, Circle
        # ダミーの簡易可視化（枠×脚質の散布）
        fig, ax = plt.subplots(figsize=(6,4))
        xs={'逃げ':0.1,'先行':0.3,'差し':0.7,'追込':0.9}
        for _,r in _dfdisp.iterrows():
            x=xs.get(r.get('脚質',''),0.5); y=float(r.get('AR100',50))/100
            ax.scatter(x,y)
            ax.annotate(str(r.get('番','')), (x,y), xytext=(3,3), textcoords='offset points', fontsize=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel('脚質側'); ax.set_ylabel('AR100正規化')
        ax.grid(alpha=.3)
        st.pyplot(fig)
    except Exception as e:
        st.caption(f"4角ポジション図は表示できませんでした：{e}")

# 可視化タブ
with st.expander('📊 散布図（AR100 × PacePts）', expanded=False):
    df_plot=_dfdisp[['馬名','AR100','PacePts','勝率%_PL','脚質','枠','番']].dropna(subset=['AR100','PacePts']).copy()
    if ALT_AVAILABLE and not df_plot.empty:
        ch=(alt.Chart(df_plot).mark_circle().encode(
            x=alt.X('PacePts:Q', title='PacePts'),
            y=alt.Y('AR100:Q', title='AR100'),
            size=alt.Size('勝率%_PL:Q', title='勝率%（PL）', scale=alt.Scale(range=[40, 600])),
            tooltip=['枠','番','馬名','脚質','AR100','勝率%_PL','PacePts']
        ).properties(height=420))
        st.altair_chart(ch, use_container_width=True)
    elif not df_plot.empty:
        fig=plt.figure(figsize=(8,5)); ax=fig.add_subplot(111)
        s=40 + (df_plot['勝率%_PL'].fillna(0).to_numpy())*6
        ax.scatter(df_plot['PacePts'], df_plot['AR100'], s=s, alpha=0.6)
        for _, r in df_plot.iterrows():
            ax.annotate(str(int(r['番'])), (r['PacePts'], r['AR100']), xytext=(2,2), textcoords='offset points', fontsize=8)
        ax.set_xlabel('PacePts'); ax.set_ylabel('AR100'); ax.grid(True, ls='--', alpha=.3)
        st.pyplot(fig)
    else:
        st.info('可視化できるデータがありません。')

# 診断タブ（校正/NDCGなど）
with st.expander('📈 診断（校正・NDCG）', expanded=False):
    # NDCG@3（参考）
    try:
        df_tmp=_df[['レース日','競走名','score_adj','確定着順']].dropna().copy()
        df_tmp['race_id']=pd.to_datetime(df_tmp['レース日'], errors='coerce').dt.strftime('%Y%m%d') + '_' + df_tmp['競走名'].astype(str)
        df_tmp['y']=(pd.to_numeric(df_tmp['確定着順'], errors='coerce')==1).astype(int)
        # proxy: 同一分布上でのsoftmax確率
        pr=[]
        for rid, g in df_tmp.groupby('race_id'):
            s=g['score_adj'].astype(float).to_numpy(); p=np.exp(beta_pl*(s-s.max())); p/=p.sum(); pr.append(p)
        p_raw=np.concatenate(pr) if pr else np.array([])
        ndcg=ndcg_by_race(df_tmp[['race_id','y']], p_raw, k=3)
        st.caption(f"NDCG@3（未校正softmaxの擬似）: {ndcg:.4f}")
    except Exception:
        pass
    if calibrator is None and do_calib:
        st.warning('校正器の学習に必要なデータが不足しています。')
    elif calibrator is not None:
        st.success('等温回帰で勝率を校正中。')

st.markdown("""
<small>
- 本版は **AUTOモード** が標準です。手動は「🎛 手動（上級者向け）」を展開して利用できます。<br>
- **score_adj**（レース内デフレート）を基準として、距離×回り/右左/勝率化を統一しました。<br>
- 勝率は **Plackett–Luce（softmax）**、Top3は **Gumbel反対称サンプリング** で近似しています。<br>
- AR100は **分位写像** でバンドの意味を固定（順位不変）。<br>
</small>
""", unsafe_allow_html=True)
