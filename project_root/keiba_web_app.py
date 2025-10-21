# -*- coding: utf-8 -*-
# 競馬予想アプリ（AUTO統合版 + スペクトル解析）

import os, sys

# 同ディレクトリのモジュールを優先して読めるようにする
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# -- 外部モジュール（ここで宣言しておく） --
from course_geometry import register_all_turf, get_course_geom
from physics_sprint1 import add_phys_s1_features  # ← 先頭でimport

import streamlit as st

@st.cache_resource
def _boot_course_geom():
    register_all_turf()
    return True
_boot_course_geom()

# （必要なら）サンプル実行は無効化して残す
if False:
    geom = get_course_geom(course_id="東京", surface="芝", distance_m=1600, layout="外回り", rail_state="A")
    # course_geometry に追加関数がある環境だけ試す
    try:
        import course_geometry as cg
        if hasattr(cg, "estimate_tci"):
            tci = cg.estimate_tci(geom)
    except Exception:
        pass

# ※ ここで races_df に対して add_phys_s1_features を即時実行しないこと！
#   実行は後半の UI（🧪 PhysS1 スモークテスト）内でのみ行います。



# keiba_web_app.py 冒頭の import 群の直後
import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from course_geometry import register_all_turf, get_course_geom
from physics_sprint1 import add_phys_s1_features

import streamlit as st

@st.cache_resource
def _boot_course_geom():
    register_all_turf()
    return True
_boot_course_geom()
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

st.set_page_config(page_title="Rikeiba", layout="wide")

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

def w_std_unbiased(x, w, ddof=1):
    x=np.asarray(x,float); w=np.asarray(w,float)
    sw=w.sum()
    if not np.isfinite(sw) or sw<=0: return np.nan
    m=np.sum(w*x)/sw
    var=np.sum(w*(x-m)**2)/sw
    n_eff=(sw**2)/np.sum(w**2) if np.sum(w**2)>0 else 0
    if ddof and n_eff>ddof: var*= n_eff/(n_eff-ddof)
    return float(np.sqrt(max(var,0.0)))

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
    grade_opts = ["G1", "G2", "G3", "L", "OP", "3勝クラス"]
    TARGET_GRADE = st.selectbox("本レースの格", grade_opts, index=grade_opts.index("OP"))
    TARGET_SURFACE  = st.selectbox("本レースの馬場", ["芝","ダ"], index=0)
    TARGET_DISTANCE = st.number_input("本レースの距離 [m]", 1000, 3600, 1800, 100)
    TARGET_TURN     = st.radio("回り", ["右","左"], index=0, horizontal=True)
    
with st.sidebar.expander("📐 本レース幾何（コース設定）", expanded=True):
    VENUES = ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]
    COURSE_ID = st.selectbox("競馬場", VENUES, index=VENUES.index("東京"))

    LAYOUT_OPTS = {
        "札幌":["内回り"], "函館":["内回り"], "福島":["内回り"],
        "新潟":["内回り","外回り","直線"], "東京":["外回り"],
        "中山":["内回り","外回り"], "中京":["外回り"],
        "京都":["内回り","外回り"], "阪神":["内回り","外回り"], "小倉":["内回り"]
    }
    LAYOUT = st.selectbox("レイアウト", LAYOUT_OPTS[COURSE_ID])
    RAIL = st.selectbox("コース区分（A/B/C/D）", ["A","B","C","D"], index=0)

    # ← ここで場に連動して既定の回りを出す
    DEFAULT_VENUE_TURN = {'札幌':'右','函館':'右','福島':'右','新潟':'左','東京':'左','中山':'右','中京':'左','京都':'右','阪神':'右','小倉':'右'}
    _turn_default = DEFAULT_VENUE_TURN.get(COURSE_ID, '右')
    TARGET_TURN = st.radio("回り", ["右","左"], index=(0 if _turn_default=="右" else 1), horizontal=True)

    TODAY_BAND = st.select_slider("通過帯域（暫定）", options=["内","中","外"], value="中")


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
    g = get_course_geom(COURSE_ID, "芝" if TARGET_SURFACE=="芝" else "ダ", int(TARGET_DISTANCE), LAYOUT, RAIL)
    st.write("geom:", g)
    races_df_today_dbg = pd.DataFrame([{
        'race_id':'DBG','course_id':COURSE_ID,'surface':'芝',
        'distance_m':int(TARGET_DISTANCE),'layout':LAYOUT,'rail_state':RAIL,
        'band':TODAY_BAND,'num_turns':2
    }])
    try:
        out = add_phys_s1_features(races_df_today_dbg, group_cols=(), band_col="band", verbose=True)
        st.write("phys列:", [c for c in out.columns if c.startswith("phys_")])
        st.dataframe(out)
    except Exception as e:
        st.error(f"PhysS1失敗: {e}")

# ===== ファイルアップロード =====
st.title("Rikeiba")
st.subheader("Excelアップロード（sheet0=過去走 / sheet1=出走表）")
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
    return s0, s1

sheet0, sheet1 = load_excel_bytes(excel_file.getvalue())

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
missing = [k for k in REQ_S0 if MAP_S0.get(k) is None]
if missing:
    MAP_S0 = _map_ui(sheet0, PAT_S0, REQ_S0, 'sheet0（過去走）', 's0')

s0 = pd.DataFrame()
for k, col in MAP_S0.items():
    if col and col in sheet0.columns:
        s0[k]=sheet0[col]

s0['レース日']=pd.to_datetime(s0['レース日'], errors='coerce')
for c in ['頭数','確定着順','枠','番','斤量','馬体重','上3F順位','通過4角','距離']:
    if c in s0: s0[c]=pd.to_numeric(s0[c], errors='coerce')
if '走破タイム秒' in s0: s0['走破タイム秒']=s0['走破タイム秒'].apply(_parse_time_to_sec)
if '上がり3Fタイム' in s0: s0['上がり3Fタイム']=s0['上がり3Fタイム'].apply(_parse_time_to_sec)
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
_df['_days_ago']=(now - _df['レース日']).dt.days
_df['_w'] = 0.5 ** (_df['_days_ago'] / (half_life_m*30.4375 if half_life_m>0 else half_life_m_default*30.4375))

# ===== A) レース内デフレート =====
def _make_race_id_for_hist(dfh: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dfh['レース日'], errors='coerce').dt.strftime('%Y%m%d').fillna('00000000') + '_' + dfh['競走名'].astype(str).fillna('')

_df['rid_hist'] = _make_race_id_for_hist(_df)
med = _df.groupby('rid_hist')['score_norm'].transform('median')
_df['score_adj'] = _df['score_norm'] - med

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

cols_to_merge = ['馬名','枠','番','脚質','性別','年齢']
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

df_agg['PacePts']=0.0  # 後でMCから
# FinalRaw（基礎：Recency/Stab/Turn/Dist + 特性）
df_agg['FinalRaw'] = (
    df_agg['RecencyZ']
    + float(stab_weight) * df_agg['StabZ']
    + 1.0 * df_agg['TurnPrefPts']
    + 1.0 * df_agg['DistTurnZ'].fillna(0.0)
    + df_agg['SexPts'] + df_agg['StylePts'] + df_agg['AgePts'] + df_agg['WakuPts']
)

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
# ★ スペクトル寄与（ユーザーの係数 × 配分）
df_agg['SpecFitZ'] = pd.to_numeric(df_agg['SpecFitZ'], errors='coerce')
df_agg['FinalRaw'] += spec_ratio * float(spectral_weight_ui) * df_agg['SpecFitZ'].fillna(0.0)

# ★ 物理寄与（Z=50を0基準, 10刻みで他Zとスケール合わせ）× 配分
df_agg['PhysicsZ'] = pd.to_numeric(df_agg['PhysicsZ'], errors='coerce')
df_agg['FinalRaw'] += phys_ratio * ((df_agg['PhysicsZ'] - 50.0) / 10.0).fillna(0.0)

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

beta_pl = tune_beta(_df.copy()) if MODE=="AUTO（推奨）" else float(mc_beta_manual)

rng = np.random.default_rng(24601)
draws = 4000
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

# ===== PhysS1（コース幾何）を 予測タイム入り で再計算 → 全馬へ付与 =====
try:
    # レース想定の代表タイム（各馬のPredTime中央値）
    race_pred_time = float(pd.to_numeric(df_agg['PredTime_s'], errors='coerce').median()) \
                     if 'PredTime_s' in df_agg.columns else np.nan

    races_df_today = pd.DataFrame([{
        'race_id': 'TODAY',
        'course_id': COURSE_ID,
        'surface': '芝' if TARGET_SURFACE == '芝' else 'ダ',
        'distance_m': int(TARGET_DISTANCE),
        'layout': LAYOUT,
        'rail_state': RAIL,
        'band': TODAY_BAND,
        'num_turns': 2,
        # ★ ここがポイント：予測タイムの代表値を渡す（無ければ欠損のままOK）
        'final_time_sec': race_pred_time if np.isfinite(race_pred_time) else None,
    }])

    phys1 = add_phys_s1_features(
        races_df_today,
        group_cols=(),      # 1行なのでOK
        band_col="band",
        verbose=False
    )

    phys_cols = {
        'phys_corner_load':'CornerLoadS1',
        'phys_start_cost':'StartCostS1',
        'phys_finish_grade':'FinishGradeS1',
        'phys_s1_score':'PhysS1'
    }
    pv = phys1.rename(columns=phys_cols).iloc[0]
    for k in phys_cols.values():
        df_agg[k] = float(pv[k])

    # スライダーの強さで加点
    df_agg['FinalRaw'] += float(PHYS_S1_GAIN) * df_agg['PhysS1'].fillna(0.0)

except Exception as e:
    st.warning(f"PhysS1の計算に失敗しました: {e}")
    for k in ['CornerLoadS1','StartCostS1','FinishGradeS1','PhysS1']:
        df_agg[k] = np.nan

# PacePts反映
df_agg['PacePts'] = pd.to_numeric(df_agg['PacePts'], errors='coerce').fillna(0.0)
df_agg['FinalRaw'] += float(pace_gain) * df_agg['PacePts']

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

draws_top3 = 8000
rng3 = np.random.default_rng(13579)
G = rng3.gumbel(size=(draws_top3//2, len(abilities)))
U = np.vstack([beta_pl*abilities[None,:] + G, beta_pl*abilities[None,:] - G])
rank_idx = np.argsort(-U, axis=1)
counts = np.zeros(len(abilities), float)
for k in range(3):
    counts += np.bincount(rank_idx[:,k], minlength=len(abilities)).astype(float)
df_agg['複勝率%_PL'] = (100*(counts / draws_top3)).round(2)

# ===== H) AR100: 分位写像 =====
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
# ▼ スペクトル列を追加した完成版
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
    'RecencyZ','StabZ','PacePts','TurnPrefPts','DistTurnZ',
    'SpecFitZ','SpecGate_horse_lbl','SpecGate_templ_lbl',
    'PhysicsZ','PeakWkg','EAP','CornerLoadS1','StartCostS1','FinishGradeS1','PhysS1',
]


JP = {
    '順位':'順位','枠':'枠','番':'馬番','馬名':'馬名','脚質':'脚質',
    'AR100':'AR100','Band':'評価帯',
    '勝率%_PL':'勝率%（PL）','複勝率%_PL':'複勝率%（PL）',
    '勝率%_TIME':'勝率%（タイム）','複勝率%_TIME':'複勝率%（タイム）','期待着順_TIME':'期待着順（タイム）',
    'PredTime_s':'予測タイム中央値[s]','PredTime_p20':'20%速い側[s]','PredTime_p80':'80%遅い側[s]','PredSigma_s':'タイム分散σ[s]',
    'RecencyZ':'近走Z','StabZ':'安定性Z','PacePts':'ペースPts','TurnPrefPts':'回り加点','DistTurnZ':'距離×回りZ',
    'SpecFitZ':'スペクトル適合Z',
    'SpecGate_horse':'走法型(0=持久,1=中庸,2=瞬発)',      # ← 追加
    'SpecGate_templ':'想定レース型(テンプレ)'             # ← 追加
}
JP.update({
    'SpecGate_horse_lbl': '走法型',
    'SpecGate_templ_lbl': '想定レース型(テンプレ)',
    'PhysicsZ':'物理Z',
    'PeakWkg':'ピークW/kg',
    'EAP':'EAP[J/kg/m]'
})


_dfdisp_view = _dfdisp[show_cols].rename(columns=JP)

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
    JP['SpecFitZ']:'{:.2f}',
}

num_fmt = {
    JP['枠']: _fmt_int,
    JP['番']: _fmt_int,
}
fmt.update({
    JP['PhysicsZ']:'{:.2f}',
    JP['PeakWkg']:'{:.2f}',
    JP['EAP']:'{:.3f}',
})
num_fmt.update(fmt)

styled = (
    _dfdisp_view
      .style
      .apply(_style_waku, subset=[JP['枠']])
      .format(num_fmt, na_rep="")
)

st.markdown("### 本命リスト（AUTO統合＋スペクトル）")
st.dataframe(styled, use_container_width=True, height=H(len(_dfdisp_view)))

# 上位抜粋（6頭）
head_cols = ['順位','枠','番','馬名','AR100','Band','勝率%_PL','勝率%_TIME','PredTime_s','PredSigma_s','PacePts','SpecFitZ','PhysicsZ','PeakWkg','EAP','CornerLoadS1','StartCostS1','FinishGradeS1','PhysS1']
base = _dfdisp.rename(columns=JP) if '_dfdisp' in globals() else _dfdisp_view
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
        JP['SpecFitZ']:'{:.2f}',
    }),
    use_container_width=True, height=H(len(head_view))
)

# 見送り目安
if not (_dfdisp['AR100'] >= 70).any():
    st.warning('今回のレースは「見送り」：A以上（AR100≥70）が不在。')
else:
    lead = _dfdisp.iloc[0]

    def _fmt_int_str(x):
        import pandas as pd, numpy as np
        try:
            v = pd.to_numeric(x)
            return "" if pd.isna(v) else f"{int(v)}"
        except Exception:
            return ""

    def _fmt_float(x, n):
        import pandas as pd
        try:
            v = float(x)
            return f"{v:.{n}f}"
        except Exception:
            return "—"

    waku   = _fmt_int_str(lead.get('枠'))
    umaban = _fmt_int_str(lead.get('番'))
    win    = _fmt_float(lead.get('勝率%_PL'), 2)
    ar100  = _fmt_float(lead.get('AR100'), 1)

    # どちらか番号が空ならハイフン省略
    num_part = f"{waku}-{umaban}".strip("-")
    title = f"**{num_part} {lead.get('馬名','')}**" if num_part else f"**{lead.get('馬名','')}**"

    st.info(f"本命候補：{title} / 勝率{win}% / AR100 {ar100}")

# 4角図（任意）
if SHOW_CORNER:
    try:
        from matplotlib.patches import Wedge, Rectangle, Circle
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

# 診断タブ（校正/NDCGなど）
with st.expander('📈 診断（校正・NDCG）', expanded=False):
    try:
        df_tmp=_df[['レース日','競走名','score_adj','確定着順']].dropna().copy()
        df_tmp['race_id']=pd.to_datetime(df_tmp['レース日'], errors='coerce').dt.strftime('%Y%m%d') + '_' + df_tmp['競走名'].astype(str)
        df_tmp['y']=(pd.to_numeric(df_tmp['確定着順'], errors='coerce')==1).astype(int)
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
- **score_adj** を基準に、距離×回り・右左・スペクトル適合を統合し、PL→Top3→AR100 へ連結しました。<br>
- スペクトルは **FFTの帯域判定** と **DTW適合Z** を使用。テンプレは同距離帯・同Surfaceの中央値。<br>
</small>
""", unsafe_allow_html=True)

# ===== ③ 公開用JSON：手入力→AR100採用で書き出し =====
st.markdown("## ③ 公開用JSON（手入力 → AR100得点で書き出し）")

with st.expander("📝 公開メタ入力", expanded=True):
    # ※ 開催日はサイト側で必要になるので、デフォルトで今日を入れておく
    PUB_DATE   = st.date_input("開催日（サイト表示に使用）", value=pd.Timestamp.today().date(), key="pub_date2")
    FILE_NAME  = st.text_input("ファイル名（.json 省略可）", value="rikeiba_picks", key="pub_fname2")
    RACE_NAME  = st.text_input("レース名（例：秋華賞(G1), 富士S(G2)）", key="pub_rname2")
    SURFACE_TX = st.radio("馬場", ["芝", "ダート"], horizontal=True, key="pub_surface2")
    DIST_M     = st.number_input("距離 [m]", min_value=1000, max_value=3600, value=2000, step=100, key="pub_dist2")
    # 任意：馬場状態は空でもOK
    GOING_TX   = st.selectbox("馬場状態（任意）", ["", "良", "稍重", "重", "不良"], index=1, key="pub_going2")
    # 任意：race_id（空でもOK）
    RACE_ID_TX = st.text_input("レースID（任意・空で可）", value="", key="pub_rid2")

# 上位6頭を AR100 で書き出し（◎ 〇 ▲ △ △ △）
MARKS6 = ["◎", "〇", "▲", "△", "△", "△"]

btn = st.button("📤 JSONを書き出す（AR100で得点出力）", use_container_width=True)
if btn:
    import os, re, json
    from datetime import datetime

    # 入力バリデーション
    problems = []
    if not str(RACE_NAME).strip():
        problems.append("レース名が未入力です。")
    if not DIST_M:
        problems.append("距離[m]が未入力です。")
    if problems:
        st.error(" / ".join(problems))
        st.stop()

    # ファイル名整形
    fname = str(FILE_NAME).strip()
    if not fname:
        fname = "rikeiba_picks"
    # 半角・安全なファイル名に寄せる
    fname = re.sub(r"[^\w\-\.\(\)]+", "_", fname)
    if not fname.lower().endswith(".json"):
        fname += ".json"

    # 上位6頭（_dfdisp は上の集計で作ってある想定）
    if '_dfdisp' not in globals() or _dfdisp.empty:
        st.error("出走表の集計が見つかりません（_dfdisp が空）。先にExcelを読み込んでください。")
        st.stop()

    top = _dfdisp[['馬名','AR100']].head(6).copy()
    if top.empty:
        st.error("上位6頭の抽出に失敗（テーブルが空）。")
        st.stop()

    # picks を AR100 採用で作成
    picks = []
    for i in range(len(top)):
        row = top.iloc[i]
        picks.append({
            "horse": str(row['馬名']),
            "mark": MARKS6[i],
            # ← ここがリクエストのポイント：score に AR100 を採用（小数1桁）
            "score": round(float(row['AR100']), 1) if pd.notna(row['AR100']) else None
        })

    # track は「芝 / ダート」のみ（サイト側は 'ダ' を含めばダートと判定できる）
    track_text = "芝" if SURFACE_TX == "芝" else "ダート"

    # 単日フォーマット（サイトは単日/累積どちらも自動対応）
    payload = {
        "date": str(PUB_DATE),           # 例: "2025-10-20"
        "brand": "Rikeiba",
        "races": [{
            "race_id": RACE_ID_TX.strip() or None,
            "race_name": RACE_NAME.strip(),
            "track": track_text,         # 例: "芝" or "ダート"
            "distance": int(DIST_M),
            "going": GOING_TX or "",
            "picks": picks
        }]
    }

    # 保存
    os.makedirs("public_exports", exist_ok=True)
    out_path = os.path.join("public_exports", fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    st.success(f"JSONを書き出しました: {out_path}")
    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
    st.caption("※ そのまま commit & push すれば、Actions → Netlify でサイトに反映されます。")
