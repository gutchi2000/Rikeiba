import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations
import altair as alt

st.set_page_config(page_title="競馬予想アプリ", layout="wide")

# ======================== 日本語フォント ========================
try:
    jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
except:
    jp_font = font_manager.FontProperties(fname="C:/Windows/Fonts/meiryo.ttc")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']

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

_fwid = str.maketrans('０１２３','0123')  # 全角数字→半角（一部）

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

# ======================== サイドバー ========================
st.sidebar.header("パラメータ設定")
lambda_part  = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight  = st.sidebar.slider("OrigZ の重み (未使用)", 0.0, 1.0, 0.5, 0.05)

with st.sidebar.expander("性別重み", expanded=False):
    gender_w = {g: st.slider(g, 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
with st.sidebar.expander("脚質重み", expanded=False):
    style_w  = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['逃げ','先行','差し','追込']}
with st.sidebar.expander("四季重み", expanded=False):
    season_w = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
with st.sidebar.expander("年齢重み", expanded=False):
    age_w = {str(age): st.slider(f"{age}歳", 0.0, 2.0, 1.0, 0.05) for age in range(3, 11)}
with st.sidebar.expander("枠順重み", expanded=False):
    frame_w = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0) for i in range(1,9)}

besttime_w   = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0)
with st.sidebar.expander("戦績率の重み（当該馬場）", expanded=False):
    win_w  = st.slider("勝率の重み",   0.0, 5.0, 1.0, 0.1, key="w_win")
    quin_w = st.slider("連対率の重み", 0.0, 5.0, 0.7, 0.1, key="w_quin")
    plc_w  = st.sidebar.slider("複勝率の重み", 0.0, 5.0, 0.5, 0.1, key="w_plc")

with st.sidebar.expander("各種ボーナス設定", expanded=False):
    grade_bonus  = st.slider("重賞実績ボーナス", 0, 20, 5)
    agari1_bonus = st.slider("上がり3F 1位ボーナス", 0, 10, 3)
    agari2_bonus = st.slider("上がり3F 2位ボーナス", 0, 5, 2)
    agari3_bonus = st.slider("上がり3F 3位ボーナス", 0, 3, 1)
    bw_bonus     = st.slider("馬体重適正ボーナス(±10kg)", 0, 10, 2)

# 新規: 時系列加重・安定性・ペース補正・点数制限
st.sidebar.markdown("---")
half_life_m = st.sidebar.slider("時系列半減期(月)", 0.0, 12.0, 6.0, 0.5)
stab_weight = st.sidebar.slider("安定性(小さいほど◎)の係数", 0.0, 2.0, 0.7, 0.1)
pace_gain   = st.sidebar.slider("ペース適性係数", 0.0, 3.0, 1.0, 0.1)
weight_coeff = st.sidebar.slider("斤量ペナルティ強度(pts/kg)", 0.0, 4.0, 1.0, 0.1)

st.sidebar.markdown("---")
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)
min_unit     = st.sidebar.selectbox("最小賭け単位", [100, 200, 300, 500], index=0)
max_lines    = st.sidebar.slider("最大点数(連系)", 1, 60, 20, 1)
scenario     = st.sidebar.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])
show_map_ui = st.sidebar.checkbox("列マッピングUIを表示", value=False)

# --- 勝率モンテカルロ設定（しっかり版） ---
with st.sidebar.expander("勝率シミュレーション（しっかり版）", expanded=False):
    mc_iters   = st.slider("反復回数", 1000, 100000, 20000, 1000)
    mc_beta    = st.slider("強さ→勝率 温度β", 0.1, 5.0, 1.5, 0.1)
    mc_tau     = st.slider("安定度ノイズ係数 τ", 0.0, 2.0, 0.6, 0.05)
    mc_seed    = st.number_input("乱数Seed", 0, 999999, 42, 1)
    st.markdown(
        """
        **使い方の目安**
        - **β（温度）**：大きいほど“実力差を強く反映”。1.2〜2.0で調整。
        - **τ（ノイズ）**：大きいほど“ムラ（WStd）”を確率に反映して荒れやすく。0.4〜0.8が使いやすい。
        - **反復回数**：精度と速度のトレードオフ。2万〜5万で十分安定。
        """
    )

# ======================== ファイルアップロード（ここでゲート） ========================
st.title("競馬予想アプリ（完成系・インタラクティブマッピング版）")
st.subheader("ファイルアップロード")

excel_file = st.file_uploader("Excel（sheet0=過去走 / sheet1=出走表）", type=['xlsx'], key="excel_up")
html_file  = st.file_uploader("HTML（血統）※任意", type=['html'], key="html_up")

if excel_file is None:
    st.info("まずExcel（.xlsx）をアップロードしてください。血統HTMLは無くても動きます。")
    st.stop()

# ---- データ読み込み + インタラクティブ・マッピング ----
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'\s+', '', s)
    # ✅ 正しい全角→半角（0-9）変換
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

    # 自動推定（前回選択があればそれも使う）
    auto = {k: st.session_state.get(f"{state_key}:{k}") or _auto_guess(cmap, pats)
            for k, pats in patterns.items()}

    # UIを出さない場合：必須が埋まっていればそのまま返す
    if not show_ui:
        missing = [k for k in required_keys if not auto.get(k)]
        if not missing:
            for k, v in auto.items():
                if v:
                    st.session_state[f"{state_key}:{k}"] = v
            return auto
        else:
            st.warning(f"{title} の必須列が自動認識できませんでした: " + ", ".join(missing))
            show_ui = True  # 不足があるのでUI表示に切替

    # 必要なときだけUIを表示
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
    if missing:
        st.stop()

    return {k: (None if v=='<未選択>' else v) for k, v in mapping.items()}

# === sheet0（過去走データ） ===
sheet0 = pd.read_excel(excel_file, sheet_name=0)
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
MAP_S0 = _interactive_map(sheet0, PAT_S0, REQ_S0, "sheet0（過去走）", "s0")

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

# === sheet1（当日出走表／プロフィール） ===
sheet1 = pd.read_excel(excel_file, sheet_name=1)
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
MAP_S1 = _interactive_map(sheet1, PAT_S1, REQ_S1, "sheet1（出走表）", "s1")

attrs = pd.DataFrame()
for k, col in MAP_S1.items():
    if col is None: continue
    attrs[k] = sheet1[col]
for c in ['枠','番','斤量','馬体重']:
    if c in attrs: attrs[c] = pd.to_numeric(attrs[c], errors='coerce')
if 'ベストタイム' in attrs: attrs['ベストタイム秒'] = attrs['ベストタイム'].apply(_parse_time_to_sec)

# --- 馬一覧の確認・編集（ここで 'horses' を作る） ---
if '脚質' not in attrs.columns: attrs['脚質'] = ''
if '斤量' not in attrs.columns: attrs['斤量'] = np.nan
if '馬体重' not in attrs.columns: attrs['馬体重'] = np.nan

st.subheader("馬一覧・脚質・斤量・当日馬体重入力")
edited = st.data_editor(
    attrs[['枠','番','馬名','性別','年齢','脚質','斤量','馬体重']],
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込']),
        '斤量': st.column_config.NumberColumn('斤量', min_value=45, max_value=65, step=0.5),
        '馬体重': st.column_config.NumberColumn('馬体重', min_value=300, max_value=600, step=1)
    },
    use_container_width=True,
    num_rows='static'
)
horses = edited.copy()

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

# ---- 血統（HTML）パース：任意（未アップ時は空でOK） ----
if html_file is not None:
    cont = html_file.read().decode(errors='ignore')
    rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont)
    blood = []
    for r in rows:
        c = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)
        if len(c) >= 2:
            blood.append((re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip()))
    blood_df = pd.DataFrame(blood, columns=['馬名','血統'])
else:
    blood_df = pd.DataFrame({'馬名': [], '血統': []})

# ===== [M2] マージ（標準列へ統一） =====
# df_score 側に同名カラムがあれば一旦落としてから attrs を上書き
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

# ===== [M3] ベストタイム正規化レンジ =====
bt_min = df_score['ベストタイム秒'].min(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_max = df_score['ベストタイム秒'].max(skipna=True) if 'ベストタイム秒' in df_score else np.nan
bt_span = (bt_max - bt_min) if pd.notna(bt_min) and pd.notna(bt_max) and (bt_max > bt_min) else 1.0

# ===== スコア算出 (1走ごと) =====
st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

# クラス→ポイント
CLASS_PTS = {
    'G1':10, 'G2':8, 'G3':6, 'リステッド':5, 'オープン特別':4,
}

# 条件戦(クラス表記ゆれ対応)
def class_points(row) -> int:
    # まずグレードを優先
    g = normalize_grade_text(row.get('クラス名')) if 'クラス名' in row else None
    if not g and '競走名' in row:
        g = normalize_grade_text(row.get('競走名'))
    if g in CLASS_PTS: return CLASS_PTS[g]
    # 条件戦
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
        now_bw = r.get('馬体重', np.nan)
        name = r['馬名']
        myhist = df_score[(df_score['馬名']==name) & (df_score['馬体重'].notna())]
        if len(myhist) > 0:
            best_row = myhist.loc[myhist['確定着順'].idxmin()]
            tekitai = best_row['馬体重']
            if not pd.isna(now_bw) and not pd.isna(tekitai) and abs(now_bw - tekitai) <= 10:
                body_bonus = bw_bonus
    except: pass

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
            bt_bonus = besttime_w * bt_norm
    except: pass

    kg_pen = 0.0
    try:
        kg = float(r.get('斤量', np.nan))
        if not np.isnan(kg):
            kg_pen = -max(0.0, kg - 56.0) * weight_coeff
    except: pass

    total_bonus = blood_bonus + grade_point + agari_bonus + body_bonus + rate_bonus + bt_bonus + kg_pen
    return raw * sw * gw * stw * fw * aw + total_bonus

# 1走スコアを計算
if 'レース日' in df_score.columns:
    df_score['レース日'] = pd.to_datetime(df_score['レース日'], errors='coerce')
else:
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

# ===== 時系列加重 (半減期) =====
now = pd.Timestamp.today()
df_score['_days_ago'] = (now - df_score['レース日']).dt.days
if half_life_m > 0:
    df_score['_w'] = 0.5 ** (df_score['_days_ago'] / (half_life_m * 30.4375))
else:
    df_score['_w'] = 1.0

# 重み付き平均＆標準偏差
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

# ===== ペース想定 =====
df_map = horses.copy()
df_map['番'] = pd.to_numeric(df_map['番'].astype(str).str.translate(str.maketrans('０１２３４５６７８９', '0123456789')), errors='coerce')
df_map = df_map.dropna(subset=['番']).astype({'番':int})
df_map['脚質'] = pd.Categorical(df_map['脚質'], categories=['逃げ','先行','差し','追込'], ordered=True)

kakusitsu = ['逃げ','先行','差し','追込']
counter = df_map['脚質'].value_counts().reindex(kakusitsu, fill_value=0)

def pace_and_favor(counter):
    nige = counter['逃げ']; sengo = counter['先行']
    if nige >= 3 or (nige==2 and sengo>=4): return "ハイペース", {'逃げ':'△','先行':'△','差し':'◎','追込':'〇'}
    if nige == 1 and sengo <= 2:           return "スローペース", {'逃げ':'◎','先行':'〇','差し':'△','追込':'×'}
    if nige <= 1:                           return "ややスローペース", {'逃げ':'〇','先行':'◎','差し':'△','追込':'×'}
    return "ミドルペース", {'逃げ':'〇','先行':'◎','差し':'〇','追込':'△'}

pace_type, mark = pace_and_favor(counter)
mark_to_pts = {'◎':2, '〇':1, '○':1, '△':0, '×':-1}

# ===== 最終スコア合成 =====
df_agg['RecencyZ'] = z_score(df_agg['WAvgZ'])
df_agg['StabZ']    = z_score(-df_agg['WStd'].fillna(df_agg['WStd'].median()))
style_map = horses.set_index('馬名')['脚質']
df_agg['脚質'] = df_agg['馬名'].map(style_map)
df_agg['PacePts'] = df_agg['脚質'].map(lambda s: mark_to_pts.get(mark.get(s,'△'),0))
df_agg['FinalRaw'] = df_agg['RecencyZ'] + stab_weight * df_agg['StabZ'] + pace_gain * df_agg['PacePts']
df_agg['FinalZ']   = z_score(df_agg['FinalRaw'])

# ===== 勝率モンテカルロ（Plackett–Luce / Gumbel-max） =====
# 強さベース: FinalRaw を標準化して使用（差が確率に直結しやすい）
S = df_agg['FinalRaw'].to_numpy(dtype=float)
S = (S - np.nanmean(S)) / (np.nanstd(S) + 1e-9)

# 安定度（WStd）が大きいほどブレやすい → ノイズスケールに反映
W = df_agg['WStd'].fillna(df_agg['WStd'].median()).to_numpy(dtype=float)
W = (W - W.min()) / (W.max() - W.min() + 1e-9)  # 0〜1 に正規化

n = len(S)
rng = np.random.default_rng(int(mc_seed))

# Gumbel-max のトリックで一括サンプリング
# U = β*S + τ*W*Normal(0,1) + Gumbel(0,1)
# 行列サイズ: (mc_iters, n)  — 5万×18頭でもメモリはOK
gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(mc_iters, n))
noise  = (mc_tau * W)[None, :] * rng.standard_normal((mc_iters, n))
U = mc_beta * S[None, :] + noise + gumbel

# 各反復の順位（降順）を取得
rank_idx = np.argsort(-U, axis=1)  # shape: (iters, n)

# 勝率/3着内率を集計
win_counts  = np.bincount(rank_idx[:, 0], minlength=n).astype(float)
top3_counts = np.zeros(n, dtype=float)
for k in range(3):
    top3_counts += np.bincount(rank_idx[:, k], minlength=n).astype(float)

p_win  = win_counts  / mc_iters
p_top3 = top3_counts / mc_iters

# df_aggに付与（%表示用に×100）
df_agg['勝率%_MC']   = (p_win  * 100).round(2)
df_agg['複勝率%_MC'] = (p_top3 * 100).round(2)

# ===== 可視化（４象限＋白文字ラベル） =====
avg_st = float(df_agg['WStd'].mean())
x_mid = 50.0
y_mid = avg_st

x_min, x_max = float(df_agg['FinalZ'].min()), float(df_agg['FinalZ'].max())
y_min, y_max = float(df_agg['WStd'].min()),  float(df_agg['WStd'].max())

# 薄い背景で４象限をハイライト
quad_rect = pd.DataFrame([
    {'x1': x_min, 'x2': x_mid, 'y1': y_mid, 'y2': y_max},  # 左上
    {'x1': x_mid, 'x2': x_max, 'y1': y_mid, 'y2': y_max},  # 右上
    {'x1': x_min, 'x2': x_mid, 'y1': y_min, 'y2': y_mid},  # 左下
    {'x1': x_mid, 'x2': x_max, 'y1': y_min, 'y2': y_mid},  # 右下
])
rect = alt.Chart(quad_rect).mark_rect(opacity=0.07).encode(
    x='x1:Q', x2='x2:Q', y='y1:Q', y2='y2:Q'
)

# 散布点
points = alt.Chart(df_agg).mark_circle(size=100).encode(
    x=alt.X('FinalZ:Q', title='最終偏差値'),
    y=alt.Y('WStd:Q',  title='加重標準偏差（小さいほど安定）'),
    tooltip=['馬名','WAvgZ','WStd','RecencyZ','StabZ','PacePts','勝率%_MC','複勝率%_MC']
)

# 馬名ラベル（白文字＋黒縁で視認性UP）
labels = alt.Chart(df_agg).mark_text(
    dx=6, dy=-6, fontSize=10, color='white',
    stroke='black', strokeWidth=2
).encode(
    x='FinalZ:Q', y='WStd:Q', text='馬名:N'
)

# 基準線
vline = alt.Chart(pd.DataFrame({'x':[x_mid]})).mark_rule(color='gray').encode(x='x:Q')
hline = alt.Chart(pd.DataFrame({'y':[y_mid]})).mark_rule(color='gray').encode(y='y:Q')

# ４象限ラベル（白文字＋黒縁）
quad_text = alt.Chart(pd.DataFrame([
    {'label':'消し・大穴',   'x': (x_min + x_mid)/2, 'y': (y_mid + y_max)/2},  # 左上
    {'label':'波乱・ムラ馬', 'x': (x_mid + x_max)/2, 'y': (y_mid + y_max)/2},  # 右上
    {'label':'堅実ヒモ',     'x': (x_min + x_mid)/2, 'y': (y_min + y_mid)/2},  # 左下
    {'label':'鉄板・本命',   'x': (x_mid + x_max)/2, 'y': (y_min + y_mid)/2},  # 右下
])).mark_text(
    fontSize=14, fontWeight='bold', color='white',
    stroke='black', strokeWidth=3
).encode(x='x:Q', y='y:Q', text='label:N')

chart = (rect + points + labels + vline + hline + quad_text).properties(
    width=700, height=420
).interactive()

st.altair_chart(chart, use_container_width=True)

# ===== 上位馬抽出（最終Z基準） =====
topN = df_agg.sort_values('FinalZ', ascending=False).head(6).copy()
if len(topN) == 0:
    st.warning("上位馬が抽出できませんでした。入力を確認してください。")
topN['印'] = ['◎','〇','▲','☆','△','△'][:len(topN)]

st.subheader("上位馬（最終偏差値ベース／根拠付き）")
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

st.subheader("■ 推定勝率・複勝率（モンテカルロ）")
prob_view = (
    df_agg[['馬名','FinalZ','WAvgZ','WStd','PacePts','勝率%_MC','複勝率%_MC']]
    .sort_values('勝率%_MC', ascending=False)
    .reset_index(drop=True)
)
st.table(prob_view)

# ===== 展開ロケーション（視覚） =====
df_map_show = df_map.sort_values(['番'])
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

# ===== 重賞好走ハイライト（上がり3F付き） =====
race_col = next((c for c in ['レース名','競走名','レース','名称'] if c in df_score.columns), None)
ag_col   = next((c for c in ['上がり3Fタイム','上がり3F','上がり３Ｆ','上3Fタイム','上3F'] if c in df_score.columns), None)
finish_col = '確定着順'

def grade_from_row(row):
    g = normalize_grade_text(row.get('クラス名')) if 'クラス名' in row else None
    if not g and race_col:
        g = normalize_grade_text(row.get(race_col))
    return g

dfg = df_score.copy()
dfg['GradeN']   = dfg.apply(grade_from_row, axis=1)
dfg['着順num']  = pd.to_numeric(dfg[finish_col], errors='coerce')
dfg['_date']    = pd.to_datetime(dfg['レース日'], errors='coerce')
dfg['_date_str']= dfg['_date'].dt.strftime('%Y.%m.%d').fillna('日付不明')

def _clean_one_line(v):
    if pd.isna(v): return ''
    return str(v).replace('\r','').replace('\n','').strip()
if race_col: dfg[race_col] = dfg[race_col].map(_clean_one_line)
if ag_col:   dfg[ag_col]   = dfg[ag_col].map(_clean_one_line)

thr_map = {'G1':5, 'G2':4, 'G3':3}
dfg = dfg[dfg['GradeN'].isin(thr_map.keys()) & dfg['着順num'].notna()].copy()
dfg = dfg[dfg.apply(lambda r: r['着順num'] <= thr_map[r['GradeN']], axis=1)]
dfg = dfg.sort_values(['馬名','_date'], ascending=[True, False])

def _fmt_ag(v):
    if v in [None, '', 'nan']: return ''
    s = str(v).replace('秒','').strip()
    try:
        return f"{float(s):.1f}"
    except:
        return s

def make_table(d: pd.DataFrame) -> pd.DataFrame:
    n = len(d)
    races = d[race_col].fillna('（不明）') if race_col else pd.Series(['（不明）']*n, index=d.index)
    ags   = d[ag_col].map(_fmt_ag) if ag_col else pd.Series(['']*n, index=d.index)
    out = pd.DataFrame({
        'レース名': races,
        '格': d['GradeN'].values,
        '着': d['着順num'].astype(int).values,
        '日付': d['_date_str'].values,
        '上がり3F': ags.values,
    })
    return out

grade_tables = {name: make_table(d) for name, d in dfg.groupby('馬名')}

st.subheader("■ 重賞好走ハイライト（上がり3F付き）")
for name in topN['馬名'].tolist():
    st.markdown(f"**{name}**")
    t = grade_tables.get(name, None)
    if t is None or t.empty:
        st.write("　重賞経験なし")
    else:
        st.table(t)

# ===== horses 情報付与（短評） =====
印map = dict(zip(topN['馬名'], topN['印']))
horses2 = horses.merge(df_agg[['馬名','WAvgZ','WStd','FinalZ','脚質','PacePts']], on='馬名', how='left')
horses2['印'] = horses2['馬名'].map(印map).fillna('')
horses2 = horses2.merge(blood_df, on='馬名', how='left', suffixes=('', '_血統'))

def ai_comment(row):
    base = ""
    if row['印'] == '◎':
        base += "本命評価。"
        base += ("高い安定感で信頼度抜群。" if row['WStd'] <= 8 else "能力上位もムラあり。")
    elif row['印'] == '〇':
        base += "対抗評価。"
        base += ("近走安定しており軸候補。" if row['WStd'] <= 10 else "展開ひとつで逆転も。")
    elif row['印'] in ['▲','☆']:
        base += "上位グループの一角。"
        base += ("ムラがあり一発タイプ。" if row['WStd'] > 15 else "安定型で堅実。")
    elif row['印'] == '△':
        base += "押さえ候補。"
        base += ("堅実だが勝ち切るまでは？" if row['WStd'] < 12 else "展開次第で浮上も。")
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

# ======================== 買い目生成＆厳密配分 ========================
h1 = topN.iloc[0]['馬名'] if len(topN) >= 1 else None
h2 = topN.iloc[1]['馬名'] if len(topN) >= 2 else None

symbols = topN['印'].tolist()
names   = topN['馬名'].tolist()
others_names   = names[1:] if len(names) > 1 else []
others_symbols = symbols[1:] if len(symbols) > 1 else []

three = ['馬連','ワイド','馬単']

def round_to_unit(x, unit):
    return int(np.floor(x / unit) * unit)

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

    from itertools import combinations
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

# 合計チェック & 最終表示用 DataFrame
_df = pd.DataFrame(bets)
spent = int(_df['金額'].fillna(0).replace('',0).sum())
diff = total_budget - spent
if diff != 0 and len(_df) > 0:
    for idx in _df.index:
        cur = int(_df.at[idx,'金額'])
        new = cur + diff
        if new >= 0 and new % min_unit == 0:
            _df.at[idx,'金額'] = new
            break

_df['金額'] = _df['金額'].map(lambda x: f"{int(x):,}円" if pd.notna(x) and int(x)>0 else "")

unique_types = _df['券種'].unique().tolist() if len(_df)>0 else []
tabs = st.tabs(['サマリー'] + unique_types) if len(unique_types)>0 else st.tabs(['サマリー'])
with tabs[0]:
    st.subheader("■ 最終買い目一覧（全券種まとめ）")
    if len(_df)==0: st.info("現在、買い目はありません。")
    else:
        st.table(_df[['券種','印','馬','相手','金額']])

for i, typ in enumerate(unique_types, start=1):
    with tabs[i]:
        df_this = _df[_df['券種'] == typ]
        st.subheader(f"{typ} 買い目一覧")
        st.table(df_this[['券種','印','馬','相手','金額']]) if len(df_this)>0 else st.info(f"{typ} の買い目はありません。")
