import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 日本語フォント設定
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(fname='ipaexg.ttf').get_name()

st.set_page_config(layout="wide")
st.title('競馬予想アプリ（完成版）')

# --- サイドバー：重み設定 ---
with st.sidebar:
    st.header("重み設定")
    with st.expander("性別の重み", expanded=False):
        male_w    = st.number_input('牡馬重み',   0.0, 2.0, 1.1, 0.01)
        female_w  = st.number_input('牝馬重み',   0.0, 2.0, 1.0, 0.01)
        gelding_w = st.number_input('せん馬重み', 0.0, 2.0, 0.95,0.01)
    with st.expander("脚質の重み", expanded=False):
        nige_w    = st.number_input('逃げ', 0.0, 2.0, 1.2, 0.01)
        senko_w   = st.number_input('先行', 0.0, 2.0, 1.1, 0.01)
        sashi_w   = st.number_input('差し', 0.0, 2.0, 1.0, 0.01)
        ooka_w    = st.number_input('追込', 0.0, 2.0, 0.9, 0.01)
    with st.expander("四季の重み", expanded=False):
        spring_w  = st.number_input('春', 0.0, 2.0, 1.0, 0.01)
        summer_w  = st.number_input('夏', 0.0, 2.0, 1.1, 0.01)
        autumn_w  = st.number_input('秋', 0.0, 2.0, 1.0, 0.01)
        winter_w  = st.number_input('冬', 0.0, 2.0, 0.95,0.01)
    w_best   = st.number_input('ベストタイム重み', 0.0, 1.0, 1.0, 0.1)
    with st.expander("枠順の重み (1〜8)", expanded=False):
        gate_w = {i: st.number_input(f'{i}枠重み', 0.0, 2.0, 1.0, 0.01) for i in range(1,9)}
    with st.expander("最終スコア重み", expanded=True):
        weight_z    = st.slider('偏差値重み', 0.0, 1.0, 0.7, 0.05)
        weight_rb   = st.slider('実績偏差値重み', 0.0, 1.0, 0.2, 0.05)
        weight_gate = st.slider('枠順偏差値重み', 0.0, 1.0, 0.1, 0.05)

st.write("**設定変更後は『…』→『Clear cache』で再実行してください。**")

# --- データアップロード ---
upload = st.file_uploader('成績＆馬情報 (XLSX)', type='xlsx')
if not upload:
    st.stop()
xls = pd.ExcelFile(upload)

# --- 成績データ読み込み ---
df = xls.parse(0, parse_dates=['レース日'])
df['馬名'] = df['馬名'].astype(str).str.strip()

# --- 馬情報シート読み込み＆整形 ---
stats = xls.parse(1, header=1)
col_map = {}
for c in stats.columns:
    s = str(c)
    if '馬名' in s:      col_map[c] = '馬名'
    if '性別' in s:      col_map[c] = '性別'
    if '年齢' in s:      col_map[c] = '年齢'
    if 'ベストタイム' in s: col_map[c] = 'ベストタイム'
stats = stats.rename(columns=col_map)[['馬名','性別','年齢','ベストタイム']]
stats['馬名'] = stats['馬名'].astype(str)

# 「枠-番-馬名」分割
if stats['馬名'].str.contains('-').any():
    sp = stats['馬名'].str.split('-', 2, expand=True)
    stats['枠']   = pd.to_numeric(sp[0], errors='coerce').fillna(1).astype(int)
    stats['番']   = pd.to_numeric(sp[1], errors='coerce')
    stats['馬名'] = sp[2].str.strip()
else:
    stats['枠'] = 1
    stats['番'] = np.nan

# ベストタイム数値化
stats['best_dist_time'] = pd.to_numeric(
    stats['ベストタイム'].str.extract(r'(\d+)')[0], errors='coerce'
).fillna(
    stats['ベストタイム'].str.extract(r'(\d+)')[0].astype(float).max()
)

# merge

df = df.merge(
    stats[['馬名','性別','年齢','best_dist_time','枠']],
    on='馬名', how='left'
)

# --- デフォルト脚質・斤量 ---
df['脚質'] = df.get('脚質', '差し')
df['today_weight'] = df.get('本斤量', 56)

# --- ファクター関数 ---
def style_f(s): return {'逃げ':nige_w,'先行':senko_w,'差し':sashi_w,'追込':ooka_w}.get(s,1)
def age_f(a):   return 1+0.2*(1-abs(a-5)/5)
def sex_f(sx):  return {'牡':male_w,'牝':female_w,'セ':gelding_w}.get(sx,1)
def sea_f(dt):  m=dt.month; return spring_w if m in [3,4,5] else summer_w if m in [6,7,8] else autumn_w if m in [9,10,11] else winter_w
def gate_f(x): return gate_w.get(x,1)

# --- スコア計算 ---
GRADE = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
grade_score = {k:v**2 for k,v in GRADE.items()}

df['RawBase'] = df.apply(lambda r: grade_score.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['Raw']     = df['RawBase'] * df['枠'].map(gate_f)

# --- 正規化＆Z化 ---
df['Z_Raw']            = (df['Raw'] - df['Raw'].mean()) / df['Raw'].std(ddof=1)
df['Z_best_dist_time'] = (df['best_dist_time'] - df['best_dist_time'].mean()) / df['best_dist_time'].std(ddof=1)
df['Z_枠']             = (df['枠'].map(gate_f) - df['枠'].map(gate_f).mean()) / df['枠'].map(gate_f).std(ddof=1)

# --- 合成スコア ---
df['total_z'] = (
      df['Z_Raw']            * 8
    + df['Z_best_dist_time'] * w_best
    + df['Z_枠']             * weight_gate
) / (8 + w_best + weight_gate)

# --- 馬別集計＆偏差値化 ---
summary = df.groupby('馬名').agg(
    mean_z       = ('total_z','mean'),
    std_z        = ('total_z','std'),
    RawBase_mean = ('RawBase','mean')
).reset_index()
summary['std_z'] = summary['std_z'].fillna(0)

mn, mx = summary['mean_z'].min(), summary['mean_z'].max()
if mx > mn:
    summary['偏差値'] = 30 + (summary['mean_z'] - mn) / (mx - mn) * 40
else:
    summary['偏差値'] = 50

a, b = summary['RawBase_mean'].min(), summary['RawBase_mean'].max()
if b > a:
    summary['実績偏差値'] = 30 + (summary['RawBase_mean'] - a) / (b - a) * 40
else:
    summary['実績偏差値'] = 50

summary['バランス'] = weight_z*summary['偏差値'] + weight_rb*summary['実績偏差値'] - summary['std_z']

# --- 結果表示 ---
st.subheader('本日の予想6頭')
top6 = summary.nlargest(6,'バランス').reset_index(drop=True)
top6['印'] = ['◎','〇','▲','△','△','△']
st.table(top6[['印','馬名','偏差値','実績偏差値','バランス']])
