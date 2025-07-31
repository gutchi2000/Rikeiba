import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations, permutations

# ── 日本語フォント設定 ──
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(
    fname='ipaexg.ttf'
).get_name()

st.title('競馬予想アプリ（完成版）')

# ── サイドバー: 重み設定 ──
male_w    = st.sidebar.number_input('牡の重み',   0.0, 2.0, 1.1, 0.01)
female_w  = st.sidebar.number_input('牝の重み',   0.0, 2.0, 1.0, 0.01)
gelding_w = st.sidebar.number_input('セの重み',   0.0, 2.0, 0.95,0.01)
nige_w    = st.sidebar.number_input('逃げの重み',  0.0, 2.0, 1.2, 0.01)
senko_w   = st.sidebar.number_input('先行の重み',  0.0, 2.0, 1.1, 0.01)
sashi_w   = st.sidebar.number_input('差しの重み',  0.0, 2.0, 1.0, 0.01)
ooka_w    = st.sidebar.number_input('追込の重み',  0.0, 2.0, 0.9, 0.01)
spring_w  = st.sidebar.number_input('春の重み',    0.0, 2.0, 1.0, 0.01)
summer_w  = st.sidebar.number_input('夏の重み',    0.0, 2.0, 1.1, 0.01)
autumn_w  = st.sidebar.number_input('秋の重み',    0.0, 2.0, 1.0, 0.01)
winter_w  = st.sidebar.number_input('冬の重み',    0.0, 2.0, 0.95,0.01)
w_jin     = st.sidebar.number_input('斤量重み',       0.0, 1.0, 1.0, 0.1)
w_best    = st.sidebar.number_input('距離ベスト重み', 0.0, 1.0, 1.0, 0.1)

# ── データ読み込み ──
uploaded = st.file_uploader('成績＆馬情報(XLSX)', type='xlsx')
if not uploaded:
    st.stop()
xls = pd.ExcelFile(uploaded)

# 成績シート読み込み
df = xls.parse(0, parse_dates=['レース日'])
df['馬名'] = df['馬名'].astype(str).str.strip()

# 馬情報シート読み込み＋動的リネーム
stats = xls.parse(1, header=1)
keys = ['馬名','性別','年齢','ベストタイム']
col_map = {}
for key in keys:
    for col in stats.columns:
        if key in str(col):
            col_map[col] = key
            break
stats = stats.rename(columns=col_map)

# 存在しない列は NaN で追加
for key in keys:
    if key not in stats.columns:
        stats[key] = np.nan

stats = stats[keys].drop_duplicates('馬名')
stats['馬名'] = stats['馬名'].astype(str).str.strip()

# ベストタイム→数値化
stats['best_dist_time'] = pd.to_numeric(
    stats['ベストタイム'].replace({'(未)': np.nan}), errors='coerce'
)
stats['best_dist_time'].fillna(stats['best_dist_time'].max(), inplace=True)

# 成績データとマージ
df = df.merge(
    stats[['馬名','性別','年齢','best_dist_time']],
    on='馬名', how='left'
)

# 脚質＆本斤量入力
equines = df['馬名'].unique()
inp = pd.DataFrame({
    '馬名': equines,
    '脚質': ['差し']*len(equines),
    '本斤量': [56]*len(equines)
})
edited = st.data_editor(
    inp,
    column_config={
        '脚質': st.column_config.SelectboxColumn(label='脚質', options=['逃げ','先行','差し','追込']),
        '本斤量': st.column_config.NumberColumn(label='本斤量', min_value=45, max_value=60, step=1)
    },
    use_container_width=True
)
edited['馬名'] = edited['馬名'].astype(str).str.strip()
df = df.merge(edited, on='馬名', how='left').rename(columns={'本斤量':'today_weight'})

# 血統表＆強調種牡馬
html = st.file_uploader('血統表(HTML)', type='html')
ped = None
if html:
    try:
        ped = pd.read_html(html.read())[0].set_index('馬名')
    except:
        ped = None
priority = [s.strip() for s in st.text_area('強調種牡馬(カンマ区切り)').split(',') if s.strip()]

# ファクター関数
def ped_factor(r):
    if ped is None or r['馬名'] not in ped.index:
        return 1.0
    return 1.2 if ped.at[r['馬名'],'父馬'] in priority else 1.0

def style_f(s): return {'逃げ':nige_w,'先行':senko_w,'差し':sashi_w,'追込':ooka_w}.get(s,1.0)
def age_f(a):   return 1+0.2*(1-abs(a-5)/5)
def sex_f(sx):  return {'牡':male_w,'牝':female_w,'セ':gelding_w}.get(sx,1.0)
def sea_f(dt):
    m=dt.month
    return spring_w if m in [3,4,5] else summer_w if m in [6,7,8] else autumn_w if m in [9,10,11] else winter_w

# 正規化＆Zスコア化
tmin, tmax = df['best_dist_time'].min(), df['best_dist_time'].max()
df['dist_n'] = (tmax - df['best_dist_time'])/(tmax - tmin)
mu, sd      = df['dist_n'].mean(), df['dist_n'].std(ddof=1)
df['Z_dist_n'] = (df['dist_n'] - mu)/sd if sd else 0

df['ped_f']   = df.apply(ped_factor, axis=1)
df['style_f'] = df['脚質'].map(style_f)
df['age_f']   = df['年齢'].map(age_f)
df['sex_f']   = df['性別'].map(sex_f)
df['sea_f']   = df['レース日'].map(sea_f)

GRADE = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
df['raw'] = df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['raw'] *= df['ped_f'] * df['style_f'] * df['age_f'] * df['sex_f'] * df['sea_f']

jmax, jmin = df['斤量'].max(), df['斤量'].min()
df['up3_n']   = df['Ave-3F']/df['上がり3Fタイム']
df['odds_n']  = 1/(1+np.log10(df['単勝オッズ']))
df['jin_n']   = (jmax - df['斤量'])/(jmax - jmin)
df['today_n'] = (jmax - df['today_weight'])/(jmax - jmin)
wmean        = df['増減'].abs().mean()
df['wd_n']    = 1 - df['増減'].abs()/wmean

metrics = ['raw','up3_n','odds_n','jin_n','today_n','dist_n','wd_n','ped_f','style_f','age_f','sex_f','sea_f']
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = (df[m] - mu)/sd if sd else 0
    df[f'Z_{m}'] = df[f'Z_{m}'].fillna(0)

# 合成スコア
weights = {
    'Z_raw':8,'Z_up3_n':2,'Z_odds_n':1,
    'Z_jin_n':w_jin,'Z_today_n':w_jin,
    'Z_dist_n':w_best,'Z_wd_n':1,
    'Z_ped_f':3,'Z_age_f':2,'Z_style_f':2,'Z_sea_f':2
}
for k in weights:
    df[k] = df.get(k, 0).fillna(0)

df['total_z'] = sum(df[k]*w for k,w in weights.items()) / sum(weights.values())

# 馬別集計＆偏差値化
summary = df.groupby('馬名')['total_z'].agg(['mean','std']).reset_index()
summary.columns = ['馬名','mean_z','std_z']
summary['std_z'].fillna(0, inplace=True)
mz, MZ = summary['mean_z'].min(), summary['mean_z'].max()
if MZ > mz:
    summary['偏差値'] = 30 + (summary['mean_z'] - mz)/(MZ - mz)*40
else:
    summary['偏差値'] = 50
summary['安定性'] = summary['std_z']
summary['バランス'] = summary['偏差値'] - summary['安定性']

# 表示
st.subheader('本日の馬別評価')
st.dataframe(summary.sort_values('バランス', ascending=False).reset_index(drop=True))

st.subheader('上位6頭')
st.table(summary.nlargest(6,'偏差値')[['馬名','偏差値']])
