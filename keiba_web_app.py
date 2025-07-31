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

# ── サイドバー: 各種重み ──
st.sidebar.subheader('1. 性別の重み')
male_weight    = st.sidebar.number_input('牡の重み',   0.0, 1.1, 0.01)
female_weight  = st.sidebar.number_input('牝の重み',   0.0, 1.0, 0.01)
gelding_weight = st.sidebar.number_input('セの重み',   0.0, 0.95,0.01)

st.sidebar.subheader('2. 脚質の重み')
nige_weight   = st.sidebar.number_input('逃げの重み', 0.0,1.2,0.01)
senko_weight  = st.sidebar.number_input('先行の重み', 0.0,1.1,0.01)
sashi_weight  = st.sidebar.number_input('差しの重み', 0.0,1.0,0.01)
ooka_weight   = st.sidebar.number_input('追込の重み', 0.0,0.9,0.01)

st.sidebar.subheader('3. 四季の重み')
spring_weight = st.sidebar.number_input('春の重み',   0.0,1.0,0.01)
summer_weight = st.sidebar.number_input('夏の重み',   0.0,1.1,0.01)
autumn_weight = st.sidebar.number_input('秋の重み',   0.0,1.0,0.01)
winter_weight = st.sidebar.number_input('冬の重み',   0.0,0.95,0.01)

st.sidebar.subheader('4. 斤量の重み')
w_jin = st.sidebar.number_input('w_jin', 0.0,1.0,0.1)

st.sidebar.subheader('5. 距離ベストタイム重み')
w_best_dist = st.sidebar.number_input('w_best_dist', 0.0,1.0,0.1)

# ── データアップロード ──
uploaded = st.file_uploader('成績＆馬情報をアップロード', type='xlsx')
if not uploaded:
    st.stop()

xls = pd.ExcelFile(uploaded)
# 成績シート
df = xls.parse(0, parse_dates=['レース日'])
# 馬情報シート
stats = xls.parse(1, header=1)
stats = stats.rename(columns={
    stats.columns[0]:'馬名', stats.columns[1]:'性別',
    stats.columns[2]:'年齢', stats.columns[-1]:'ベストタイム'
})[['馬名','性別','年齢','ベストタイム']]
stats['best_dist_time'] = pd.to_numeric(
    stats['ベストタイム'].replace({'(未)':np.nan}), errors='coerce'
)
stats['best_dist_time'].fillna(stats['best_dist_time'].max(), inplace=True)
df = df.merge(stats[['馬名','性別','年齢','best_dist_time']], on='馬名', how='left')

# ── 脚質・本斤量入力 ──
equines = df['馬名'].unique()
input_df = pd.DataFrame({
    '馬名':equines, '脚質':['差し']*len(equines),
    '本斤量':[56]*len(equines)
})
edited = st.data_editor(
    input_df,
    column_config={
        '脚質':st.column_config.SelectboxColumn('脚質',
                 options=['逃げ','先行','差し','追込']),
        '本斤量':st.column_config.NumberColumn('本斤量',
                 min_value=45,max_value=60,step=1)
    },
    use_container_width=True
)
df = df.merge(edited, on='馬名', how='left').rename(
    columns={'本斤量':'today_weight'}
)

# ── 血統表アップロード ──
html = st.file_uploader('血統表をアップロード', type='html')
ped_df = None
if html:
    try:
        ped_df = pd.read_html(html.read())[0].set_index('馬名')
    except:
        ped_df = None

priority = [s.strip() for s in st.text_area(
    '強調種牡馬(カンマ区切り)', ''
).split(',') if s.strip()]

# ── ファクター関数 ──
def eval_pedigree(r):
    if ped_df is None or r['馬名'] not in ped_df.index:
        return 1.0
    return 1.2 if ped_df.at[r['馬名'],'父馬'] in priority else 1.0

def style_factor(s):
    return {'逃げ':nige_weight,'先行':senko_weight,
            '差し':sashi_weight,'追込':ooka_weight}.get(s,1.0)
def age_factor(a):
    return 1+0.2*(1-abs(a-5)/5)
def sex_factor(sx):
    return {'牡':male_weight,'牝':female_weight,
            'セ':gelding_weight}.get(sx,1.0)
def seasonal(dt):
    m=dt.month
    if m in [3,4,5]:   return spring_weight
    if m in [6,7,8]:   return summer_weight
    if m in [9,10,11]: return autumn_weight
    return winter_weight

# ── 正規化 & Z化 ──
tmin,tmax = df['best_dist_time'].min(),df['best_dist_time'].max()
df['dist_norm'] = (tmax-df['best_dist_time'])/(tmax-tmin)
mu,sd = df['dist_norm'].mean(),df['dist_norm'].std(ddof=1)
df['Z_dist_norm'] = (df['dist_norm']-mu)/sd if sd else 0

# ファクター適用
df['pedigree_factor'] = df.apply(eval_pedigree,axis=1)
df['style_factor']    = df['脚質'].map(style_factor)
df['age_factor']      = df['年齢'].map(age_factor)
df['sex_factor']      = df['性別'].map(sex_factor)
df['seasonal_factor'] = df['レース日'].map(seasonal)

# raw スコア
GRADE = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
         '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
raw = df.apply(lambda r:GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']),axis=1)
df['raw'] = raw * df['pedigree_factor'] * df['style_factor'] \
            * df['age_factor'] * df['sex_factor'] * df['seasonal_factor']

# 追加指標正規化
jmax,jmin = df['斤量'].max(),df['斤量'].min()
df['up3_norm']   = df['Ave-3F'] / df['上がり3Fタイム']
df['odds_norm']  = 1/(1+np.log10(df['単勝オッズ']))
df['jin_norm']   = (jmax-df['斤量'])/(jmax-jmin)
df['today_norm'] = (jmax-df['today_weight'])/(jmax-jmin)
wmean = df['増減'].abs().mean()
df['wdiff_norm']=1-df['増減'].abs()/wmean

# Z スコア化
metrics = ['raw','up3_norm','odds_norm','jin_norm','today_norm',
           'dist_norm','wdiff_norm','pedigree_factor',
           'style_factor','age_factor','sex_factor','seasonal_factor']
for m in metrics:
    mu,sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = (df[m]-mu)/sd if sd else 0

# 合成 total_z
weights = {
    'Z_raw':8, 'Z_up3_norm':2, 'Z_odds_norm':1,
    'Z_jin_norm':w_jin,'Z_today_norm':w_jin,
    'Z_dist_norm':w_best_dist,'Z_wdiff_norm':1,
    'Z_pedigree_factor':3,'Z_age_factor':2,
    'Z_style_factor':2,'Z_seasonal_factor':2
}
tot_w = sum(weights.values())
df['total_z'] = sum(df[k]*w for k,w in weights.items())/tot_w

# ── 馬別集計：ここで偏差値化 ──
summary = df.groupby('馬名')['total_z'].agg(['mean','std']).reset_index()
summary.columns = ['馬名','mean_z','std_z']
# 安定性は std、欠損は 0 に
summary['std_z'] = summary['std_z'].fillna(0)
# 偏差値化
zmin, zmax = summary['mean_z'].min(), summary['mean_z'].max()
if zmax > zmin:
    summary['平均偏差値'] = 30 + (summary['mean_z']-zmin)/(zmax-zmin)*40
else:
    summary['平均偏差値'] = 50
summary['安定性']    = summary['std_z']
summary['バランススコア'] = summary['平均偏差値'] - summary['安定性']

# ── 表示 ──
st.subheader('馬別 評価一覧')
st.dataframe(summary[['馬名','平均偏差値','安定性','バランススコア']]\
             .sort_values('バランススコア',ascending=False).reset_index(drop=True))

st.subheader('偏差値上位10頭')
st.table(summary.nlargest(10,'平均偏差値')[['馬名','平均偏差値']])

st.subheader('本日の予想6頭')
pred6 = summary.nlargest(6,'バランススコア').reset_index(drop=True)
st.table(pred6[['馬名','平均偏差値','安定性','バランススコア']])

# 以下、ビジュアル化・ベット設定部分は前と同様に記述してください。
