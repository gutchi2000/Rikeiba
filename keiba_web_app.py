import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations, permutations

# 日本語フォント設定
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(fname='ipaexg.ttf').get_name()

st.title('競馬予想アプリ（完成版）')

# --- サイドバー: パラメータ設定 ---
st.sidebar.subheader('1. 性別の重み')
male_weight    = st.sidebar.number_input('牡の重み', min_value=0.0, value=1.1, step=0.01)
female_weight  = st.sidebar.number_input('牝の重み', min_value=0.0, value=1.0, step=0.01)
gelding_weight = st.sidebar.number_input('セの重み', min_value=0.0, value=0.95, step=0.01)

st.sidebar.subheader('2. 脚質の重み')
nige_weight   = st.sidebar.number_input('逃げの重み', min_value=0.0, value=1.2, step=0.01)
senko_weight  = st.sidebar.number_input('先行の重み', min_value=0.0, value=1.1, step=0.01)
sashi_weight  = st.sidebar.number_input('差しの重み', min_value=0.0, value=1.0, step=0.01)
ooka_weight  = st.sidebar.number_input('追込の重み', min_value=0.0, value=0.9, step=0.01)

st.sidebar.subheader('3. 四季の重み')
spring_weight = st.sidebar.number_input('春の重み', min_value=0.0, value=1.0, step=0.01)
summer_weight = st.sidebar.number_input('夏の重み', min_value=0.0, value=1.1, step=0.01)
autumn_weight = st.sidebar.number_input('秋の重み', min_value=0.0, value=1.0, step=0.01)
winter_weight = st.sidebar.number_input('冬の重み', min_value=0.0, value=0.95, step=0.01)

st.sidebar.subheader('4. 斤量の重み (Z_斤量正規化)')
w_jin = st.sidebar.number_input('斤量重み(w_jin)', min_value=0.0, value=1.0, step=0.1)

st.sidebar.subheader('5. 距離ベストタイム重み')
w_best_dist = st.sidebar.number_input('距離ベストタイム重み(w_best_dist)', min_value=0.0, value=1.0, step=0.1)

# --- Excel アップロード ---
uploaded_file = st.file_uploader('成績＆馬情報データをアップロード (Excel)', type=['xlsx'])
if not uploaded_file:
    st.stop()

# --- シート読み込みと結合 ---
xls = pd.ExcelFile(uploaded_file)
# 1枚目: 成績データ
df = xls.parse(sheet_name=0, parse_dates=['レース日'])
# 2枚目: 馬情報 (header=1 行を列名とする)
stats = xls.parse(sheet_name=1, header=1)
if 'Unnamed: 0' in stats.columns and '馬名' not in stats.columns:
    stats.rename(columns={'Unnamed: 0':'馬名'}, inplace=True)
# デバッグ: 2枚目シートのカラムを表示
st.write('馬情報シートのカラム:', stats.columns.tolist())
# 馬情報の必須列: 馬名, 性別, 年齢
required = ['馬名','性別','年齢']
if not all(col in stats.columns for col in required):
    st.error(f"馬情報シートに必要列がありません: {stats.columns.tolist()}")
    st.stop()
# ベストタイムはシートの最終列と仮定し、数値化: "(未)" は NaN
best_col = stats.columns[-1]
stats['best_dist_time'] = pd.to_numeric(stats[best_col], errors='coerce')
stats = stats[['馬名','性別','年齢','best_dist_time']]
# 結合
df = df.merge(stats, on='馬名', how='left')

# --- 欠損ベストタイム (未出走) を最大タイムで補完 --- (未出走) を最大タイムで補完 ---
tmax = df['best_dist_time'].max()
df['best_dist_time'] = df['best_dist_time'].fillna(tmax)

# --- 脚質 & 本斤量入力 ---
equine_list = df['馬名'].unique().tolist()
input_df = pd.DataFrame({'馬名': equine_list, '脚質': ['差し']*len(equine_list), '本斤量': [56]*len(equine_list)})
edited = st.data_editor(
    input_df,
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込']),
        '本斤量': st.column_config.NumberColumn('本斤量', min_value=45, max_value=60, step=1)
    },
    use_container_width=True
)
df = df.merge(edited[['馬名','脚質','本斤量']], on='馬名', how='left')
df.rename(columns={'本斤量':'today_weight'}, inplace=True)

# --- 血統表 (HTML) アップロード ---
html_file = st.file_uploader('血統表をアップロード (HTML)', type=['html'])
ped_df = None
if html_file:
    try:
        ped_df = pd.read_html(html_file.read())[0].set_index('馬名')
    except:
        ped_df = None

# --- 強調種牡馬リスト ---
ped_input = st.text_area('強調種牡馬リスト (カンマ区切り)', '')
priority_sires = [s.strip() for s in ped_input.split(',') if s.strip()]

# --- ファクター計算関数 ---
def eval_pedigree(row):
    sire = ped_df.at[row['馬名'],'父馬'] if ped_df is not None and row['馬名'] in ped_df.index else ''
    return 1.2 if sire in priority_sires else 1.0

def style_factor(style): return {'逃げ':nige_weight,'先行':senko_weight,'差し':sashi_weight,'追込':ooka_weight}.get(style,1.0)
def age_factor(age):   return 1+0.2*(1-abs(age-5)/5)
def sex_factor(sex):  return {'牡':male_weight,'牝':female_weight,'セ':gelding_weight}.get(sex,1.0)
def seasonal_factor(dt):
    m=dt.month
    return spring_weight if m in [3,4,5] else summer_weight if m in [6,7,8] else autumn_weight if m in [9,10,11] else winter_weight

# --- ベストタイム指標: 正規化 & Zスコア化 ---
tmin = df['best_dist_time'].min(); tmax = df['best_dist_time'].max()
df['dist_norm'] = (tmax - df['best_dist_time']) / (tmax - tmin)
mu_d, sd_d = df['dist_norm'].mean(), df['dist_norm'].std(ddof=1)
df['Z_dist_norm'] = (df['dist_norm'] - mu_d) / sd_d if sd_d!=0 else 0

# --- ファクター適用 ---
df['pedigree_factor'] = df.apply(eval_pedigree, axis=1)
df['style_factor']    = df['脚質'].apply(style_factor)
df['age_factor']      = df['年齢'].apply(age_factor)
df['sex_factor']      = df['性別'].apply(sex_factor)
df['seasonal_factor'] = df['レース日'].apply(seasonal_factor)

# --- 基本スコア計算 ---
GRADE={'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
raw=df.apply(lambda r:GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']),axis=1)
df['raw']=raw*df['pedigree_factor']*df['style_factor']*df['age_factor']*df['sex_factor']*df['seasonal_factor']

# --- 正規化 ---
jmax,jmin=df['斤量'].max(),df['斤量'].min()
df['raw_norm'] = (df['raw']-1)/(10*df['頭数']-1)
df['up3_norm'] = df['Ave-3F']/df['上がり3Fタイム']
df['odds_norm']=1/(1+np.log10(df['単勝オッズ']))
df['jin_norm'] = (jmax-df['斤量'])/(jmax-jmin)
df['today_norm'] = (jmax-df['today_weight'])/(jmax-jmin)
wmean=df['増減'].abs().mean()
df['wdiff_norm']=1-df['増減'].abs()/wmean

# --- Zスコア化 ---
metrics=['raw_norm','up3_norm','odds_norm','jin_norm','today_norm','dist_norm','wdiff_norm','pedigree_factor','style_factor','age_factor','sex_factor','seasonal_factor']
for m in metrics:
    mu,sd=df[m].mean(),df[m].std(ddof=1)
    df[f'Z_{m}']=(df[m]-mu)/sd if sd else 0

# --- 合成偏差値化 ---
weights={
    'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,'Z_jin_norm':w_jin,'Z_today_norm':w_jin,'Z_dist_norm':w_best_dist,'Z_wdiff_norm':1,
    'Z_pedigree_factor':3,'Z_age_factor':2,'Z_style_factor':2,'Z_seasonal_factor':2
}
tot_w=sum(weights.values())
df['total_z']=sum(df[k]*w for k,w in weights.items())/tot_w
zmin,zmax=df['total_z'].min(),df['total_z'].max()
df['偏差値']=30+(df['total_z']-zmin)/(zmax-zmin)*40

# --- 馬別集計 & 表示 ---
summary=df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
summary.columns=['馬名','平均偏差値','安定性']
summary['バランススコア']=summary['平均偏差値']-summary['安定性']

st.subheader('馬別 評価一覧')
st.dataframe(summary.sort_values('バランススコア',ascending=False).reset_index(drop=True))

st.subheader('偏差値上位10頭')
top10 = summary.sort_values('平均偏差値',ascending=False).head(10)
st.table(top10[['馬名','平均偏差値']])

st.subheader('本日の予想6頭')
combined = summary.sort_values('バランススコア',ascending=False).head(6)
st.table(combined)

# 可視化: バランススコア棒グラフ
fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.barh(combined['馬名'], combined['バランススコア'])
ax1.invert_yaxis()
ax1.set_xlabel('バランススコア')
st.pyplot(fig1)

# 散布図: 平均偏差値 vs 安定性
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.scatter(summary['平均偏差値'], summary['安定性'], alpha=0.7)
ax2.axvline(summary['平均偏差値'].mean(), linestyle='--')
ax2.axhline(summary['安定性'].mean(), linestyle='--')
for _, row in summary.iterrows():
    ax2.text(row['平均偏差値'], row['安定性'], row['馬名'], fontsize=8)
ax2.set_xlabel('平均偏差値')
ax2.set_ylabel('安定性')
st.pyplot(fig2)

# 予想タグ
st.subheader('本日の予想タグ')
tag_map = {1:'◎',2:'〇',3:'▲',4:'☆',5:'△',6:'△'}
pred = combined.reset_index(drop=True).copy()
pred['タグ'] = pred.index.map(lambda i: tag_map.get(i+1,''))
st.table(pred[['馬名','タグ','平均偏差値']])

# ベット設定
st.subheader('ベット設定')
scenarios = {
    '通常':{'単勝':8,'複勝':22,'ワイド':40,'馬連':20,'三連複':0,'三連単':0},
    '攻め':{'単勝':5,'複勝':15,'ワイド':30,'馬連':20,'三連複':15,'三連単':15}
}
scenario = st.selectbox('シナリオ', list(scenarios.keys()))
budget = st.number_input('予算 (円)', min_value=1000, value=10000, step=1000)

@st.cache_data
def allocate_budget(budget, perc):
    raw = {k:budget*v/100 for k,v in perc.items()}
    rnd = {k:int(v//100)*100 for k,v in raw.items()}
    diff = budget - sum(rnd.values())
    if diff>0:
        rnd[max(perc, key=perc.get)] += diff
    return rnd

alloc = allocate_budget(budget, scenarios[scenario])
alloc_df = pd.DataFrame.from_dict(alloc, orient='index', columns=['金額']).reset_index()
alloc_df.columns = ['券種','金額(円)']
st.table(alloc_df)

detail = st.selectbox('詳細表示', alloc_df['券種'])
names = combined['馬名'].tolist()
axis = names[0] if names else ''
amt = alloc.get(detail, 0)
if detail in ['単勝','複勝']:
    st.write(f"{detail}：軸馬 {axis} に {amt:,}円")
else:
    if detail in ['馬連','ワイド']:
        combos = [f"{a}-{b}" for a,b in combinations(names,2)]
    elif detail=='馬連':
        combos = [f"{axis}->{o}" for o in names if o!=axis]
    elif detail=='三連複':
        combos = [f"{axis}-{o1}-{o2}" for o1,o2 in combinations(names[1:],2)]
    elif detail=='三連単':
        combos = ["->".join(p) for p in permutations(names,3)]
    else:
        combos = []
    cnt = len(combos)
    if cnt>0:
        unit = (amt//cnt)//100*100
        rem = amt - unit*cnt
        amounts = [unit+100 if i<rem//100 else unit for i in range(cnt)]
        bet_df = pd.DataFrame({'組合せ':combos,'金額':amounts})
        st.dataframe(bet_df)
    else:
        st.write('対象の買い目がありません')
