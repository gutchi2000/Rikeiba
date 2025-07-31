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

st.set_page_config(layout="wide")
st.title('競馬予想アプリ（完成版）')

# ── サイドバー：重み設定 ──
with st.sidebar:
    st.header("重み設定")
    male_w    = st.number_input('牡の重み',   0.0, 2.0, 1.1, 0.01)
    female_w  = st.number_input('牝の重み',   0.0, 2.0, 1.0, 0.01)
    gelding_w = st.number_input('セの重み',   0.0, 2.0, 0.95,0.01)
    nige_w    = st.number_input('逃げの重み',  0.0, 2.0, 1.2, 0.01)
    senko_w   = st.number_input('先行の重み',  0.0, 2.0, 1.1, 0.01)
    sashi_w   = st.number_input('差しの重み',  0.0, 2.0, 1.0, 0.01)
    ooka_w    = st.number_input('追込の重み',  0.0, 2.0, 0.9, 0.01)
    spring_w  = st.number_input('春の重み',    0.0, 2.0, 1.0, 0.01)
    summer_w  = st.number_input('夏の重み',    0.0, 2.0, 1.1, 0.01)
    autumn_w  = st.number_input('秋の重み',    0.0, 2.0, 1.0, 0.01)
    winter_w  = st.number_input('冬の重み',    0.0, 2.0, 0.95,0.01)
    w_jin     = st.number_input('斤量重み',       0.0, 1.0, 1.0, 0.1)
    w_best    = st.number_input('距離ベスト重み', 0.0, 1.0, 1.0, 0.1)

# ── データアップロード ──
st.subheader("データアップロード")
col1, col2 = st.columns(2)
with col1:
    upload_xlsx = st.file_uploader('成績＆馬情報 (XLSX)', type='xlsx')
with col2:
    upload_html = st.file_uploader('血統表 (HTML)', type='html')
if not upload_xlsx:
    st.stop()

# ── 成績と馬情報の読み込み ──
xls = pd.ExcelFile(upload_xlsx)
df  = xls.parse(0, parse_dates=['レース日'])
df['馬名'] = df['馬名'].astype(str).str.strip()

# 馬情報シート
stats = xls.parse(1, header=1)
keys = ['馬名','性別','年齢','ベストタイム']
col_map = {}
for k in keys:
    for c in stats.columns:
        if k in str(c):
            col_map[c] = k
            break
stats = stats.rename(columns=col_map)
for k in keys:
    if k not in stats.columns:
        stats[k] = np.nan
stats = stats[keys].drop_duplicates('馬名')
stats['馬名'] = stats['馬名'].astype(str).str.strip()
stats['best_dist_time'] = pd.to_numeric(
    stats['ベストタイム'].replace({'(未)':np.nan}), errors='coerce'
).fillna(stats['ベストタイム'].astype(str)
         .str.extract(r'(\d+)', expand=False).astype(float).max())
df = df.merge(stats[['馬名','性別','年齢','best_dist_time']], on='馬名', how='left')

# ── 血統表読み込み ──
ped = None
if upload_html:
    try:
        ped = pd.read_html(upload_html.read())[0].set_index('馬名')
    except:
        ped = None
highlight_sires = [s.strip() for s in st.text_area('強調種牡馬 (カンマ区切り)').split(',') if s.strip()]

# ── 脚質＆斤量入力 ──
st.subheader("脚質・本斤量設定")
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

# ── ファクター関数定義 ──
def ped_factor(r):
    if ped is None or r['馬名'] not in ped.index:
        return 1.0
    return 1.2 if ped.at[r['馬名'],'父馬'] in highlight_sires else 1.0
def style_f(s): return {'逃げ':nige_w,'先行':senko_w,'差し':sashi_w,'追込':ooka_w}.get(s,1.0)
def age_f(a):   return 1+0.2*(1-abs(a-5)/5)
def sex_f(sx):  return {'牡':male_w,'牝':female_w,'セ':gelding_w}.get(sx,1.0)
def sea_f(dt):
    m = dt.month
    return spring_w if m in [3,4,5] else summer_w if m in [6,7,8] else autumn_w if m in [9,10,11] else winter_w

# ── 生スコア計算 ──
GRADE = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
         '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
df['RawBase'] = df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['Raw']     = df['RawBase'] \
                * df.apply(ped_factor,axis=1) \
                * df['脚質'].map(style_f) \
                * df['年齢'].map(age_f) \
                * df['性別'].map(sex_f) \
                * df['レース日'].map(sea_f)

# ── 指標の正規化＆Zスコア化 ──
tmin,tmax = df['best_dist_time'].min(), df['best_dist_time'].max()
df['dist_n']   = (tmax-df['best_dist_time'])/(tmax-tmin)
df['up3_n']    = df['Ave-3F']/df['上がり3Fタイム']
df['odds_n']   = 1/(1+np.log10(df['単勝オッズ']))
jmax,jmin      = df['斤量'].max(), df['斤量'].min()
df['jin_n']    = (jmax-df['斤量'])/(jmax-jmin)
df['today_n']  = (jmax-df['today_weight'])/(jmax-jmin)
wabs           = df['増減'].abs().mean()
df['wd_n']     = 1 - df['増減'].abs()/wabs
df['raw_n']    = (df['Raw']-1)/(10*df['頭数']-1)

metrics = ['raw_n','up3_n','odds_n','jin_n','today_n','dist_n','wd_n']
for m in metrics:
    mu,sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = ((df[m]-mu)/sd).fillna(0) if sd else 0

# ── 合成スコア total_z ──
weights = {
    'Z_raw_n':8,'Z_up3_n':2,'Z_odds_n':1,
    'Z_jin_n':w_jin,'Z_today_n':w_jin,
    'Z_dist_n':w_best,'Z_wd_n':1
}
totw = sum(weights.values())
df['total_z'] = sum(df[k]*w for k,w in weights.items())/totw

# ── 馬別集計＆偏差値化＋RawBase偏差値 ──
summary = df.groupby('馬名').agg(
    mean_z=('total_z','mean'),
    std_z =('total_z','std'),
    RawBase_mean=('RawBase','mean')
).reset_index()
summary['std_z'] = summary['std_z'].fillna(0)

# 偏差値化(mean_z→偏差値)
mz,MZ = summary['mean_z'].min(), summary['mean_z'].max()
summary['偏差値'] = 30 + (summary['mean_z']-mz)/(MZ-mz)*40 if MZ>mz else 50

# RawBase_meanを偏差値化
rb_min, rb_max = summary['RawBase_mean'].min(), summary['RawBase_mean'].max()
summary['RawBase偏差値'] = 30 + (summary['RawBase_mean']-rb_min)/(rb_max-rb_min)*40 if rb_max>rb_min else 50

# 最終バランス
summary['バランス'] = summary['偏差値']*0.8 + summary['RawBase偏差値']*0.2 - summary['std_z']

# ── 結果表示 ──
st.subheader("馬別 評価一覧")
st.dataframe(summary.sort_values('バランス',ascending=False).reset_index(drop=True), use_container_width=True)

st.subheader("偏差値上位10頭")
st.table(summary.nlargest(10,'偏差値')[['馬名','偏差値']])

# ── 本日の予想6頭＋印付け ──
st.subheader("本日の予想6頭（最終バランス順）")
top6 = summary.nlargest(6,'バランス').reset_index(drop=True)
tag_map = {1:'◎',2:'〇',3:'▲',4:'△',5:'△',6:'△'}
top6['印'] = top6.index.to_series().map(lambda i: tag_map[i+1])
st.table(top6[['印','馬名','偏差値','RawBase偏差値','std_z','バランス']])

# ── グラフ ──
col3, col4 = st.columns(2)
with col3:
    st.markdown("**バランススコア棒グラフ**")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.barh(top6['印'] + ' ' + top6['馬名'], top6['バランス'])
    ax1.invert_yaxis()
    ax1.set_xlabel('最終バランス')
    st.pyplot(fig1)
with col4:
    st.markdown("**偏差値 vs 安定性**")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.scatter(summary['偏差値'], summary['std_z'], alpha=0.7)
    ax2.axvline(summary['偏差値'].mean(), linestyle='--')
    ax2.axhline(summary['std_z'].mean(), linestyle='--')
    for _,r in summary.iterrows():
        ax2.text(r['偏差値'], r['std_z'], r['馬名'], fontsize=6)
    ax2.set_xlabel('偏差値'); ax2.set_ylabel('安定性')
    st.pyplot(fig2)

# ── ベット設定 ──
st.subheader("ベットシナリオ＆配分")
scenarios = {
    '通常': {'単勝':8,'複勝':22,'ワイド':40,'馬連':20,'三連複':0,'三連単':0},
    '攻め': {'単勝':5,'複勝':15,'ワイド':30,'馬連':20,'三連複':15,'三連単':15},
    '余裕': {'単勝':5,'複勝':15,'ワイド':25,'馬連':15,'三連複':20,'三連単':25}
}
scenario = st.selectbox('シナリオ', list(scenarios.keys()))
budget   = st.number_input('予算 (円)', 1000, 1000000, 10000, step=1000)

@st.cache_data
def allocate(bud, pct):
    raw = {k: bud*v/100 for k,v in pct.items()}
    rnd = {k: int(x//100)*100 for k,x in raw.items()}
    diff= bud - sum(rnd.values())
    if diff>0: rnd[max(pct, key=pct.get)] += diff
    return rnd

alloc = allocate(budget, scenarios[scenario])
alloc_df = pd.DataFrame.from_dict(alloc, orient='index', columns=['金額']).reset_index().rename(columns={'index':'券種'})
st.table(alloc_df)

st.subheader("買い目詳細")
bets = st.selectbox('券種を選択', alloc_df['券種'])
names = top6['馬名'].tolist()
axis = names[0] if names else ''
amt  = alloc.get(bets,0)

if bets in ['単勝','複勝']:
    st.write(f"{bets}：軸馬 **{axis}** に **{amt:,} 円**")
else:
    if bets in ['馬連','ワイド']:
        combos = [f"{a}-{b}" for a,b in combinations(names,2)]
    elif bets=='三連複':
        combos = [f"{axis}-{b1}-{b2}" for b1,b2 in combinations(names[1:],2)]
    elif bets=='三連単':
        combos = ["->".join(p) for p in permutations(names,3)]
    else:
        combos = []
    cnt  = len(combos)
    unit = (amt//cnt)//100*100 if cnt>0 else 0
    rem  = amt - unit*cnt
    amounts = [unit+100 if i<rem//100 else unit for i in range(cnt)]
    bet_df = pd.DataFrame({'組合せ':combos,'金額(円)':amounts})
    st.data_editor(bet_df, use_container_width=True)
