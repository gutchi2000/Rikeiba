import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations, permutations
# 日本語フォント設定

# 日本語フォント設定
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(
    fname='ipaexg.ttf'
).get_name()

st.set_page_config(layout="wide")
st.title('競馬予想アプリ（完成版 + 血統加点）')

# サイドバー：重み設定
with st.sidebar:
    st.header("重み設定")
    # 性別・脚質・季節・斤量重み
    male_w    = st.number_input('牡の重み',   0.0, 2.0, 1.1)
    female_w  = st.number_input('牝の重み',   0.0, 2.0, 1.0)
    gelding_w = st.number_input('セの重み',   0.0, 2.0, 0.95)
    nige_w    = st.number_input('逃げの重み',  0.0, 2.0, 1.2)
    senko_w   = st.number_input('先行の重み',  0.0, 2.0, 1.1)
    sashi_w   = st.number_input('差しの重み',  0.0, 2.0, 1.0)
    ooka_w    = st.number_input('追込の重み',  0.0, 2.0, 0.9)
    spring_w  = st.number_input('春の重み',    0.0, 2.0, 1.0)
    summer_w  = st.number_input('夏の重み',    0.0, 2.0, 1.1)
    autumn_w  = st.number_input('秋の重み',    0.0, 2.0, 1.0)
    winter_w  = st.number_input('冬の重み',    0.0, 2.0, 0.95)
    w_jin     = st.number_input('斤量重み',       0.0, 1.0, 1.0)
    w_best    = st.number_input('距離ベスト重み', 0.0, 1.0, 1.0)
    weight_z  = st.slider('偏差値の重み',        0.0, 1.0, 0.7)
    weight_rb = st.slider('RawBase偏差値の重み', 0.0, 1.0, 0.3)
    # 血統・馬名強調入力
    st.subheader("血統・馬名強調入力")
    highlight_inputs = st.text_area(
        '強調系統・馬名（カンマ区切り）',
        value='ミスタープロスペクター系,ヘイロー系,ナルスーラ系'
    )

# データアップロード
st.subheader("データアップロード")
c1, c2 = st.columns(2)
with c1:
    upload_xlsx = st.file_uploader('成績＆馬情報 (XLSX)', type='xlsx')
with c2:
    upload_html = st.file_uploader('血統表 (HTML)', type='html')
if not upload_xlsx:
    st.stop()

# 成績データ読み込み
xls = pd.ExcelFile(upload_xlsx)
df = xls.parse(0, parse_dates=['レース日'])
df['馬名'] = df['馬名'].astype(str).str.strip()
stats = xls.parse(1, header=1)
# 馬情報整形
keys = ['馬名','性別','年齢','ベストタイム']
col_map = {}
for k in keys:
    if k not in stats.columns:
        stats[k] = np.nan
stats = stats[keys].drop_duplicates('馬名').drop_duplicates('馬名')
stats['馬名'] = stats['馬名'].astype(str).str.strip()
stats['best_dist_time'] = pd.to_numeric(
    stats['ベストタイム'].replace({'(未)':np.nan}), errors='coerce'
).fillna(
    stats['ベストタイム'].str.extract(r'(\d+)', expand=False).astype(float)
)
df = df.merge(stats[['馬名','性別','年齢','best_dist_time']], on='馬名', how='left')

# 脚質・本斤量設定
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
df = df.merge(edited, on='馬名', how='left')

# 血統表読み込み
ped = None
if upload_html:
    try:
        ped_table = pd.read_html(upload_html.read())[0]
        ped = ped_table.set_index('馬名')
    except Exception:
        ped = None
user_terms = [t.strip() for t in highlight_inputs.split(',') if t.strip()]

# ファクター関数
def style_f(s): return {'逃げ':nige_w,'先行':senko_w,'差し':sashi_w,'追込':ooka_w}.get(s,1)
def age_f(a):   return 1 + 0.2*(1-abs(a-5)/5)
def sex_f(s):   return {'牡':male_w,'牝':female_w,'セ':gelding_w}.get(s,1)
def sea_f(d):
    m = d.month
    return spring_w if m in [3,4,5] else summer_w if m in [6,7,8] else autumn_w if m in [9,10,11] else winter_w

def ped_factor(r):
    if ped is None or r['馬名'] not in ped.index:
        return 1
    score = 0
    father = ped.at[r['馬名'],'父馬']
    mother = ped.at[r['馬名'],'母父']
    for term in user_terms:
        if term == r['馬名']:
            score += 2
        if term in [father, mother]:
            score += 1
    return 1 + 0.1 * score

# 生スコア
grades = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
grade_score = {k:v**2 for k,v in grades.items()}

df['RawBase'] = df.apply(lambda r: grade_score.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['Raw'] = df['RawBase'] * df.apply(ped_factor,axis=1) * df['脚質'].map(style_f) * df['年齢'].map(age_f) * df['性別'].map(sex_f) * df['レース日'].map(sea_f)

# 正規化＆Z化
# 距離
tmin, tmax = df['best_dist_time'].min(), df['best_dist_time'].max()
df['dist_n'] = (tmax - df['best_dist_time'])/(tmax - tmin)
# 上がり3F
df['up3_n']  = df['Ave-3F']/df['上がり3Fタイム']
# オッズ
df['odds_n'] = 1/(1+np.log10(df['単勝オッズ']))
# 斤量
jmax, jmin = df['斤量'].max(), df['斤量'].min()
df['jin_n']   = (jmax - df['斤量'])/(jmax - jmin)
# 今日の斤量
if '本斤量' in df.columns:
    df['today_n'] = (jmax - df['本斤量'])/(jmax - jmin)
# 増減重量
wabs = df['増減'].abs().mean()
df['wd_n']    = 1 - df['増減'].abs()/wabs
# Raw
df['raw_n']   = (df['Raw'] - df['Raw'].min())/(df['Raw'].max()-df['Raw'].min())

# Zスコア
metrics = ['raw_n','up3_n','odds_n','jin_n','today_n','dist_n','wd_n']
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = ((df[m]-mu)/sd).fillna(0)

# 合成スコア
cw = {'Z_raw_n':8,'Z_up3_n':2,'Z_odds_n':1,'Z_jin_n':w_jin,'Z_today_n':w_jin,'Z_dist_n':w_best,'Z_wd_n':1}
tot = sum(cw.values())
df['total_z'] = sum(df[k]*w for k,w in cw.items())/tot

# 馬別集計＆偏差値
summary = df.groupby('馬名').agg(mean_z=('total_z','mean'),std_z=('total_z','std'),RB_mean=('RawBase','mean')).reset_index()
summary['std_z'] = summary['std_z'].fillna(0)
# 偏差値
mn,Mx = summary['mean_z'].min(),summary['mean_z'].max()
summary['偏差値'] = 30+(summary['mean_z']-mn)/(Mx-mn)*40
# RawBase偏差値
rbm,rbM = summary['RB_mean'].min(),summary['RB_mean'].max()
summary['RawBase偏差値'] = 30+(summary['RB_mean']-rbm)/(rbM-rbm)*40
# バランス
summary['バランス'] = weight_z*summary['偏差値']+weight_rb*summary['RawBase偏差値']-summary['std_z']

# --- デバッグ: 指標別平均Zスコアチェック ---
st.subheader("[デバッグ] 生スコア系 Zスコア平均")
debug_df = pd.DataFrame({
    '指標': ['Z_raw_n','Z_up3_n','Z_odds_n','Z_jin_n','Z_today_n','Z_dist_n','Z_wd_n'],
    '平均Z': [df[f'Z_{m.split("_")[1]}'].mean() for m in ['Z_raw_n','Z_up3_n','Z_odds_n','Z_jin_n','Z_today_n','Z_dist_n','Z_wd_n']]
})
st.table(debug_df)


# （デバッグ用テーブルは削除しました）

# 本日の予想6頭
top6 = summary.nlargest(6,'バランス').reset_index(drop=True)
tag = {1:'◎',2:'〇',3:'▲',4:'△',5:'△',6:'△'}
top6['印'] = top6.index.map(lambda i: tag[i+1])
st.table(top6[['印','馬名','偏差値','RawBase偏差値','バランス']])

# グラフ
c3,c4 = st.columns(2)
with c3:
    fig,ax=plt.subplots(figsize=(6,4))
    ax.barh(top6['印']+' '+top6['馬名'],top6['バランス'])
    ax.invert_yaxis()
    ax.set_xlabel('バランス')
    st.pyplot(fig)
with c4:
    fig,ax=plt.subplots(figsize=(6,4))
    ax.scatter(summary['偏差値'],summary['std_z'],alpha=0.7)
    ax.axvline(summary['偏差値'].mean(),ls='--')
    ax.axhline(summary['std_z'].mean(),ls='--')
    for _,r in summary.iterrows(): ax.text(r['偏差値'],r['std_z'],r['馬名'],fontsize=6)
    st.pyplot(fig)

# ベット配分
st.subheader("ベットシナリオ＆配分")
sc = {'通常':{'単勝':8,'複勝':22,'ワイド':40,'馬連':20,'三連複':0,'三連単':0},
      '攻め':{'単勝':5,'複勝':15,'ワイド':30,'馬連':20,'三連複':15,'三連単':15},
      '余裕':{'単勝':5,'複勝':15,'ワイド':25,'馬連':15,'三連複':20,'三連単':25}}
mode = st.selectbox('シナリオ',list(sc))
bud  = st.number_input('予算',1000,1000000,10000,step=1000)
@st.cache_data
def alloc(b,p):
    raw={k:b*v/100 for k,v in p.items()}
    rnd={k:int(x//100)*100 for k,x in raw.items()}
    d=b-sum(rnd.values())
    if d>0: rnd[max(p,key=p.get)]+=d
    return rnd
alc=alloc(bud,sc[mode])
st.table(pd.DataFrame.from_dict(alc,orient='index',columns=['金額']).reset_index().rename(columns={'index':'券種'}))

# 買い目詳細
st.subheader("買い目詳細")
bet = st.selectbox('券種',list(alc))
nm=top6['馬名'].tolist()
ax = nm[0] if nm else ''
am = alc[bet]
if bet in ['単勝','複勝']:
    st.write(f"{bet}：軸馬{ax}に{am}円")
else:
    if bet in ['馬連','ワイド']:
        cmb=[f"{a}-{b}" for a,b in combinations(nm,2)]
    elif bet=='三連複':
        cmb=[f"{ax}-{x}-{y}" for x,y in combinations(nm[1:],2)]
    else:
        cmb=["->".join(p) for p in permutations(nm,3)]
    cnt=len(cmb)
    unit=(am//cnt)//100*100 if cnt else 0
    rem=am-unit*cnt
    vals=[unit+100 if i<rem//100 else unit for i in range(cnt)]
    st.data_editor(pd.DataFrame({'組合せ':cmb,'金額':vals}))
