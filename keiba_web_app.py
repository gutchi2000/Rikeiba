import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations, permutations

# ── 日本語フォント設定 ──
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(fname='ipaexg.ttf').get_name()

st.set_page_config(layout="wide")
st.title('競馬予想アプリ（完成版）')

# ── サイドバー：重み設定 ──
with st.sidebar:
    st.header("重み設定")
    # 性別補正
    with st.expander("性別の重み"):
        male_w    = st.number_input('牡の重み',   0.0, 2.0, 1.1, 0.01)
        female_w  = st.number_input('牝の重み',   0.0, 2.0, 1.0, 0.01)
        gelding_w = st.number_input('セの重み',   0.0, 2.0, 0.95,0.01)
    # 脚質補正
    with st.expander("脚質の重み"):
        nige_w    = st.number_input('逃げの重み', 0.0, 2.0, 1.2, 0.01)
        senko_w   = st.number_input('先行の重み', 0.0, 2.0, 1.1, 0.01)
        sashi_w   = st.number_input('差しの重み', 0.0, 2.0, 1.0, 0.01)
        ooka_w    = st.number_input('追込の重み', 0.0, 2.0, 0.9, 0.01)
    # 四季補正
    with st.expander("四季の重み"):
        spring_w  = st.number_input('春の重み',   0.0, 2.0, 1.0, 0.01)
        summer_w  = st.number_input('夏の重み',   0.0, 2.0, 1.1, 0.01)
        autumn_w  = st.number_input('秋の重み',   0.0, 2.0, 1.0, 0.01)
        winter_w  = st.number_input('冬の重み',   0.0, 2.0, 0.95,0.01)
    # その他指標
    w_jin     = st.number_input('斤量重み',      0.0, 1.0, 1.0, 0.1)
    w_best    = st.number_input('距離ベスト重み',0.0, 1.0, 1.0, 0.1)
    # 枠順補正
    with st.expander("枠順の重み (1〜8)"):
        gate_weights = {i: st.number_input(f'{i}枠の重み',0.0,2.0,1.0,0.01) for i in range(1,9)}
    # 最終スコア重み
    with st.expander("最終スコア重み", expanded=True):
        weight_z    = st.slider('偏差値 の重み',         0.0, 1.0, 0.7, 0.05)
        weight_rb   = st.slider('RawBase偏差値 の重み',  0.0, 1.0, 0.2, 0.05)
        weight_gate = st.slider('枠偏差値 の重み',      0.0, 1.0, 0.1, 0.05)

# キャッシュ注意
st.write("**注意：設定変更後はClear cacheで再実行してください。**")

# --- アップロード ---
st.subheader("データアップロード")
c1, c2 = st.columns(2)
with c1:
    upload_xlsx = st.file_uploader('成績＆馬情報 (XLSX)', type='xlsx')
with c2:
    upload_html = st.file_uploader('血統表 (HTML)', type='html')
if not upload_xlsx:
    st.stop()

# --- データ読み込み ---
xls = pd.ExcelFile(upload_xlsx)
df  = xls.parse(0, parse_dates=['レース日'])
df['馬名'] = df['馬名'].astype(str).str.strip()
stats = xls.parse(1, header=1)
# 馬情報整形
keys=['馬名','性別','年齢','ベストタイム']
col_map={c:k for k in keys for c in stats.columns if k in str(c)}
stats=stats.rename(columns=col_map)
for k in keys:
    if k not in stats.columns:
        stats[k] = np.nan
stats=stats[keys].drop_duplicates('馬名')
stats['馬名']=stats['馬名'].astype(str)
# 枠-番-馬名 分割

def split_frame(x):
    p=str(x).split('-',2)
    return pd.Series({'枠':int(p[0]) if p[0].isdigit() else 1,
                      '番':int(p[1]) if len(p)>1 and p[1].isdigit() else np.nan,
                      '馬名':p[-1].strip()})
sf=stats['馬名'].apply(split_frame)
stats=pd.concat([stats.drop(columns='馬名'),sf],axis=1)
stats['best_dist_time']=pd.to_numeric(stats['ベストタイム'].replace({'(未)':np.nan}),errors='coerce').fillna(stats['ベストタイム'].str.extract(r'(\d+)',expand=False).astype(float).max())
# マージ
df=df.merge(stats[['枠','番','馬名','性別','年齢','best_dist_time']],on='馬名',how='left')
# 血統表（未利用）
if upload_html:
    try: ped=pd.read_html(upload_html.read())[0].set_index('馬名')
    except: ped=None
else: ped=None

# --- 脚質・斤量 ---
st.subheader("脚質・斤量設定")
input_df=pd.DataFrame({'馬名':df['馬名'].unique(),'脚質':['差し']*len(df),'本斤量':[56]*len(df)})
ed=st.data_editor(input_df,column_config={'脚質':st.column_config.SelectboxColumn('脚質',['逃げ','先行','差し','追込']),'本斤量':st.column_config.NumberColumn('本斤量',45,60,1)},use_container_width=True)
ed['馬名']=ed['馬名'].astype(str).str.strip()
df=df.merge(ed.rename(columns={'本斤量':'today_weight'}),on='馬名',how='left')

# --- ファクター ---
def ped_factor(r): return 1.0
def style_f(s): return {'逃げ':nige_w,'先行':senko_w,'差し':sashi_w,'追込':ooka_w}.get(s,1)
def age_f(a): return 1+0.2*(1-abs(a-5)/5)
def sex_f(sx): return {'牡':male_w,'牝':female_w,'セ':gelding_w}.get(sx,1)
def sea_f(dt):m=dt.month;return spring_w if m in range(3,6) else summer_w if m in range(6,9) else autumn_w if m in range(9,12) else winter_w
def gate_Z(r): return gate_weights.get(r['枠'],1)

# --- スコア計算 ---
GRADE={'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
gs={k:v**2 for k,v in GRADE.items()}
df['RawBase']=df.apply(lambda r:gs.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']),axis=1)
df['Raw']=df['RawBase']*df['脚質'].map(style_f)*df['年齢'].map(age_f)*df['性別'].map(sex_f)*df['レース日'].map(sea_f)*df.apply(lambda r:gate_weights.get(r['枠'],1),axis=1)
# 正規化＆Z化
metrics=['Raw','Ave-3F','単勝オッズ','best_dist_time','斤量','today_weight','増減']
for m in metrics: df[f'Z_{m}']=(df[m]-df[m].mean())/df[m].std(ddof=1)
# 合成
comp={'Z_Raw':8,'Z_Ave-3F':2,'Z_単勝オッズ':1,'Z_best_dist_time':w_best,'Z_斤量':w_jin,'Z_today_weight':w_jin,'Z_増減':1,'Z_gate':weight_gate}
df['total_z']=sum(df[k]*w for k,w in comp.items())/sum(comp.values())
# まとめ
summary=df.groupby('馬名').agg(mean_z=('total_z','mean'),std_z=('total_z','std'),RawBase_mean=('RawBase','mean')).reset_index()
summary['std_z']=summary['std_z'].fillna(0)
mn,mx=summary['mean_z'].min(),summary['mean_z'].max()
summary['偏差値']=30+(summary['mean_z']-mn)/(mx-mn)*40
rb_min,rb_max=summary['RawBase_mean'].min(),summary['RawBase_mean'].max()
summary['RawBase偏差値']=30+(summary['RawBase_mean']-rb_min)/(rb_max-rb_min)*40
summary['バランス']=weight_z*summary['偏差値']+weight_rb*summary['RawBase偏差値']+weight_gate*((summary['枠']-4.5)/3.5)-summary['std_z']
# 出力
st.subheader("本日の予想6頭")
top6=summary.nlargest(6,'バランス').reset_index(drop=True)
tag={1:'◎',2:'〇',3:'▲',4:'△',5:'△',6:'△'}
top6['印']=top6.index.map(lambda i:tag[i+1])
st.table(top6[['印','馬名','偏差値','RawBase偏差値','バランス']])
