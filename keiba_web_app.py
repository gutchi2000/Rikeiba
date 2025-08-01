import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# --- ヘルパー関数 ---
def z_score(s: pd.Series) -> pd.Series:
    return 50 + 10 * (s - s.mean()) / s.std(ddof=0)

def season_of(m: int) -> str:
    if 3 <= m <= 5: return '春'
    if 6 <= m <= 8: return '夏'
    if 9 <= m <= 11: return '秋'
    return '冬'

# --- サイドバー設定 ---
st.sidebar.header("パラメータ設定")
lambda_part = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)
hist_weight = 1 - orig_weight
st.sidebar.subheader("属性重み")
gender_w = {g: st.sidebar.slider(g, 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
style_w  = {s: st.sidebar.slider(s, 0.0, 2.0, 1.0) for s in ['逃げ','先行','差し','追込']}
season_w = {s: st.sidebar.slider(s, 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
age_w      = st.sidebar.number_input("年齢重み", 0.0, 5.0, 1.0)
frame_w    = {str(i): st.sidebar.slider(f"{i}枠",0.0,2.0,1.0) for i in range(1,9)}
besttime_w = st.sidebar.slider("ベストタイム重み",0.0,2.0,1.0)
weight_coeff = st.sidebar.slider("斤量効果強度",0.0,2.0,1.0)
total_budget = st.sidebar.slider("合計予算",500,50000,10000,500)
scenario     = st.sidebar.selectbox("シナリオ",['通常','ちょい余裕','余裕'])

# --- メイン ---
st.title("競馬予想アプリ")
# ファイルアップロード
e1=st.file_uploader("Excel (成績) ",type='xlsx')
e2=st.file_uploader("HTML (血統)",type='html')
if not e1 or not e2: st.stop()

# シート読み込み
df_score = pd.read_excel(e1, sheet_name=0)
sheet2   = pd.read_excel(e1, sheet_name=1)
attrs = sheet2.iloc[:, [0,2,5,3,4]].copy()
attrs.columns = ['枠','馬名','脚質','性別','年齢']
st.subheader("馬一覧と入力")
edited = st.data_editor(
    attrs,
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=list(style_w.keys())),
        '枠': st.column_config.NumberColumn('枠')
    }, num_rows='static'
)
# 斤量取得 and merge
df_wt = df_score[['馬名','斤量']].drop_duplicates().rename(columns={'斤量':'input_wt'})
horses = pd.merge(edited, df_wt, on='馬名', how='left')
# HTML パース
cont=e2.read().decode(errors='ignore'); rows=re.findall(r'<tr[\s\S]*?<\/tr>',cont)
blood=[]
for r in rows:
    c=re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>',r)
    if len(c)>=2: blood.append((re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip()))
blood_df=pd.DataFrame(blood,columns=['馬名','血統'])
# merge all
df_score = df_score.merge(horses, on='馬名').merge(blood_df, on='馬名', how='left')
keys = st.text_area("血統系統",height=100).splitlines(); bp=st.slider("血統ボーナス",0,20,5)
avg_wt = horses['input_wt'].mean(); style_map = dict(zip(horses['馬名'],horses['脚質']))

def calc_score(r):
    gp={'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
    g=gp.get(r['クラス名'],1)
    raw=g*(r['頭数']+1-r['確定着順'])+lambda_part*g
    sw=season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw=gender_w.get(r['性別'],1); stw=style_w.get(style_map.get(r['馬名'],''),1)
    fw=frame_w.get(str(r['枠']),1); aw=age_w; bt=besttime_w
    wt=r.get('input_wt',None); wfac=(wt/avg_wt)**weight_coeff if wt and avg_wt>0 else 1
    bonus=bp if any(k in str(r.get('血統','')) for k in keys) else 0
    return raw*sw*gw*stw*fw*aw*bt*wfac+bonus

df_score['score_raw']=df_score.apply(calc_score,axis=1)
df_score['score_norm']=(df_score['score_raw']-df_score['score_raw'].min())/(
    df_score['score_raw'].max()-df_score['score_raw'].min())*100
df_agg = df_score.groupby('馬名')['score_norm'].agg(['mean','std']).reset_index()
df_agg.columns=['馬名','AvgZ','Stdev']; df_agg['Stability']=-df_agg['Stdev']; df_agg['RankZ']=z_score(df_agg['AvgZ'])

# 改善した散布図
st.subheader("散布図")
fig, ax = plt.subplots()
points = ax.scatter(df_agg['RankZ'], df_agg['Stability'])
for x, y, name in zip(df_agg['RankZ'], df_agg['Stability'], df_agg['馬名']):
    ax.annotate(name, (x, y), xytext=(5,5), textcoords='offset points', fontsize=8)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
avg_st = df_agg['Stability'].mean()
ax.axvline(50, color='gray'); ax.axhline(avg_st, color='gray')
ax.text(60, avg_st+0.1, '一発警戒'); ax.text(40, avg_st+0.1, '警戒必須')
ax.text(60, avg_st-0.1, '鉄板級'); ax.text(40, avg_st-0.1, '堅実型')
plt.tight_layout()
st.pyplot(fig)

# 上位6頭 and bets... (省略)
