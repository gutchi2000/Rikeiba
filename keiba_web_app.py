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
df_score = pd.read_excel(e1, sheet_name=0)  # 計算用全レースデータ
sheet2   = pd.read_excel(e1, sheet_name=1)  # 出走馬リスト・属性
# シート2から必要列を位置指定で抽出
attrs = sheet2.iloc[:, [0,2,5,3,4]].copy()
attrs.columns = ['枠','馬名','脚質','性別','年齢']
# 馬一覧入力テーブル
st.subheader("馬一覧と入力")
edited = st.data_editor(
    attrs,
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=list(style_w.keys())),
        '枠': st.column_config.NumberColumn('枠')
    }, num_rows='static'
)
# 斤量はsheet0から取得
df_wt = df_score[['馬名','斤量']].drop_duplicates()
# 編集結果と斤量結合
horses = pd.merge(edited, df_wt, on='馬名', how='left')

# 血統HTML解析
cont=e2.read().decode(errors='ignore')
rows=re.findall(r'<tr[\s\S]*?<\/tr>',cont)
blood=[]
for r in rows:
    c=re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>',r)
    if len(c)>=2:
        name=re.sub(r'<.*?>','',c[0]).strip()
        kin =re.sub(r'<.*?>','',c[1]).strip()
        blood.append((name,kin))
blood_df=pd.DataFrame(blood,columns=['馬名','血統'])
# 合体
df_score = df_score.merge(horses, on='馬名').merge(blood_df, on='馬名', how='left')

# 血統キーワード
keys = st.text_area("血統系統",height=100).splitlines()
bp   = st.slider("血統ボーナス",0,20,5)

# 平均斤量算
avg_wt = horses['斤量'].mean()
# スタイルマップ
style_map = dict(zip(horses['馬名'],horses['脚質']))

# スコア計算関数
def calc_score(r):
    gp= {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
          '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
    g=gp.get(r['クラス名'],1)
    raw=g*(r['頭数']+1-r['確定着順'])+lambda_part*g
    sw=season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw=gender_w.get(r['性別'],1)
    stw=style_w.get(style_map.get(r['馬名'],''),1)
    fw=frame_w.get(str(r['枠']),1)
    aw=age_w; bt=besttime_w
    wt=r['斤量']; wfac=(wt/avg_wt)**weight_coeff if avg_wt>0 else 1
    bonus=bp if any(k in str(r['血統']) for k in keys) else 0
    return raw*sw*gw*stw*fw*aw*bt*wfac+bonus

# 全レースに適用
df_score['score_raw']=df_score.apply(calc_score,axis=1)
df_score['score_norm']=(df_score['score_raw']-df_score['score_raw'].min())/(df_score['score_raw'].max()-df_score['score_raw'].min())*100
# 馬ごと統計
df_agg=df_score.groupby('馬名')['score_norm'].agg(['mean','std']).reset_index()
df_agg.columns=['馬名','AvgZ','Stdev']; df_agg['Stability']=-df_agg['Stdev']; df_agg['RankZ']=z_score(df_agg['AvgZ'])

# 散布図 st
st.subheader("散布図")
fig,ax=plt.subplots();ax.scatter(df_agg['RankZ'],df_agg['Stability'])
avg_st=df_agg['Stability'].mean(); ax.axvline(50); ax.axhline(avg_st)
ax.text(60,avg_st+0.1,'一発警戒'); ax.text(40,avg_st+0.1,'警戒必須')
ax.text(60,avg_st-0.1,'鉄板級'); ax.text(40,avg_st-0.1,'堅実型')
st.pyplot(fig)

# 上位6頭
top6=df_agg.sort_values('RankZ',ascending=False).head(6)
top6['印']=['◎','〇','▲','☆','△','△']
st.table(top6[['馬名','印']])

# 買い目 and 配分 
p1=total_budget*0.25; p2=total_budget*0.75; rem=total_budget-p1-p2
parts={'通常':['馬連','ワイド','馬単'],'ちょい余裕':['馬連','ワイド','馬単','三連複'],'余裕':['馬連','ワイド','馬単','三連複','三連単']}[scenario]
bet={x:rem/len(parts) for x in parts}
st.write(f"単勝:{p1:.0f}円 複勝:{p2:.0f}円"); st.table(pd.DataFrame.from_dict(bet,orient='index',columns=['金額']))

# 買い目例
st.write("単勝:",top6.iloc[0]['馬名'])
st.write("複勝:",top6.iloc[1]['馬名'])
st.write("馬連:",f"{top6.iloc[0]['馬名']}-{top6.iloc[1]['馬名']}")
st.write("三連複:",f"{top6.iloc[0]['馬名']}-{','.join(top6.iloc[1:5]['馬名'])}")
st.write("三連単:",f"{top6.iloc[0]['馬名']}軸→{','.join(top6.iloc[1:6]['馬名'])}")
