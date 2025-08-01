import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re

# --- ヘルパー関数 ---
def z_score(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    return 50 + 10 * (series - mu) / sigma

def season_of(month: int) -> str:
    if 3 <= month <= 5:
        return '春'
    if 6 <= month <= 8:
        return '夏'
    if 9 <= month <= 11:
        return '秋'
    return '冬'

# --- サイドバー設定 ---
st.sidebar.header("パラメータ設定")
lambda_part = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)
hist_weight = 1.0 - orig_weight

st.sidebar.subheader("性別重み")
gender_w = {
    '牡': st.sidebar.slider('牡馬', 0.0, 2.0, 1.0, 0.1),
    '牝': st.sidebar.slider('牝馬', 0.0, 2.0, 1.0, 0.1),
    'セ': st.sidebar.slider('せん馬', 0.0, 2.0, 1.0, 0.1)
}

st.sidebar.subheader("脚質重み")
style_w = {
    '逃げ': st.sidebar.slider('逃げ', 0.0, 2.0, 1.0, 0.1),
    '先行': st.sidebar.slider('先行', 0.0, 2.0, 1.0, 0.1),
    '差し': st.sidebar.slider('差し', 0.0, 2.0, 1.0, 0.1),
    '追込': st.sidebar.slider('追込', 0.0, 2.0, 1.0, 0.1)
}

st.sidebar.subheader("四季重み")
season_w = {s: st.sidebar.slider(f'{s}', 0.0, 2.0, 1.0, 0.1) for s in ['春','夏','秋','冬']}

age_w = st.sidebar.number_input("年齢重み", 0.0, 5.0, 1.0, 0.1)

st.sidebar.subheader("枠順重み")
frame_w = {str(i): st.sidebar.slider(f'{i}枠', 0.0, 2.0, 1.0, 0.1) for i in range(1,9)}

besttime_w = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0, 0.1)
weight_coeff = st.sidebar.slider("斤量効果強度", 0.0, 2.0, 1.0, 0.1)

total_budget = st.sidebar.slider("合計予算 (円)", 500, 50000, 10000, 500)
scenario = st.sidebar.selectbox("シナリオ", ['通常', 'ちょい余裕', '余裕'])

# --- メイン画面 ---
st.title("競馬予想アプリ")

# ファイルアップロード
e1 = st.file_uploader("Excel (成績データ)", type=['xlsx'])
e2 = st.file_uploader("HTML (血統データ)", type=['html'])
if not e1 or not e2:
    st.stop()

# データ読み込み
df1 = pd.read_excel(e1, sheet_name=0)  # シート1：斤量など計算用
sheet2 = pd.read_excel(e1, sheet_name=1)  # シート2：枠・馬名・脚質・性別・年齢
# 位置指定で必要列を取得
attrs = sheet2.iloc[:, [0,2,5,3,4]].copy()  # 馬番含む並びなら調整
attrs.columns = ['枠','馬名','脚質','性別','年齢']
# マージして df が斤量を含む
df = pd.merge(df1, attrs, on='馬名', how='inner')

# 血統HTML解析
cont = e2.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont)
blood = []
for r in rows:
    c = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)
    if len(c)>=2:
        name = re.sub(r'<.*?>','',c[0]).strip()
        kin = re.sub(r'<.*?>','',c[1]).strip()
        blood.append((name,kin))
blood_df = pd.DataFrame(blood, columns=['馬名','血統'])
df = df.merge(blood_df, on='馬名', how='left')

# 血統キーワード
keys = st.text_area("血統系統", height=100).splitlines()
bp = st.slider("血統ボーナス点数", 0, 20, 5)

# スコア計算
# 平均斤量
avg_wt = df['斤量'].mean()
# スタイルマップ
style_map = dict(zip(df['馬名'], df['脚質']))

def calc_score(r):
    GP = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
          '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
    gp = GP.get(r['クラス名'],1)
    raw = gp * (r['頭数']+1-r['確定着順']) + lambda_part*gp
    sw = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw = gender_w.get(r['性別'],1)
    stw = style_w.get(style_map.get(r['馬名'],''),1)
    fw = frame_w.get(str(r['枠']),1)
    aw = age_w
    bt = besttime_w
    wt = r['斤量']
    wfac = (wt/avg_wt)**weight_coeff if avg_wt>0 else 1
    bonus = bp if any(k in str(r['血統']) for k in keys) else 0
    return raw*sw*gw*stw*fw*aw*bt*wfac + bonus

df['score_raw'] = df.apply(calc_score,axis=1)
df['score_norm'] = (df['score_raw']-df['score_raw'].min())/(df['score_raw'].max()-df['score_raw'].min())*100

# 馬統計と散布図
agg = df.groupby('馬名')['score_norm'].agg(['mean','std']).reset_index()
agg.columns=['馬名','AvgZ','Stdev']
agg['Stability']=-agg['Stdev']
agg['RankZ']=z_score(agg['AvgZ'])

fig,ax = plt.subplots()
ax.scatter(agg['RankZ'],agg['Stability'])
avg_st=agg['Stability'].mean()
ax.axvline(50);ax.axhline(avg_st)
ax.text(60,avg_st+0.1,'一発警戒');ax.text(40,avg_st+0.1,'警戒必須')
ax.text(60,avg_st-0.1,'鉄板級');ax.text(40,avg_st-0.1,'堅実型')
st.pyplot(fig)

# 上位6頭印付け
top6=agg.sort_values('RankZ',ascending=False).head(6)
top6['印']=['◎','〇','▲','☆','△','△']
st.table(top6[['馬名','印']])

# 配分と買い目
pur1=total_budget*0.25;pur2=total_budget*0.75;rem=total_budget-pur1-pur2
parts={'通常':['馬連','ワイド','馬単'],'ちょい余裕':['馬連','ワイド','馬単','三連複'],'余裕':['馬連','ワイド','馬単','三連複','三連単']}[scenario]
b_share={p:rem/len(parts) for p in parts}
st.write(f"単勝:{pur1:.0f}円,複勝:{pur2:.0f}円");st.table(pd.DataFrame.from_dict(b_share,orient='index',columns=['金額']))

# 買い目例
st.write("単勝:",top6.iloc[0]['馬名'])
st.write("複勝:",top6.iloc[1]['馬名'])
st.write("馬連:",f"{top6.iloc[0]['馬名']}-{top6.iloc[1]['馬名']}")
st.write("三連複:",f"{top6.iloc[0]['馬名']}-{','.join(top6.iloc[1:5]['馬名'])}")
st.write("三連単:",f"{top6.iloc[0]['馬名']}軸→{','.join(top6.iloc[1:6]['馬名'])}")
