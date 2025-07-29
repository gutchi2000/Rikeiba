import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="ipaexg.ttf").get_name()

st.title("競馬予想アプリ（完成版）")

# --- 入力 ---
uploaded_file = st.file_uploader("成績データをアップロード (Excel)", type=["xlsx"])
html_file = st.file_uploader("血統表をアップロード (HTML)", type=["html"])
radio_course = st.text_area("コース解説を入力", height=150)
style_map_input = st.text_area("馬の脚質（CSV:馬名,脚質）入力", height=100)
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)
cols = ["馬名","年齢","脚質","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
if any(c not in df.columns for c in cols):
    st.error(f"必要列が不足しています: {set(cols)-set(df.columns)}")
    st.stop()

df = df[cols].copy()
# 型変換
for c in ["年齢","頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
df.dropna(subset=cols, inplace=True)

# --- 血統評価 ---
pedigree_map = {}
if html_file:
    import pandas as _pd
    tables = _pd.read_html(html_file)
    # assume first table has 馬名, 父馬
    ped = tables[0][["馬名","父馬"]]
    # 仮評価: MP系なら1.2、Halo系1.1、その他1.0
    def eval_ped(sire):
        if sire in ["サクラバクシンオー","スウェプトオーヴァーボード"]:
            return 1.2
        if sire in ["マンハッタンカフェ","フジキセキ"]:
            return 1.1
        return 1.0
    ped['pedigree_factor'] = ped['父馬'].map(eval_ped)
    df = df.merge(ped[['馬名','pedigree_factor']], on='馬名', how='left')
else:
    df['pedigree_factor'] = 1.0

# --- 脚質評価 ---
style_map = {}
if style_map_input:
    for line in style_map_input.splitlines():
        mn, sty = line.split(',')
        style_map[mn.strip()] = sty.strip()
def style_factor(mn):
    sty = style_map.get(mn,"")
    return {"逃げ":1.2,"先行":1.1,"差し":1.0,"追込":0.9}.get(sty,1.0)
df['style_factor'] = df['馬名'].map(style_factor)

# --- 年齢評価 ---
def age_factor(a):
    peak=5
    return 1+0.2*(1 - abs(a-peak)/peak)

df['age_factor'] = df['年齢'].apply(age_factor)

# --- 基本指標計算 ---
GRADE={"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,"オープン特別":4,"3勝クラス":3,"2勝クラス":2,"1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN,GP_MAX=1,10
df['raw']=df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']),axis=1)
df['raw']*=df['pedigree_factor']
df['raw_norm']=(df['raw']-GP_MIN)/(GP_MAX*df['頭数']-GP_MIN)
df['up3_norm']=df['Ave-3F']/df['上がり3Fタイム']
df['odds_norm']=1/(1+np.log10(df['単勝オッズ']))
jmax,jmin=df['斤量'].max(),df['斤量'].min()
df['jin_norm']=(jmax-df['斤量'])/(jmax-jmin)
mean_w=df['増減'].abs().mean()
df['wdiff_norm']=1-df['増減'].abs()/mean_w

# --- Zスコア化 ---
metrics=['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']
for m in metrics:
    mu,sd=df[m].mean(),df[m].std(ddof=1)
    df[f'Z_{m}']=df[m].apply(lambda x:0 if sd==0 else (x-mu)/sd)
# pedigree, age, style Z化
for m in ['pedigree_factor','age_factor','style_factor']:
    mu,sd=df[m].mean(),df[m].std(ddof=1)
    df[f'Z_{m}']=df[m].apply(lambda x:0 if sd==0 else (x-mu)/sd)

# --- 合成偏差値化 ---
weights={'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,'Z_jin_norm':1,'Z_wdiff_norm':1,
         'Z_pedigree_factor':3,'Z_age_factor':2,'Z_style_factor':2}
total= sum(weights.values())
df['total_z']=sum(df[k]*w for k,w in weights.items())/total
z=df['total_z'];zmin,zmax=z.min(),z.max()
df['偏差値']=30+(z-zmin)/(zmax-zmin)*40

# --- 馬別集計 ---
df_avg=df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
df_avg.columns=['馬名','平均偏差値','安定性']
df_avg['バランススコア']=df_avg['平均偏差値']-df_avg['安定性']

# --- 表示: 馬別評価 ---
st.subheader('馬別 評価一覧')
st.dataframe(df_avg.sort_values('バランススコア',ascending=False))

# --- 上位抽出 ---
top10=df_avg.nlargest(10,'平均偏差値')['馬名']
st.subheader('偏差値上位10頭')
st.write(top10.tolist())
combined=df_avg[df_avg['馬名'].isin(top10)].nlargest(6,'バランススコア')
st.subheader('偏差値10頭中の安定&調子上位6頭')
st.table(combined)

# --- 可視化 ---
fig,ax=plt.subplots(figsize=(8,6))
sns.barplot(x='バランススコア',y='馬名',data=combined,palette='viridis',ax=ax)
st.pyplot(fig)

# --- 結果ダウンロード ---
buf=BytesIO()
with pd.ExcelWriter(buf,engine='openpyxl') as w:df_avg.to_excel(w,index=False,sheet_name='評価')
st.download_button('評価一覧ダウンロード',buf.getvalue(),'evaluation.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- 完 ---
