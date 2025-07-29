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
if not uploaded_file:
    st.stop()

# --- 後から入力: 年齢と脚質 ---
# 馬名一覧取得
equine_list = df['馬名'].unique().tolist()
# 年齢入力
st.subheader('馬ごとの年齢選択')
age_map = {}
for mn in equine_list:
    age_map[mn] = st.selectbox(f"{mn} の年齢", options=list(range(1,11)), index=4, key=f"age_{mn}")
# 脚質入力
st.subheader('馬ごとの脚質選択')
style_map = {}
style_choices = ['逃げ','先行','差し','追込']
for mn in equine_list:
    style_map[mn] = st.selectbox(f"{mn} の脚質", options=style_choices, index=2, key=f"style_{mn}")

# 血統表入力
html_file = st.file_uploader("血統表をアップロード (HTML)", type=["html"])("血統表をアップロード (HTML)", type=["html"])

df = pd.read_excel(uploaded_file)(uploaded_file)
cols = ["馬名","年齢","脚質","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
if any(c not in df.columns for c in cols):
    st.error(f"必要列が不足しています: {set(cols)-set(df.columns)}")
    st.stop()

df = df[cols].copy()
# --- 馬名・年齢・脚質表示 ---
st.subheader('馬名・年齢・脚質一覧')
st.dataframe(df[['馬名','年齢','脚質']].drop_duplicates().reset_index(drop=True))
# --- 計算式 ---
st.markdown('''
**計算式**

- RawScore = GRADE × (頭数 + 1 - 着順)
- RawNorm = (RawScore - GP_min) / (GP_max × 頭数 - GP_min)
- Up3Norm = Ave-3F / 上がり3Fタイム
- OddsNorm = 1 / (1 + log10(単勝オッズ))
- JinNorm = (max(斤量) - 斤量) / (max(斤量) - min(斤量))
- WdiffNorm = 1 - |増減| / 平均|増減|
- 各指標を Z-スコア化し、重み付け合成後、30-70の偏差値にマッピング
''')
# 型変換
for c in ["年齢","頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
df.dropna(subset=cols, inplace=True)

# --- 血統評価 ---
# 血統入力（CSV形式: 馬名,父馬スコア）
pedigree_map_input = st.text_area("血統スコア入力 (CSV: 馬名,スコア)", height=100)
pedigree_map = {}
if pedigree_map_input:
    for line in pedigree_map_input.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts)==2:
            pedigree_map[parts[0]] = float(parts[1])

if html_file:
    import pandas as _pd
    tables = _pd.read_html(html_file)
    ped = tables[0][["馬名","父馬"]]
    # CSV入力があれば優先
    def get_pedigree_factor(mn):
        if mn in pedigree_map:
            return pedigree_map[mn]
        # fallback: MP系 or Halo系
        sire = ped.set_index('馬名').get('父馬').get(mn, '')
        if sire in ["サクラバクシンオー","スウェプトオーヴァーボード"]:
            return 1.2
        if sire in ["マンハッタンカフェ","フジキセキ"]:
            return 1.1
        return 1.0
    ped['pedigree_factor'] = ped['馬名'].map(get_pedigree_factor)
    df = df.merge(ped[['馬名','pedigree_factor']], on='馬名', how='left')
else:
    # CSV入力だけでも利用可能
    df['pedigree_factor'] = df['馬名'].map(lambda mn: pedigree_map.get(mn, 1.0))

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

# --- ベット設定 ---
with st.expander('ベット設定'):
    bet_type = st.selectbox('券種を選択', ['単勝','複勝','馬連','ワイド','三連複','三連単'])
    budget = st.number_input('ベット総予算（円）', min_value=1000, step=1000, value=10000)
    st.write(f"選択: {bet_type}, 予算: {budget:,} 円")
    # 単勝・複勝は一頭軸、他は組み合わせ生成
    axis = combined['馬名'].iloc[0]
    if bet_type in ['単勝','複勝']:
        ratio = {'単勝':0.25,'複勝':0.75}[bet_type]
        amt = (budget * ratio)//100*100
        st.write(f"おすすめ {bet_type} {axis} に {amt:,} 円")
    else:
        combos = []
        others = combined['馬名'].tolist()[1:]
        if bet_type in ['馬連','ワイド']:
            combos = [f"{axis}-{h}" for h in others]
        elif bet_type=='三連複':
            f, s, t = combined['馬名'].tolist()[0], combined['馬名'].tolist()[1:3], combined['馬名'].tolist()[3:]
            combos = ["-".join(sorted([f,b,c])) for b in s for c in t]
        elif bet_type=='三連単':
            fix = combined['馬名'].tolist()[:4]
            combos = [f"{fix[0]}→{i}→{j}" for i in fix[1:] for j in fix[1:] if i!=j]
        cnt = len(combos)
        amt = (budget//cnt)//100*100
        df_bet = pd.DataFrame([{'券種':bet_type,'組合せ':c,'金額':amt} for c in combos])
        st.dataframe(df_bet)

# --- 結果ダウンロード ---
buf=BytesIO()
with pd.ExcelWriter(buf,engine='openpyxl') as w:df_avg.to_excel(w,index=False,sheet_name='評価')
st.download_button('評価一覧ダウンロード',buf.getvalue(),'evaluation.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- 完 ---
