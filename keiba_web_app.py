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

# --- 入力: 成績データ ---
uploaded_file = st.file_uploader("成績データをアップロード (Excel)", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# --- 入力: 馬名／年齢／脚質（テーブル編集） ---
equine_list = df['馬名'].unique().tolist()
equine_df = pd.DataFrame({'馬名': equine_list, '年齢': [5]*len(equine_list), '脚質': ['差し']*len(equine_list)})
edited = st.data_editor(
    equine_df,
    column_config={
        '年齢': st.column_config.NumberColumn("年齢", min_value=1, max_value=10, step=1),
        '脚質': st.column_config.SelectboxColumn("脚質", options=['逃げ','先行','差し','追込'])
    },
    use_container_width=True,
    key="equine_editor"
)
# マッピング
age_map = dict(zip(edited['馬名'], edited['年齢']))
style_map = dict(zip(edited['馬名'], edited['脚質']))

# --- 入力: 血統表 (HTML) ---
html_file = st.file_uploader("血統表をアップロード (HTML)", type=["html"])

# --- 必要列チェック ---
cols = ["馬名","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
if any(c not in df.columns for c in cols):
    st.error(f"必要列が不足しています: {set(cols)-set(df.columns)}")
    st.stop()

# --- 前処理 ---
df = df[cols].copy()
for c in ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
df.dropna(subset=cols, inplace=True)

# --- 血統評価 ---
def eval_pedigree(mn):
    if html_file:
        tables = pd.read_html(html_file)
        ped = tables[0]
        sire = ped.set_index('馬名').get('父馬', {}).get(mn, '')
        if sire in ["サクラバクシンオー","スウェプトオーヴァーボード"]:
            return 1.2
        if sire in ["マンハッタンカフェ","フジキセキ"]:
            return 1.1
    return 1.0

df['pedigree_factor'] = df['馬名'].map(eval_pedigree)

# --- 脚質評価 ---
def style_factor(mn):
    return {'逃げ':1.2,'先行':1.1,'差し':1.0,'追込':0.9}.get(style_map.get(mn,''),1.0)

df['style_factor'] = df['馬名'].map(style_factor)

# --- 年齢評価 ---
def age_factor(a):
    peak = 5
    return 1 + 0.2 * (1 - abs(a - peak) / peak)

df['年齢'] = df['馬名'].map(lambda mn: age_map.get(mn, 5))
df['age_factor'] = df['年齢'].apply(age_factor)

# --- 基本指標計算 ---
GRADE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,"オープン特別":4,"3勝クラス":3,"2勝クラス":2,"1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

df['raw'] = df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1) * df['pedigree_factor']
df['raw_norm'] = (df['raw'] - GP_MIN) / (GP_MAX * df['頭数'] - GP_MIN)
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
jmax, jmin = df['斤量'].max(), df['斤量'].min()
df['jin_norm'] = (jmax - df['斤量']) / (jmax - jmin)
wmean = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / wmean

# --- Zスコア化 ---
metrics = ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = df[m].apply(lambda x: 0 if sd == 0 else (x - mu) / sd)
for m in ['pedigree_factor','age_factor','style_factor']:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = df[m].apply(lambda x: 0 if sd == 0 else (x - mu) / sd)

# --- 合成偏差値化 ---
weights = {'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,'Z_jin_norm':1,'Z_wdiff_norm':1,
           'Z_pedigree_factor':3,'Z_age_factor':2,'Z_style_factor':2}
total_w = sum(weights.values())
df['total_z'] = sum(df[k]*w for k,w in weights.items()) / total_w
 z = df['total_z']; zmin, zmax = z.min(), z.max()
 df['偏差値'] = 30 + (z - zmin) / (zmax - zmin) * 40

# --- 馬別集計 ---
df_avg = df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
df_avg.columns = ['馬名','平均偏差値','安定性']
df_avg['バランススコア'] = df_avg['平均偏差値'] - df_avg['安定性']

# --- 表示: 馬別評価 ---
st.subheader('馬別 評価一覧')
st.dataframe(df_avg.sort_values('バランススコア',ascending=False))

# --- 上位抽出 ---
top10 = df_avg.nlargest(10,'平均偏差値')['馬名']
st.subheader('偏差値上位10頭')
st.write(top10.tolist())
combined = df_avg[df_avg['馬名'].isin(top10)].nlargest(6,'バランススコア')
st.subheader('偏差値10頭中の安定&調子上位6頭')
st.table(combined)

# --- 可視化: 棒グラフ ---
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x='バランススコア', y='馬名', data=combined, palette='viridis', ax=ax)
st.pyplot(fig)

# --- ベット設定 ---
with st.expander('ベット設定'):
    bet_type = st.selectbox('券種', ['単勝','複勝','馬連','ワイド','三連複','三連単'])
    budget = st.number_input('予算 (円)', min_value=1000, step=1000, value=10000)
    axis = combined['馬名'].iat[0]
    if bet_type in ['単勝','複勝']:
        ratio = {'単勝':0.25,'複勝':0.75}[bet_type]
        amt = (budget * ratio)//100*100
        st.write(f"{bet_type} {axis} に {amt} 円")
    else:
        others = combined['馬名'].tolist()[1:]
        if bet_type in ['馬連','ワイド']:
            combos = [f"{axis}-{h}" for h in others]
        elif bet_type == '三連複':
            f, s, t = combined['馬名'].tolist()[0], combined['馬名'].tolist()[1:3], combined['馬名'].tolist()[3:]
            combos = ["-".join(sorted([f,b,c])) for b in s for c in t]
        else:
            fix = combined['馬名'].tolist()[:4]
            combos = [f"{fix[0]}→{i}→{j}" for i in fix[1:] for j in fix[1:] if i != j]
        amt = (budget // len(combos))//100*100
        st.dataframe(pd.DataFrame({'券種':[bet_type]*len(combos),'組合せ':combos,'金額':[amt]*len(combos)}))
