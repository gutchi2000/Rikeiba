import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_path = "ipaexg.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

st.title("競馬スコア分析アプリ（完成版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み & プレビュー ---
df = pd.read_excel(uploaded_file)
st.subheader('データプレビュー')
st.write(df.head())

# --- 必要列チェック ---
required = ["馬名","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"必要な列が不足しています: {missing}")
    st.stop()

# --- 列抽出 & 型変換 ---
df = df[required].copy()
num_cols = ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- 指標計算 ---
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# 生スコアと正規化
N = df['頭数']
P = df['確定着順']
df['raw'] = df.apply(lambda r: GRADE_SCORE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['raw_norm'] = (df['raw'] - GP_MIN) / (GP_MAX*df['頭数'] - GP_MIN)
# 上がり3F
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
# オッズ
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
# 斤量（軽いほど高）
jin_max, jin_min = df['斤量'].max(), df['斤量'].min()
df['jin_norm'] = (jin_max - df['斤量']) / (jin_max - jin_min)
# 体重増減
mean_w = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / mean_w

# --- Zスコア標準化 ---
for col in ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']:
    mu, sigma = df[col].mean(), df[col].std(ddof=0)
    df[f'Z_{col}'] = (df[col] - mu) / sigma

# --- 重み付け合成と偏差値化 ---
weights = {'Z_raw_norm':8, 'Z_up3_norm':2, 'Z_odds_norm':1, 'Z_jin_norm':1, 'Z_wdiff_norm':1}
total_w = sum(weights.values())
df['total_z'] = sum(df[k]*w for k,w in weights.items()) / total_w
mu_t, sigma_t = df['total_z'].mean(), df['total_z'].std(ddof=0)
df['総合偏差値'] = 50 + 10*(df['total_z'] - mu_t) / sigma_t

# --- 馬別平均偏差値 ---
df_avg = df.groupby('馬名')['総合偏差値'].mean().reset_index()
df_avg.columns = ['馬名','平均総合偏差値']

# --- 表示: 全馬偏差値一覧 ---
st.subheader('全馬 偏差値一覧')
st.dataframe(df_avg.sort_values('平均総合偏差値', ascending=False).reset_index(drop=True))

# --- 表示: 上位6頭 ---
top6 = df_avg.nlargest(6,'平均総合偏差値')
st.subheader('平均偏差値 上位6頭')
st.write(top6)

# --- 棒グラフ ---
fig1, ax1 = plt.subplots(figsize=(8,5))
import seaborn as sns
sns.barplot(x='平均総合偏差値', y='馬名', data=top6, ax=ax1)
ax1.set_xlabel('平均総合偏差値')
ax1.set_ylabel('馬名')
st.pyplot(fig1)

# --- 散布図: 調子×安定性 ---
df_out = df.groupby('馬名')['総合偏差値'].agg(['mean','std']).reset_index()
df_out.columns = ['馬名','mean_z','std_z']
fig2, ax2 = plt.subplots(figsize=(10,6))
# 閾値: 平均+σ\ nx0 = df_out['mean_z'].mean() + df_out['mean_z'].std()
y0 = df_out['std_z'].mean() + df_out['std_z'].std()
xmin,xmax=df_out['mean_z'].min(),df_out['mean_z'].max()
ymin,ymax=df_out['std_z'].min(),df_out['std_z'].max()
# 背景
ax2.fill_betweenx([y0,ymax],xmin,x0,color='#a6cee3',alpha=0.3)
ax2.fill_betweenx([ymin,y0],xmin,x0,color='#b2df8a',alpha=0.3)
ax2.fill_betweenx([y0,ymax],x0,xmax,color='#fb9a99',alpha=0.3)
ax2.fill_betweenx([ymin,y0],x0,xmax,color='#fdbf6f',alpha=0.3)
ax2.axvline(x0,linestyle='--',color='gray')
ax2.axhline(y0,linestyle='--',color='gray')
ax2.scatter(df_out['mean_z'],df_out['std_z'],color='black',s=30)
for _,r in df_out.iterrows(): ax2.text(r['mean_z'],r['std_z'],r['馬名'],fontsize=8)
ax2.text((xmin+x0)/2,(y0+ymax)/2,'軽視',ha='center',va='center')
ax2.text((xmin+x0)/2,(ymin+ y0)/2,'堅軸',ha='center',va='center')
ax2.text((x0+xmax)/2,(y0+ymax)/2,'抑え穴',ha='center',va='center')
ax2.text((x0+xmax)/2,(ymin+ y0)/2,'本命',ha='center',va='center')
ax2.set_xlabel('平均総合偏差値')
ax2.set_ylabel('安定性(偏差値標準偏差)')
st.pyplot(fig2)

# --- Excel ダウンロード ---
download_df = df_avg.sort_values('平均総合偏差値', ascending=False)
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    download_df.to_excel(writer, index=False, sheet_name='偏差値一覧')
processed_data = output.getvalue()
st.download_button(label='偏差値一覧をExcelでダウンロード',
                   data=processed_data,
                   file_name='score_list.xlsx',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
