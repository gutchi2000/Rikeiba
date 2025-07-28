import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_path = "ipaexg.ttf"  # アプリフォルダに配置
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

st.title("競馬スコア分析アプリ（標準化＆重み付け版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み & 必要列抽出 ---
df = pd.read_excel(uploaded_file)
st.subheader('アップロードデータ プレビュー')
st.write(df.head(5))
st.write('列見出し:', df.columns.tolist())
# 必要カラム定義
required = ["馬名","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"必要な列が不足しています: {missing}")
    st.stop()
# 列抽出 & 型変換
df = df[required].copy()
for c in ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df[required].copy()
for c in ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- 基本指標計算 ---
# グレード点数マップ
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# 生スコア & 各正規化値を計算
def compute_metrics(df):
    # 元スコア
    df['raw'] = df.apply(lambda r: GRADE_SCORE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
    # 正規化 raw_norm
    denom = GP_MAX*df['頭数'] - GP_MIN
    df['raw_norm'] = (df['raw'] - GP_MIN) / denom
    # up3_norm
    df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
    # odds_norm
    df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
    # jin_norm (斤量軽いほど高)
    jin_max, jin_min = df['斤量'].max(), df['斤量'].min()
    df['jin_norm'] = (jin_max - df['斤量']) / (jin_max - jin_min)
    # wdiff_norm (増減少ないほど高)
    mean_w = df['増減'].abs().mean()
    df['wdiff_norm'] = 1 - df['増減'].abs() / mean_w
    return df

df = compute_metrics(df)

# --- 標準化 (Z-score) ---
for col in ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']:
    mu, sigma = df[col].mean(), df[col].std(ddof=0)
    df[f'Z_{col}'] = (df[col] - mu) / sigma

# --- 重み付け合成 ---
weights = {'Z_raw_norm':8, 'Z_up3_norm':2, 'Z_odds_norm':1, 'Z_jin_norm':1, 'Z_wdiff_norm':1}
total_w = sum(weights.values())
df['total_z'] = sum(df[k]*w for k,w in weights.items()) / total_w

# --- 偏差値化 ---
mu_t, sigma_t = df['total_z'].mean(), df['total_z'].std(ddof=0)
df['総合偏差値'] = 50 + 10*(df['total_z'] - mu_t) / sigma_t

# --- 出力 ---
# 馬名ごとに平均した総合偏差値を使い上位6頭を選出
df_avg = df.groupby('馬名')['総合偏差値'].mean().reset_index()
df_avg.columns = ['馬名','平均総合偏差値']
top6 = df_avg.nlargest(6, '平均総合偏差値')
st.subheader('平均総合偏差値 上位6頭')
st.write(top6)

# 棒グラフ
st.subheader('総合偏差値 上位6頭（棒グラフ）')
fig, ax = plt.subplots(figsize=(8,5))
import seaborn as sns
sns.barplot(x='総合偏差値', y='馬名', data=top6, ax=ax)
ax.set_xlabel('総合偏差値')
ax.set_ylabel('馬名')
st.pyplot(fig)

# 散布図
st.subheader('調子(総合偏差値)×安定性(偏差値標準偏差)')
# 馬別「総合偏差値」の平均と標準偏差を算出
df_out = df.groupby('馬名')['総合偏差値'].agg(['mean','std']).reset_index()
# 列名をわかりやすく変更
df_out.columns = ['馬名','mean_z','std_z']
# 四象限プロットの準備
fig2, ax2 = plt.subplots(figsize=(10,6))
# 閾値設定: 調子と安定性を平均+σに
mu_z, sigma_z = df_out['mean_z'].mean(), df_out['mean_z'].std()
x0 = mu_z + sigma_z
mu_s, sigma_s = df_out['std_z'].mean(), df_out['std_z'].std()
y0 = mu_s + sigma_s
xmin, xmax = df_out['mean_z'].min(), df_out['mean_z'].max()
ymin, ymax = df_out['std_z'].min(), df_out['std_z'].max()
# 背景ゾーン塗り分け
ax2.fill_betweenx([y0, ymax], xmin, x0, color='#a6cee3', alpha=0.3)
ax2.fill_betweenx([y0,ymax],x0,xmax,color='#fb9a99',alpha=0.3)
ax2.fill_betweenx([ymin,y0],xmin,x0,color='#b2df8a',alpha=0.3)
ax2.fill_betweenx([ymin,y0],x0,xmax,color='#fdbf6f',alpha=0.3)
ax2.axvline(x0,linestyle='--',color='gray')
ax2.axhline(y0,linestyle='--',color='gray')
# points/labels
ax2.scatter(df_out['mean_z'],df_out['std_z'],color='black')
for _,r in df_out.iterrows(): ax2.text(r['mean_z'],r['std_z'],r['馬名'],fontsize=8)
# labels
ax2.text((xmin+x0)/2,(y0+ymax)/2,'軽視',ha='center',va='center')
ax2.text((x0+xmax)/2,(y0+ymax)/2,'抑え穴',ha='center',va='center')
ax2.text((xmin+x0)/2,(ymin+y0)/2,'堅軸',ha='center',va='center')
ax2.text((x0+xmax)/2,(ymin+y0)/2,'本命',ha='center',va='center')
ax2.set_xlabel('総合偏差値')
ax2.set_ylabel('安定性(std_z)')
st.pyplot(fig2)

# テーブル
st.subheader('馬別スコア一覧 (全馬21頭)')
# df_out は馬名ごとに mean_z, std_z を持つのでこれを利用
# 総合偏差値の降順で全馬を表示
table = df_out[['馬名']].copy()
# map back 総合偏差値 from df
composite = df.groupby('馬名')['総合偏差値'].mean().reset_index()
composite.columns = ['馬名','総合偏差値']
result = composite.merge(df_out[['馬名']], on='馬名')
st.dataframe(result.sort_values('総合偏差値', ascending=False).reset_index(drop=True))
