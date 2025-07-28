import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# seaborn不要なので削除

# --- 日本語フォント設定 ---
font_path = "ipaexg.ttf"  # アプリフォルダに配置
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

st.title("競馬スコア分析アプリ（騎手有無対応版 改良）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み ---
df = pd.read_excel(uploaded_file)
cols = df.columns.tolist()

# 騎手あり/なし判定
has_jockey = '騎手' in cols

# 必要列設定
base_cols = ["馬名","頭数","クラス名","確定着順",
             "上がり3Fタイム","Ave-3F",
             "馬場状態","馬体重","増減","斤量","単勝オッズ"]
if has_jockey:
    base_cols.insert(1, '騎手')

# 列抽出／ヘッダー名がない場合は位置で指定
missing = set(base_cols) - set(cols)
if not missing:
    df = df[base_cols]
else:
    st.warning(f"列名が見つかりませんでした。先頭{len(base_cols)}列を使用します。期待列: {missing}")
    df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    df = df.iloc[:, :len(base_cols)]
    df.columns = base_cols

# 型変換
for c in ["頭数","確定着順","上がり3Fタイム","Ave-3F","馬体重","増減","斤量","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# グレードマップ
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# スコア計算
def calc_score(r):
    N, p = r['頭数'], r['確定着順']
    GP = GRADE_SCORE.get(r['クラス名'],1)
    raw = GP * (N + 1 - p)
    denom = GP_MAX*N - GP_MIN
    raw_norm = (raw - GP_MIN)/denom if denom>0 else 0
    up3_norm = r['Ave-3F']/r['上がり3Fタイム'] if r['上がり3Fタイム']>0 else 0
    odds_norm = 1/(1+np.log10(r['単勝オッズ'])) if r['単勝オッズ']>1 else 1
    jin_min, jin_max = df['斤量'].min(), df['斤量'].max()
    jin_norm = (jin_max - r['斤量'])/(jin_max-jin_min) if jin_max>jin_min else 0
    wdiff_norm = 1 - abs(r['増減'])/df['馬体重'].mean() if df['馬体重'].mean()>0 else 0
    bonus = 0.1 if has_jockey else 0
    # 全体重み合計: 8+2+1+1+1+0.1=13.1 -> 正規化
    total = 13.1
    s = raw_norm*8 + up3_norm*2 + odds_norm*1 + jin_norm*1 + wdiff_norm*1 + bonus
    return s/total*100

# スコア適用
df['Score'] = df.apply(calc_score, axis=1)

# 馬別偏差値算出
df_out = df.groupby('馬名')['Score'].agg(['mean','std']).reset_index()
df_out.columns=['馬名','平均スコア','標準偏差']
mu, sigma = df_out['平均スコア'].mean(), df_out['平均スコア'].std()
df_out['偏差値'] = 50 + 10*(df_out['平均スコア']-mu)/sigma

# 偏差値上位6頭
st.subheader('偏差値 上位6頭表')
st.write(df_out.nlargest(6,'偏差値')[['馬名','偏差値']])

# 棒グラフ
st.subheader('偏差値 上位6頭（棒グラフ）')
import seaborn as sns
fig1, ax1 = plt.subplots(figsize=(8,5))
top6 = df_out.nlargest(6,'偏差値')
sns.barplot(x='偏差値', y='馬名', data=top6, ax=ax1)
ax1.set_xlabel('偏差値')
ax1.set_ylabel('馬名')
st.pyplot(fig1)

# 散布図: 調子×安定性（散布図）
st.subheader('調子×安定性（散布図）')
fig2, ax2 = plt.subplots(figsize=(10,6))

# 四象限の境界を調整: 本命候補ゾーンを狭めるため、垂直線は平均偏差値+標準偏差に設定
mu_val = df_out['偏差値'].mean()
sigma_val = df_out['偏差値'].std()
x0 = mu_val + sigma_val  # 標準偏差だけ右にシフト
ny0 = df_out['標準偏差'].mean()

xmin, xmax = df_out['偏差値'].min()-1, df_out['偏差値'].max()+1
ymin, ymax = df_out['標準偏差'].min()-0.5, df_out['標準偏差'].max()+0.5

# 背景拡大: 軽視ゾーン広げる
ax2.fill_betweenx([ny0, ymax], xmin, x0, color='#a6cee3', alpha=0.3)
ax2.fill_betweenx([ymin, ny0], xmin, x0, color='#b2df8a', alpha=0.3)
ax2.fill_betweenx([ny0, ymax], x0, xmax, color='#fb9a99', alpha=0.3)
ax2.fill_betweenx([ymin, ny0], x0, xmax, color='#fdbf6f', alpha=0.3)
ax2.axvline(x0, color='gray', linestyle='--')
ax2.axhline(ny0, color='gray', linestyle='--')

# 散布
ax2.scatter(df_out['偏差値'], df_out['標準偏差'], color='black', s=30)
for _, r in df_out.iterrows():
    ax2.text(r['偏差値'], r['標準偏差']+0.05, r['馬名'], fontsize=8, ha='center')

# ラベル配置中心調整
ax2.text((xmin+x0)/2, (ny0+ymax)/2, '軽視ゾーン', fontsize=12, ha='center')
ax2.text((x0+xmax)/2, (ny0+ymax)/2, '抑え・穴狙い', fontsize=12, ha='center')
ax2.text((xmin+x0)/2, (ymin+ny0)/2, '堅軸ゾーン', fontsize=12, ha='center')
ax2.text((x0+xmax)/2, (ymin+ny0)/2, '本命候補', fontsize=12, ha='center')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('偏差値')
ax2.set_ylabel('標準偏差')
st.pyplot(fig2)
