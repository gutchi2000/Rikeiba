import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 日本語フォント設定: IPAexゴシックを登録
font_path = "/mnt/data/ipaexg.ttf"
font_manager.fontManager.addfont(font_path)
jp_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = jp_font.get_name()
sns.set(font=jp_font.get_name())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("競馬スコア分析アプリ（完成版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み ---
use_cols = ["馬名","頭数","クラス名","確定着順",
            "上がり3Fタイム","Ave-3F",
            "馬場状態","馬体重","増減","斤量","単勝オッズ"]
try:
    df = pd.read_excel(uploaded_file, sheet_name=0, usecols=use_cols)
    df.columns = ["馬名","頭数","グレード","着順",
                  "上がり3F","Ave-3F",
                  "track_condition","weight","weight_diff","jinryo","odds"]
except ValueError:
    df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    df = df.iloc[:, :len(use_cols)]
    df.columns = ["馬名","頭数","グレード","着順",
                  "上がり3F","Ave-3F",
                  "track_condition","weight","weight_diff","jinryo","odds"]
# 列名を統一
df.columns = ["馬名","頭数","グレード","着順",
              "上がり3F","Ave-3F",
              "track_condition","weight","weight_diff","jinryo","odds"]

# --- 型変換 ---
for c in ["頭数","着順","上がり3F","Ave-3F","weight","weight_diff","jinryo","odds"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- パラメータ ---
GRADE_SCORE = {"GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5,
               "オープン特別":4, "3勝クラス":3, "2勝クラス":2,
               "1勝クラス":1, "新馬":1, "未勝利":1}
GP_MIN, GP_MAX = 1, 10

# --- スコア計算 ---
def calc_score(row):
    N = row['頭数']
    p = row['着順']
    GP = GRADE_SCORE.get(row['グレード'], 1)
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN) if GP_MAX*N - GP_MIN != 0 else 0

    up3 = row['上がり3F']
    ave3 = row['Ave-3F']
    up3_norm = ave3 / up3 if up3 > 0 else 0

    odds = row['odds']
    odds_norm = 1 / (1 + np.log10(odds)) if odds > 1 else 1

    jin = row['jinryo']
    jin_min, jin_max = df['jinryo'].min(), df['jinryo'].max()
    jin_norm = (jin_max - jin) / (jin_max - jin_min) if jin_max > jin_min else 1

    wdiff = row['weight_diff']
    wt_mean = df['weight'].mean()
    wdiff_norm = 1 - abs(wdiff) / wt_mean if wt_mean > 0 else 0

    # 合計重み 8+2+1+1+1 = 13
    score = raw_norm*8 + up3_norm*2 + odds_norm + jin_norm + wdiff_norm
    return score / 13 * 100

# スコア適用
df['Score'] = df.apply(calc_score, axis=1)

# --- 偏差値計算 ---
df_avg = df.groupby('馬名')['Score'].agg(['mean','std']).reset_index()
df_avg.columns = ['馬名','平均スコア','標準偏差']
mu, sigma = df_avg['平均スコア'].mean(), df_avg['平均スコア'].std()
df_avg['偏差値'] = 50 + 10 * (df_avg['平均スコア'] - mu) / sigma

# --- 棒グラフ: 偏差値上位6頭 ---
top6 = df_avg.nlargest(6, '偏差値')
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='偏差値', y='馬名', data=top6, ax=ax)
ax.set_title('偏差値 上位6頭')
ax.set_xlabel('偏差値')
ax.set_ylabel('馬名')
st.pyplot(fig)

# --- 散布図: 調子×安定性 ---
fig2, ax2 = plt.subplots(figsize=(10,6))
x0 = df_avg['偏差値'].mean()
y0 = df_avg['標準偏差'].mean()
xmin, xmax = df_avg['偏差値'].min(), df_avg['偏差値'].max()
ymin, ymax = df_avg['標準偏差'].min(), df_avg['標準偏差'].max()
# 背景ゾーン
ax2.fill_betweenx([ymin, y0], xmin, x0, color='green', alpha=0.2)
ax2.fill_betweenx([ymin, y0], x0, xmax, color='yellow', alpha=0.2)
ax2.fill_betweenx([y0, ymax], xmin, x0, color='blue', alpha=0.2)
ax2.fill_betweenx([y0, ymax], x0, xmax, color='red', alpha=0.2)
# 基準線
ax2.axvline(x0, color='gray', linestyle='--')
ax2.axhline(y0, color='gray', linestyle='--')
# 散布点と注釈
ax2.scatter(df_avg['偏差値'], df_avg['標準偏差'], color='black', s=20)
for _, r in df_avg.iterrows():
    ax2.text(r['偏差値'], r['標準偏差'] + 0.1, r['馬名'], fontsize=8, ha='center')
# 四象限ラベル
ax2.text((x0 + xmax) / 2, (y0 + ymin) / 2, '本命候補', ha='center')
ax2.text((x0 + xmax) / 2, (y0 + ymax) / 2, '抑え・穴狙い', ha='center')
ax2.text((xmin + x0) / 2, (y0 + ymax) / 2, '軽視ゾーン', ha='center')
ax2.text((xmin + x0) / 2, (y0 + ymin) / 2, '堅軸ゾーン', ha='center')
ax2.set_xlabel('偏差値')
ax2.set_ylabel('標準偏差')
ax2.set_title('調子×安定性')
st.pyplot(fig2)

# --- テーブル表示 ---
st.subheader('馬別スコア一覧')
st.dataframe(df_avg)
