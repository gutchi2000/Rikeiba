import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_path = "ipaexg.ttf"  # 事前にアップロード済みフォント
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

st.title("競馬スコア分析アプリ（完成版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み ---
use_cols = [
    "馬名","頭数","クラス名","確定着順",
    "上がり3Fタイム","Ave-3F",
    "馬場状態","馬体重","増減","斤量","単勝オッズ"
]
try:
    df = pd.read_excel(uploaded_file, sheet_name=0, usecols=use_cols)
    df.columns = [
        "馬名","頭数","グレード","着順",
        "上がり3F","Ave-3F",
        "track_condition","weight","weight_diff","jinryo","odds"
    ]
except ValueError:
    # 列数不足時は先頭から順に割り当て
    df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    df = df.iloc[:, :len(use_cols)]
    df.columns = [
        "馬名","頭数","グレード","着順",
        "上がり3F","Ave-3F",
        "track_condition","weight","weight_diff","jinryo","odds"
    ]

# --- 型変換 ---
for c in ["頭数","着順","上がり3F","Ave-3F","weight","weight_diff","jinryo","odds"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- グレード点数マップ ---
GRADE_SCORE = {
    "GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5,
    "オープン特別":4, "3勝クラス":3, "2勝クラス":2,
    "1勝クラス":1, "新馬":1, "未勝利":1
}
GP_MIN, GP_MAX = 1, 10

# --- スコア計算 ---
def calc_score(row):
    N, p = row['頭数'], row['着順']
    GP = GRADE_SCORE.get(row['グレード'], 1)
    raw = GP * (N + 1 - p)
    denom = GP_MAX * N - GP_MIN
    raw_norm = (raw - GP_MIN) / denom if denom>0 else 0

    # 上がり3F
    up3, ave3 = row['上がり3F'], row['Ave-3F']
    up3_norm = ave3 / up3 if up3>0 else 0

    # 単勝オッズ
    odds = row['odds']
    odds_norm = 1/(1+np.log10(odds)) if odds>1 else 1

    # 斤量（軽いほど有利）
    jin = row['jinryo']
    jin_max, jin_min = df['jinryo'].max(), df['jinryo'].min()
    jin_norm = (jin_max - jin)/(jin_max-jin_min) if jin_max>jin_min else 0

    # 馬体重増減
    wdiff = row['weight_diff']
    w_mean = df['weight'].mean()
    wdiff_norm = 1 - abs(wdiff)/w_mean if w_mean>0 else 0

    # 重み配分: raw 8, up3 2, odds/jin/wdiff 各1
    base = raw_norm*8 + up3_norm*2
    extra = odds_norm + jin_norm + wdiff_norm
    score = (base + extra) / (8+2+3)
    return score * 100

# --- スコア適用 ---
df['Score'] = df.apply(calc_score, axis=1)

# --- 偏差値計算 ---
stats = df.groupby('馬名')['Score'].agg(['mean','std']).reset_index()
stats.columns = ['馬名','平均スコア','標準偏差']
mu, sigma = stats['平均スコア'].mean(), stats['平均スコア'].std()
stats['偏差値'] = 50 + 10*(stats['平均スコア']-mu)/sigma

# --- 棒グラフ: 偏差値上位6頭 ---
top6 = stats.nlargest(6,'偏差値')
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='偏差値', y='馬名', data=top6, ax=ax)
ax.set_title('偏差値 上位6頭')
ax.set_xlabel('偏差値')
ax.set_ylabel('馬名')
st.pyplot(fig)

# --- 散布図: 調子×安定性 ---
fig2, ax2 = plt.subplots(figsize=(10,6))
# 分割閾値
x0 = mu             # 調子閾値
y0 = stats['標準偏差'].mean()  # 安定性閾値
xmin, xmax = stats['偏差値'].min() - 1, stats['偏差値'].max() + 1
ymin, ymax = stats['標準偏差'].min() - 0.5, stats['標準偏差'].max() + 0.5

# 背景ゾーン拡大：軽視ゾーンを上下いっぱいに
ax2.fill_betweenx([ymin, ymax], xmin, x0, color='blue', alpha=0.2)    # 軽視ゾーン
ax2.fill_betweenx([y0, ymax], x0, xmax, color='red', alpha=0.2)     # 抑え・穴狙い
ax2.fill_betweenx([ymin, y0], xmin, x0, color='green', alpha=0.2)   # 堅軸ゾーン
ax2.fill_betweenx([ymin, y0], x0, xmax, color='yellow', alpha=0.2)  # 本命候補

# 基準線
ax2.axvline(x0, color='gray', linestyle='--')
ax2.axhline(y0, color='gray', linestyle='--')

# 散布点
ax2.scatter(stats['偏差値'], stats['標準偏差'], color='black', s=30)

# 注釈
for _, r in stats.iterrows():
    ax2.text(r['偏差値'], r['標準偏差'] + 0.05, r['馬名'], fontproperties=jp_font, fontsize=9, ha='center')

# 四象限ラベル位置を少しずらして重なりを回避
x_low = xmin + (x0 - xmin) * 0.3
x_high = x0 + (xmax - x0) * 0.3
y_low = ymin + (y0 - ymin) * 0.3
y_high = y0 + (ymax - y0) * 0.3

ax2.text(x_low,  y_high, '軽視ゾーン', ha='center', va='center', fontproperties=jp_font)
ax2.text(x_high, y_high, '抑え・穴狙い', ha='center', va='center', fontproperties=jp_font)
ax2.text(x_low,  y_low,  '堅軸ゾーン', ha='center', va='center', fontproperties=jp_font)
ax2.text(x_high, y_low,  '本命候補', ha='center', va='center', fontproperties=jp_font)

# 軸調整
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('偏差値', fontproperties=jp_font)
ax2.set_ylabel('標準偏差', fontproperties=jp_font)
ax2.set_title('調子×安定性', fontproperties=jp_font)
st.pyplot(fig2)
