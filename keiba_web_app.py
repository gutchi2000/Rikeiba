import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 日本語フォント設定
jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
plt.rcParams["font.family"] = jp_font.get_name()
sns.set(font=jp_font.get_name())

st.title("競馬スコア分析アプリ（完成版）")

# --- ファイルアップロード ---
uploaded = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded:
    st.stop()
# 読み込み (ヘッダー有無両対応)
df = pd.read_excel(uploaded, sheet_name=0)
# 必要列チェック & 先頭7列に絞る
cols = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F","馬場状態"]
if not all(c in df.columns for c in cols):
    df = pd.read_excel(uploaded, sheet_name=0, header=None)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
else:
    df = df[cols]

# 列名統一
df = df.rename(columns={"馬場状態":"track_condition"})

# グレード点数マップ
GRADE_SCORE = {
    "GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5,
    "オープン特別":4, "3勝クラス":3, "2勝クラス":2,
    "1勝クラス":1, "新馬":1, "未勝利":1
}
GP_MIN, GP_MAX = 1, 10

# スコア計算
def calc_score(r):
    N, p = r['頭数'], r['着順']
    GP = GRADE_SCORE.get(r['グレード'], 1)
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN)
    up3 = r['Ave-3F'] / r['上がり3F'] if r['上がり3F']>0 else 0
    # 9:1 重み
    return (raw_norm*9 + up3*1)/10*100

df['Score'] = df.apply(calc_score, axis=1)

# 馬別 平均 & 偏差値
avg = df.groupby('馬名')['Score'].mean().reset_index()
avg.columns = ['馬名','平均スコア']
m, s = avg['平均スコア'].mean(), avg['平均スコア'].std()
avg['偏差値'] = avg['平均スコア'].apply(lambda x:50+10*(x-m)/s)

# 棒グラフ
st.subheader('偏差値 上位6頭')
top6 = avg.nlargest(6,'偏差値')
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='偏差値',y='馬名',data=top6,ax=ax)
ax.set_title('偏差値 上位6頭', fontproperties=jp_font)
ax.set_xlabel('偏差値', fontproperties=jp_font)
ax.set_ylabel('馬名', fontproperties=jp_font)
ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontproperties=jp_font)
st.pyplot(fig)

# 散布図: 調子×安定性 with annotate and quadrants
st.subheader('調子×安定性')
stds = df.groupby('馬名')['Score'].std().reset_index()
stds.columns=['馬名','標準偏差']
avg2 = avg.merge(stds,on='馬名')
fig2, ax2 = plt.subplots(figsize=(10,6))
# プロット
ax2.scatter(avg2['偏差値'],avg2['標準偏差'],color='black',s=20)
# 四象限線
x0 = avg2['偏差値'].mean()
y0 = avg2['標準偏差'].mean()
ax2.axvline(x0, color='gray', linestyle='--')
ax2.axhline(y0, color='gray', linestyle='--')
# ラベル
for _,r in avg2.iterrows():
    ax2.text(r['偏差値'],r['標準偏差'],r['馬名'],fontproperties=jp_font,fontsize=8,ha='center',va='bottom')
# 四象限注釈
offset_x = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.02
offset_y = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.02
ax2.text(x0+offset_x, y0-offset_y, '本命候補', fontproperties=jp_font)
ax2.text(x0+offset_x, y0+offset_y, '抑え・穴狙い', fontproperties=jp_font)
ax2.text(x0-offset_x*5, y0+offset_y, '軽視ゾーン', fontproperties=jp_font)
ax2.text(x0-offset_x*5, y0-offset_y, '堅軸ゾーン', fontproperties=jp_font)
# 軸設定
ax2.set_xlabel('調子（偏差値）',fontproperties=jp_font)
ax2.set_ylabel('安定性（標準偏差）',fontproperties=jp_font)
ax2.set_title('調子×安定性',fontproperties=jp_font)
st.pyplot(fig2)
st.subheader('調子×安定性')
stds = df.groupby('馬名')['Score'].std().reset_index()
stds.columns=['馬名','標準偏差']
avg2 = avg.merge(stds,on='馬名')
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.scatter(avg2['偏差値'],avg2['標準偏差'],color='black',s=20)
for _,r in avg2.iterrows():
    ax2.text(r['偏差値'],r['標準偏差'],r['馬名'],fontproperties=jp_font,fontsize=8,
             ha='center',va='bottom')
ax2.set_xlabel('調子（偏差値）',fontproperties=jp_font)
ax2.set_ylabel('安定性（標準偏差）',fontproperties=jp_font)
ax2.set_title('調子×安定性',fontproperties=jp_font)
st.pyplot(fig2)

# テーブル
st.subheader('馬別スコア一覧')
st.dataframe(avg)
