import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from adjustText import adjust_text

# ─── 日本語フォント読み込み ───
jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
plt.rcParams["font.family"] = jp_font.get_name()
sns.set(font=jp_font.get_name())

st.title("競馬スコア分析アプリ（完成版）")

# ─── ファイルアップロード ───
uploaded = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded:
    st.stop()

# ─── シート1 読み込み（ヘッダー有無対応） ───
cols = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F","馬場状態"]
df = pd.read_excel(uploaded, sheet_name=0)
if not all(c in df.columns for c in cols):
    df = pd.read_excel(uploaded, sheet_name=0, header=None)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
else:
    df = df[cols]

# 馬場状態を内部列名にマッピング
df = df.rename(columns={"馬場状態":"track_condition"})

# ─── スコア計算パラメータ ───
GRADE_SCORE = {
    "GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5,
    "オープン特別":4, "3勝クラス":3, "2勝クラス":2,
    "1勝クラス":1, "新馬":1, "未勝利":1
}
GP_MIN, GP_MAX = 1, 10

# ─── 拡張スコア計算（9:1 重み） ───
def calc_score(r):
    N, p = r["頭数"], r["着順"]
    GP = GRADE_SCORE.get(r["グレード"], 1)
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN)
    up3_norm = r["Ave-3F"] / r["上がり3F"] if r["上がり3F"]>0 else 0
    return (raw_norm*9 + up3_norm*1)/10*100

df["Score"] = df.apply(calc_score, axis=1)

# ─── 馬別 平均スコア＆偏差値 ───
avg = df.groupby("馬名")["Score"].mean().reset_index()
avg.columns = ["馬名","平均スコア"]
m, s = avg["平均スコア"].mean(), avg["平均スコア"].std()
avg["偏差値"] = avg["平均スコア"].apply(lambda x: 50 + 10*(x-m)/s)

# ─── 棒グラフ：偏差値上位6頭 ───
st.subheader("偏差値 上位6頭")
top6 = avg.nlargest(6, "偏差値")
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="偏差値", y="馬名", data=top6, ax=ax)
ax.set_title("偏差値 上位6頭", fontproperties=jp_font)
ax.set_xlabel("偏差値", fontproperties=jp_font)
ax.set_ylabel("馬名", fontproperties=jp_font)
ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontproperties=jp_font)
st.pyplot(fig)

# ─── 散布図：調子×安定性（重なり自動回避） ───
st.subheader("調子×安定性")
# 安定性（Score の標準偏差）を算出
stds = df.groupby("馬名")["Score"].std().reset_index()
stds.columns = ["馬名","標準偏差"]
avg2 = avg.merge(stds, on="馬名")

fig2, ax2 = plt.subplots(figsize=(10,6))
# プロット
ax2.scatter(avg2["偏差値"], avg2["標準偏差"], color="black", s=20)
# ラベルをまとめて生成
texts = []
for _, r in avg2.iterrows():
    texts.append(
        ax2.text(r["偏差値"], r["標準偏差"], r["馬名"],
                 fontproperties=jp_font, fontsize=8)
    )
# 重なり回避
adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray"), 
            expand_points=(1.2, 1.2), force_text=0.5)

# 四象限線（平均値軸）
x0, y0 = avg2["偏差値"].mean(), avg2["標準偏差"].mean()
ax2.axvline(x0, color="gray", linestyle="--")
ax2.axhline(y0, color="gray", linestyle="--")
# 四象限注釈
dx = (ax2.get_xlim()[1]-ax2.get_xlim()[0]) * 0.02
dy = (ax2.get_ylim()[1]-ax2.get_ylim()[0]) * 0.02
ax2.text(x0+dx, y0-dy, "本命候補", fontproperties=jp_font)
ax2.text(x0+dx, y0+dy, "抑え・穴狙い", fontproperties=jp_font)
ax2.text(x0-dx*5, y0+dy, "軽視ゾーン", fontproperties=jp_font)
ax2.text(x0-dx*5, y0-dy, "堅軸ゾーン", fontproperties=jp_font)

ax2.set_xlabel("調子（偏差値）", fontproperties=jp_font)
ax2.set_ylabel("安定性（標準偏差）", fontproperties=jp_font)
ax2.set_title("調子×安定性", fontproperties=jp_font)
st.pyplot(fig2)

# ─── テーブル表示 ───
st.subheader("馬別スコア一覧")
st.dataframe(avg2)
