import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# ——— 日本語フォント設定 ———
jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
plt.rcParams["font.family"] = jp_font.get_name()
sns.set(font=jp_font.get_name())

st.title("競馬スコア分析アプリ（完成版）")

# ——— Excelアップロード ———
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# ——— シート1 読み込み（ヘッダー有り／無し対応） ———
col_req = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F"]
try:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    # ヘッダー行がない場合
    if not all(c in df.columns for c in col_req):
        df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        df.columns = col_req
except Exception as e:
    st.error(f"シート1読み込み失敗: {e}")
    st.stop()
df = df[col_req]  # 必須列だけ残す

# ——— スコア計算パラメータ ———
GRADE_SCORE = {
    "GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
    "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
    "1勝クラス":1,"新馬":1,"未勝利":1
}
GP_MIN, GP_MAX = 1, 10

# ——— 拡張スコア計算式 ———
def calculate_score_ext(row):
    N, p = row["頭数"], row["着順"]
    GP = GRADE_SCORE.get(row["グレード"], 1)
    U3i, U3std = row["上がり3F"], row["Ave-3F"]
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN)
    up3_norm = U3std / U3i if U3i>0 else 0
    return (raw_norm + up3_norm)/2 * 100

df["Score"] = df.apply(calculate_score_ext, axis=1)

# ——— 馬ごとの平均スコア＆偏差値 ———
avg = df.groupby("馬名")["Score"].mean().reset_index()
avg.columns = ["馬名","平均スコア"]
m, s = avg["平均スコア"].mean(), avg["平均スコア"].std()
avg["偏差値"] = avg["平均スコア"].apply(lambda x: 50 + 10*(x-m)/s)

# ——— 安定性（標準偏差） ———
stds = df.groupby("馬名")["Score"].std().reset_index()
stds.columns = ["馬名","スコア標準偏差"]
avg = avg.merge(stds, on="馬名")

# ——— 直近3走の加重偏差値 ———
df["順"] = df.groupby("馬名").cumcount(ascending=False) + 1
recent = df.sort_values(["馬名","順"]).groupby("馬名").head(3)
def wavg(x): return np.average(x[::-1], weights=[3,2,1][:len(x)][::-1])
w = recent.groupby("馬名")["Score"].apply(wavg).reset_index()
w.columns = ["馬名","加重平均スコア"]
mw, sw = w["加重平均スコア"].mean(), w["加重平均スコア"].std()
w["加重平均偏差値"] = w["加重平均スコア"].apply(lambda x: 50 + 10*(x-mw)/sw)
avg = avg.merge(w, on="馬名")

# ——— 出力 ———
st.subheader("馬別スコア一覧")
st.dataframe(avg)

# ——— 棒グラフ: 偏差値上位6頭 ———
top6 = avg.sort_values("偏差値", ascending=False).head(6)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="偏差値", y="馬名", data=top6, ax=ax)
ax.set_title("偏差値 上位6頭", fontproperties=jp_font)
ax.set_xlabel("偏差値", fontproperties=jp_font)
ax.set_ylabel("馬名", fontproperties=jp_font)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=jp_font)
st.pyplot(fig)

# ——— 散布図: 調子(加重偏差値)×安定性 ———
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.scatterplot(
    data=avg, x="偏差値", y="加重平均偏差値",
    hue="馬名", s=100, ax=ax2
)
ax2.set_title("調子(加重偏差値)×安定性", fontproperties=jp_font)
ax2.set_xlabel("偏差値", fontproperties=jp_font)
ax2.set_ylabel("加重平均偏差値", fontproperties=jp_font)
for lbl in ax2.get_xticklabels(): lbl.set_fontproperties(jp_font)
for lbl in ax2.get_yticklabels(): lbl.set_fontproperties(jp_font)
legend = ax2.get_legend()
if legend:
    legend.set_bbox_to_anchor((1.02,1))
    legend.set_loc("upper left")
    for text in legend.get_texts(): text.set_fontproperties(jp_font)
st.pyplot(fig2)
