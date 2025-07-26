import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib import font_manager

# 日本語フォント読み込み
jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
plt.rcParams["font.family"] = jp_font.get_name()
sns.set(font=jp_font.get_name())

st.title("競馬スコア分析アプリ（完成版）")

# Excelアップロード
df = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not df:
    st.stop()
df = pd.read_excel(df, sheet_name=0, header=None)

# カラム設定
col_req = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F","馬場状態"]
df = df.iloc[:, :len(col_req)]
df.columns = col_req

# 列名マッピング
df = df.rename(columns={'馬場状態':'track_condition'})

# スコア計算パラメータ
GRADE_SCORE = {"GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5,
               "オープン特別":4, "3勝クラス":3, "2勝クラス":2,
               "1勝クラス":1, "新馬":1, "未勝利":1}
GP_MIN, GP_MAX = 1, 10

# 拡張スコア計算式（9:1重み）
def calculate_score_ext(row):
    N, p = row["頭数"], row["着順"]
    GP = GRADE_SCORE.get(row["グレード"], 1)
    U3i, U3std = row["上がり3F"], row["Ave-3F"]
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN)
    up3_norm = U3std / U3i if U3i > 0 else 0
    weighted = raw_norm * 9 + up3_norm * 1
    return (weighted / 10) * 100

df["Score"] = df.apply(calculate_score_ext, axis=1)

# 馬ごとの平均スコア＆偏差値
avg = df.groupby("馬名")["Score"].mean().reset_index()
avg.columns = ["馬名","平均スコア"]
m, s = avg["平均スコア"].mean(), avg["平均スコア"].std()
avg["偏差値"] = avg["平均スコア"].apply(lambda x: 50 + 10*(x-m)/s)

# 棒グラフ: 偏差値上位6頭
st.subheader("偏差値 上位6頭（棒グラフ）")
top6 = avg.sort_values("偏差値", ascending=False).head(6)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="偏差値", y="馬名", data=top6, ax=ax)
ax.set_title("偏差値 上位6頭", fontproperties=jp_font)
ax.set_xlabel("偏差値", fontproperties=jp_font)
ax.set_ylabel("馬名", fontproperties=jp_font)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=jp_font)
st.pyplot(fig)

# 散布図: 調子×安定性（調整テキスト使用）
st.subheader("調子×安定性（馬名ラベル版、重なり調整）")
fig2, ax2 = plt.subplots(figsize=(10,6))
df_std = df.groupby("馬名")["Score"].std().reset_index()
df_std.columns = ["馬名","標準偏差"]
avg2 = avg.merge(df_std, on="馬名")
# プロット
ax2.scatter(avg2["偏差値"], avg2["標準偏差"], color='black', s=30)
# ラベル配置と重なり調整
texts = []
for _, row in avg2.iterrows():
    texts.append(
        ax2.text(row['偏差値'], row['標準偏差'], row['馬名'], fontproperties=jp_font)
    )
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

# 軸・タイトル設定
ax2.set_xlabel("調子（偏差値）", fontproperties=jp_font)
ax2.set_ylabel("安定性（標準偏差）", fontproperties=jp_font)
ax2.set_title("調子×安定性", fontproperties=jp_font)
st.pyplot(fig2)

# テーブル表示
st.subheader("馬別スコア一覧（テーブル）")
st.dataframe(avg)
