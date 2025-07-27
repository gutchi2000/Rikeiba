import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("競馬スコア分析アプリ（完成版 8:2:1:1:1 重み）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み（ヘッダー有無対応） ---
cols = ["馬名","頭数","クラス名","確定着順","上がり3Fタイム",
        "Ave-3F","馬場状態","馬体重","増減","斤量","単勝オッズ"]
try:
    df = pd.read_excel(uploaded_file, sheet_name=0, usecols=cols)
    df.columns = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F",
                  "track_condition","weight","weight_diff","jinryo","odds"]
except ValueError:
    df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    df = df.iloc[:, :len(cols)]
    df.columns = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F",
                  "track_condition","weight","weight_diff","jinryo","odds"]

# --- 数値列の型変換 ---
for c in ["頭数","着順","上がり3F","Ave-3F","weight","weight_diff","jinryo","odds"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- パラメータ ---
GRADE_SCORE = {"GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5,
               "オープン特別":4, "3勝クラス":3, "2勝クラス":2,
               "1勝クラス":1, "新馬":1, "未勝利":1}
GP_MIN, GP_MAX = 1, 10

# --- スコア計算 (基本8 + 上がり2 + オッズ1 + 斤量1 + 体重1 =13) ---
def calc_score(row):
    N, p = row["頭数"], row["着順"]
    GP = GRADE_SCORE.get(row["グレード"], 1)
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN)
    up3_norm = row["Ave-3F"] / row["上がり3F"] if row["上がり3F"] > 0 else 0
    odds_norm = 1 / (1 + np.log10(row["odds"])) if row["odds"] > 1 else 1
    mean_jin = df["jinryo"].mean()
    jin_diff_norm = 1 - abs(row["jinryo"] - mean_jin) / mean_jin
    wdiff_norm = 1 - abs(row["weight_diff"]) / df["weight"].mean()
    score = raw_norm*8 + up3_norm*2 + odds_norm*1 + jin_diff_norm*1 + wdiff_norm*1
    return score / 13 * 100

df["Score"] = df.apply(calc_score, axis=1)

# --- 馬別 平均スコア & 偏差値 ---
df_avg = df.groupby("馬名")["Score"].mean().reset_index()
df_avg.columns = ["馬名","平均スコア"]
avg_mean = df_avg["平均スコア"].mean()
avg_std = df_avg["平均スコア"].std()
df_avg["偏差値"] = df_avg["平均スコア"].apply(lambda x: 50 + 10*(x-avg_mean)/avg_std)

# --- 棒グラフ: 偏差値上位6頭 ---
st.subheader("偏差値 上位6頭")
top6 = df_avg.nlargest(6, "偏差値")
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.barplot(x="偏差値", y="馬名", data=top6, ax=ax1)
ax1.set_title("偏差値 上位6頭")
ax1.set_xlabel("偏差値")
ax1.set_ylabel("馬名")
st.pyplot(fig1)

# --- 散布図: 調子×安定性 ---
st.subheader("調子×安定性")
df_std = df.groupby("馬名")["Score"].std().reset_index()
df_std.columns = ["馬名","標準偏差"]
avg2 = df_avg.merge(df_std, on="馬名")
fig2, ax2 = plt.subplots(figsize=(10,6))
x0 = avg2["偏差値"].mean()
y0 = avg2["標準偏差"].mean()
xmin, xmax = avg2["偏差値"].min(), avg2["偏差値"].max()
ymin, ymax = avg2["標準偏差"].min(), avg2["標準偏差"].max()
ax2.fill_betweenx([ymin,y0], xmin, x0, color="#dff0d8", alpha=0.3)
ax2.fill_betweenx([ymin,y0], x0, xmax, color="#fcf8e3", alpha=0.3)
ax2.fill_betweenx([y0,ymax], xmin, x0, color="#d9edf7", alpha=0.3)
ax2.fill_betweenx([y0,ymax], x0, xmax, color="#f2dede", alpha=0.3)
ax2.axvline(x0, color="gray", linestyle="--")
ax2.axhline(y0, color="gray", linestyle="--")
ax2.scatter(avg2["偏差値"], avg2["標準偏差"], color="black", s=20)
for i, row in avg2.iterrows():
    dy = (i % 3) * 0.1
    ax2.text(row["偏差値"], row["標準偏差"]+dy, row["馬名"], fontsize=8,
             ha="center", va="bottom")
ax2.text((x0+xmax)/2, (y0+ymin)/2, "本命候補", ha="center", va="center")
ax2.text((x0+xmax)/2, (y0+ymax)/2, "抑え・穴狙い", ha="center", va="center")
ax2.text((xmin+x0)/2, (y0+ymax)/2, "軽視ゾーン", ha="center", va="center")
ax2.text((xmin+x0)/2, (y0+ymin)/2, "堅軸ゾーン", ha="center", va="center")
ax2.set_xlabel("調子（偏差値）")
ax2.set_ylabel("安定性（標準偏差）")
st.pyplot(fig2)

# --- テーブル表示 ---
st.subheader("馬別スコア一覧")
st.dataframe(df_avg)
