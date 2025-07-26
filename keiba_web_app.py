import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 日本語フォント指定（必要なら有効にしてください） ---
# from matplotlib.font_manager import FontProperties
# jp_font = FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")

st.title("競馬スコア分析Webアプリ")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("CSVファイルを選択してください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("データサンプル", df.head())

    # --- スコア計算（例：適宜調整してください） ---
    # GP列がある場合の例（なければ適宜変更）
    GP_min, GP_max = 1, 10
    df["RawScore"] = df["GP"] * (df["field_size"] + 1 - df["place"])
    df["Score"] = (df["RawScore"] - GP_min) / (GP_max * df["field_size"] - GP_min) * 100

    # 馬ごとの平均スコア＆偏差値
    avg = df.groupby("horse_id")["Score"].mean().reset_index()
    avg.columns = ["horse_id", "平均スコア"]
    m, s = avg["平均スコア"].mean(), avg["平均スコア"].std()
    avg["偏差値"] = avg["平均スコア"].apply(lambda x: 50 + 10*(x-m)/s)

    # --- 棒グラフ: 偏差値上位6頭 ---
    st.subheader("偏差値 上位6頭（棒グラフ）")
    top6 = avg.sort_values("偏差値", ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="偏差値", y="horse_id", data=top6, ax=ax)
    # ax.set_title("偏差値 上位6頭", fontproperties=jp_font)
    # ax.set_xlabel("偏差値", fontproperties=jp_font)
    # ax.set_ylabel("馬ID", fontproperties=jp_font)
    # ax.set_yticklabels(ax.get_yticklabels(), fontproperties=jp_font)
    ax.set_title("偏差値 上位6頭")
    ax.set_xlabel("偏差値")
    ax.set_ylabel("馬ID")
    st.pyplot(fig)

    # --- 散布図: 調子×安定性（馬名ラベル版） ---
    st.subheader("調子×安定性（馬IDラベル版）")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    stds = df.groupby("horse_id")["Score"].std().reset_index()
    stds.columns = ["horse_id", "標準偏差"]
    avg2 = avg.merge(stds, on="horse_id")
    ax2.scatter(avg2["偏差値"], avg2["標準偏差"])
    for _, row in avg2.iterrows():
        ax2.annotate(
            row['horse_id'],
            (row['偏差値'], row['標準偏差']),
            # fontproperties=jp_font,
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points"
        )
    # ax2.set_xlabel("調子（偏差値）", fontproperties=jp_font)
    # ax2.set_ylabel("安定性（標準偏差）", fontproperties=jp_font)
    # ax2.set_title("調子×安定性", fontproperties=jp_font)
    ax2.set_xlabel("調子（偏差値）")
    ax2.set_ylabel("安定性（標準偏差）")
    ax2.set_title("調子×安定性")
    st.pyplot(fig2)

    # --- テーブル表示 ---
    st.subheader("馬別スコア一覧（テーブル）")
    st.dataframe(avg2)

else:
    st.info("まずCSVファイルをアップロードしてください。")
