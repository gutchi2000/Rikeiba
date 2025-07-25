import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 日本語フォント読み込み
jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
plt.rcParams["font.family"] = jp_font.get_name()

st.title("競馬スコア分析アプリ")

uploaded_file = st.file_uploader("Excelファイル（出走馬データ）をアップロードしてください", type=["xlsx"])

if uploaded_file:
    # シート1: 出走データ
    df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    df.columns = ["馬名", "頭数", "グレード", "着順"]

    try:
        # シート2: 成績データ（MultiIndexを平坦化）
        df_stats = pd.read_excel(uploaded_file, sheet_name=1, header=[0, 1])
        df_stats.columns = ["馬名"] + [f"{col1}_{col2}" for col1, col2 in df_stats.columns[1:]]
    except:
        df_stats = None

    st.subheader("アップロードされた出走データ")
    st.write(df)

    # スコア計算
    GRADE_SCORE = {
        "GⅠ": 10, "GⅡ": 8, "GⅢ": 6, "リステッド": 5,
        "オープン特別": 4, "3勝クラス": 3, "2勝クラス": 2,
        "1勝クラス": 1, "新馬": 1, "未勝利": 1
    }
    GP_MIN = 1
    GP_MAX = 10

    def calculate_score(row):
        try:
            GP = GRADE_SCORE.get(row["グレード"], 1)
            N = int(row["頭数"])
            p = int(row["着順"])
            raw_score = GP * (N + 1 - p)
            score = (raw_score - GP_MIN) / (GP_MAX * N - GP_MIN) * 100
            return score
        except:
            return np.nan

    df["Score"] = df.apply(calculate_score, axis=1)

    # 平均スコアと偏差値
    avg_scores = df.groupby("馬名")["Score"].mean().reset_index()
    avg_scores.columns = ["馬名", "平均スコア"]
    mean = avg_scores["平均スコア"].mean()
    std = avg_scores["平均スコア"].std()
    avg_scores["偏差値"] = avg_scores["平均スコア"].apply(lambda x: 50 + 10 * (x - mean) / std)

    # 追加: 安定性と加重平均スコア
    std_scores = df.groupby("馬名")["Score"].std().reset_index()
    std_scores.columns = ["馬名", "スコア標準偏差"]

    df_sorted = df.sort_values(["馬名", "着順"], ascending=[True, True])
    df_recent = df_sorted.groupby("馬名").head(3).copy()

    def weighted_avg(x):
        weights = np.array([3, 2, 1][:len(x)])
        return np.average(x[::-1], weights=weights[::-1])

    weighted_scores = df_recent.groupby("馬名")["Score"].apply(weighted_avg).reset_index()
    weighted_scores.columns = ["馬名", "加重平均スコア"]

    mean_w = weighted_scores["加重平均スコア"].mean()
    std_w = weighted_scores["加重平均スコア"].std()
    weighted_scores["加重平均偏差値"] = weighted_scores["加重平均スコア"].apply(
        lambda x: 50 + 10 * (x - mean_w) / std_w if pd.notnull(x) else np.nan
    )

    avg_scores = avg_scores.merge(std_scores, on="馬名", how="left")
    avg_scores = avg_scores.merge(weighted_scores, on="馬名", how="left")

    if df_stats is not None:
        rate_types = ["勝率", "連対率", "複勝率"]
        thresholds_map = {
            "勝率": (30, 20, 10),
            "連対率": (50, 40, 30),
            "複勝率": (70, 60, 50)
        }

        def rate_mark(value, thresholds):
            if value >= thresholds[0]:
                return "◎", 4
            elif value >= thresholds[1]:
                return "〇", 3
            elif value >= thresholds[2]:
                return "△", 2
            else:
                return "×", 1

        for rt in rate_types:
            colname = f"{rt}_芝"
            if colname in df_stats.columns:
                marks, points = zip(*df_stats[colname].apply(lambda x: rate_mark(x, thresholds_map[rt])))
                df_stats[f"{rt}_印"] = marks
                df_stats[f"{rt}_点"] = points
            else:
                st.warning(f"{colname} 列が見つかりませんでした。")

        df_stats["評価点"] = df_stats[[f"{rt}_点" for rt in rate_types]].sum(axis=1)

        merged = pd.merge(avg_scores, df_stats[["馬名", "評価点"]], on="馬名", how="left")
        merged["評価点"] = merged["評価点"].fillna(0)
        merged["最終スコア"] = merged["偏差値"] * merged["評価点"]
        final_mean = merged["最終スコア"].mean()
        final_std = merged["最終スコア"].std()
        merged["最終偏差値"] = merged["最終スコア"].apply(lambda x: 50 + 10 * (x - final_mean) / final_std)

        top6 = merged.sort_values("最終偏差値", ascending=False).head(6)

        st.subheader("最終偏差値 上位6頭（棒グラフ）")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x="最終偏差値", y="馬名", data=top6, palette="Blues_d", ax=ax1)
        ax1.set_title("最終偏差値（上位6頭）", fontproperties=jp_font)
        ax1.set_xlabel("最終偏差値", fontproperties=jp_font)
        ax1.set_ylabel("馬名", fontproperties=jp_font)
        ax1.set_yticklabels(ax1.get_yticklabels(), fontproperties=jp_font)
        st.pyplot(fig1)

        st.subheader("偏差値 × 評価点 散布図（全馬）")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=merged, x="偏差値", y="評価点", hue="馬名", s=100, ax=ax2)
        ax2.set_title("偏差値 × 評価点 散布図", fontproperties=jp_font)
        ax2.set_xlabel("偏差値", fontproperties=jp_font)
        ax2.set_ylabel("評価点", fontproperties=jp_font)
        st.pyplot(fig2)

        st.subheader("上位6頭（最終偏差値順）")
        st.write(top6[["馬名", "最終偏差値"]])

        st.subheader("全馬データ（詳細）")
        st.write(merged)
    else:
        st.warning("勝率などの統計データ（シート2）が見つかりませんでした。評価点付きの分析はスキップされました。")
