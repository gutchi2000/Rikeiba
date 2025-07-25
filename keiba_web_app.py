import streamlit as st
import pandas as pd
import numpy as np

st.title("競馬スコア分析アプリ")

uploaded_file = st.file_uploader("Excelファイル（出走馬データ）をアップロードしてください", type=["xlsx"])

if uploaded_file:
    # 出走馬データ（シート1）読み込み
    df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    df.columns = ["馬名", "頭数", "グレード", "着順"]

    # 統計データ（シート2）マルチヘッダーとして読み込み
    try:
        df_stats = pd.read_excel(uploaded_file, sheet_name=1, header=[0,1])
    except:
        df_stats = None

    st.subheader("アップロードされた出走馬データ")
    st.write(df)

    GRADE_SCORE = {"GⅠ": 10, "GⅡ": 8, "GⅢ": 6, "リステッド": 5, "オープン特別": 4,
                   "3勝クラス": 3, "2勝クラス": 2, "1勝クラス": 1, "新馬": 1, "未勝利": 1}
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

    avg_scores = df.groupby("馬名")["Score"].mean().reset_index()
    avg_scores.columns = ["馬名", "平均スコア"]

    mean = avg_scores["平均スコア"].mean()
    std = avg_scores["平均スコア"].std()
    avg_scores["偏差値"] = avg_scores["平均スコア"].apply(lambda x: 50 + 10 * (x - mean) / std)

    if df_stats is not None:
        # インデックスをリセットし、1列目を「馬名」にする
        df_stats = df_stats.reset_index()
        df_stats.columns.values[0] = "馬名"  # 最初の列名を "馬名" に

        # 必要な芝列だけ取得
        rate_types = ["勝率", "連対率", "複勝率"]
        thresholds_map = {
            "勝率": (30, 20, 10),
            "連対率": (50, 40, 30),
            "複勝率": (70, 60, 50)
        }

        def rate_mark(rate, thresholds):
            if rate >= thresholds[0]:
                return "◎", 4
            elif rate >= thresholds[1]:
                return "〇", 3
            elif rate >= thresholds[2]:
                return "△", 2
            else:
                return "×", 1

        for rt in rate_types:
            colname = (rt, "芝")
            if colname in df_stats.columns:
                marks, points = zip(*df_stats[colname].apply(lambda x: rate_mark(x, thresholds_map[rt])))
                df_stats[f"{rt}_印"] = marks
                df_stats[f"{rt}_点"] = points
            else:
                st.warning(f"{colname} 列が見つかりませんでした。")

        df_stats["評価点"] = df_stats[[f"{rt}_点" for rt in rate_types]].sum(axis=1)

        # 結合して最終スコア
        merged = pd.merge(avg_scores, df_stats[["馬名", "評価点"]], on="馬名", how="left")
        merged["評価点"] = merged["評価点"].fillna(0)
        merged["最終スコア"] = merged["偏差値"] * merged["評価点"]

        final_mean = merged["最終スコア"].mean()
        final_std = merged["最終スコア"].std()
        merged["最終偏差値"] = merged["最終スコア"].apply(lambda x: 50 + 10 * (x - final_mean) / final_std)

        top6 = merged.sort_values("最終偏差値", ascending=False).head(6)

        st.subheader("上位6頭（最終偏差値順）")
        st.write(top6[["馬名", "最終偏差値"]])

        st.subheader("全馬データ（詳細）")
        st.write(merged)
    else:
        st.warning("統計データ（シート2）が見つかりませんでした。")
