
import streamlit as st
import pandas as pd
import numpy as np

st.title("競馬スコア分析アプリ（複数評価方式対応）")

uploaded_file = st.file_uploader("Excelファイル（出走馬データ＋統計）をアップロードしてください", type=["xlsx"])

if uploaded_file:
    # 成績データ読み込み
    df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    df.columns = ["馬名", "頭数", "グレード", "着順"]

    # 統計データ読み込み
    df_stats = pd.read_excel(uploaded_file, sheet_name=1)

    st.subheader("アップロードされた出走成績データ")
    st.write(df)

    st.subheader("アップロードされた統計データ")
    st.write(df_stats)

    # レーススコアの計算
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

    # 馬ごとの平均スコア
    avg_scores = df.groupby("馬名")["Score"].mean().reset_index()
    avg_scores.columns = ["馬名", "平均スコア"]

    # 偏差値計算
    mean = avg_scores["平均スコア"].mean()
    std = avg_scores["平均スコア"].std()
    avg_scores["偏差値"] = avg_scores["平均スコア"].apply(lambda x: 50 + 10 * (x - mean) / std)

    # 評価関数
    def rate_mark(val, thresholds):
        if val >= thresholds[0]:
            return "◎", 4
        elif val >= thresholds[1]:
            return "〇", 3
        elif val >= thresholds[2]:
            return "△", 2
        else:
            return "×", 1

    # 対象：中央・全芝・芝
    targets = ["中央", "全芝", "芝"]
    rate_types = ["勝率", "連対率", "複勝率"]
    thresholds_map = {
        "勝率": (30, 20, 10),
        "連対率": (50, 40, 30),
        "複勝率": (70, 60, 50)
    }

    for tgt in targets:
        mark_cols = []
        score_col = []
        for rt in rate_types:
            colname = f"{rt}{tgt}"
            marks, points = zip(*df_stats[colname].apply(lambda x: rate_mark(x, thresholds_map[rt])))
            df_stats[f"{rt}印_{tgt}"] = marks
            df_stats[f"{rt}点_{tgt}"] = points
            mark_cols.append(f"{rt}印_{tgt}")
            score_col.append(f"{rt}点_{tgt}")
        df_stats[f"評価点_{tgt}"] = df_stats[score_col].sum(axis=1)

    # 最終評価は「芝」ベース
    merged = pd.merge(avg_scores, df_stats[["馬名", "評価点_芝"]], on="馬名", how="left")
    merged["最終スコア"] = merged["偏差値"] * merged["評価点_芝"]
    final_mean = merged["最終スコア"].mean()
    final_std = merged["最終スコア"].std()
    merged["最終偏差値"] = merged["最終スコア"].apply(lambda x: 50 + 10 * (x - final_mean) / final_std)

    top6 = merged.sort_values("最終偏差値", ascending=False).head(6)

    st.subheader("上位6頭（最終偏差値順）")
    st.write(top6[["馬名", "最終偏差値"]])

    st.subheader("全馬データ（最終スコア・偏差値）")
    st.write(merged)

    st.subheader("印付き統計評価（中央・全芝・芝）")
    st.write(df_stats)
