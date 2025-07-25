
import streamlit as st
import pandas as pd
import numpy as np

st.title("競馬スコア分析アプリ")

uploaded_file = st.file_uploader("Excelファイル（出走馬データ）をアップロードしてください", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        df.columns = ["馬名", "頭数", "グレード", "着順"]
        st.success("1枚目（出走履歴）読み込み成功")
    except Exception as e:
        st.error(f"1枚目の読み込みエラー: {e}")
        st.stop()

    try:
        df_stats = pd.read_excel(uploaded_file, sheet_name=1)
        st.success("2枚目（統計データ）読み込み成功")
    except:
        df_stats = None
        st.warning("2枚目の統計データ（勝率など）が見つかりません。")

    st.subheader("アップロードされたデータ")
    st.write(df)

    # 各レースのスコア計算（着順, 頭数, グレード）
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

    # 馬ごとに平均スコアを出す
    avg_scores = df.groupby("馬名")["Score"].mean().reset_index()
    avg_scores.columns = ["馬名", "平均スコア"]

    # 偏差値計算
    mean = avg_scores["平均スコア"].mean()
    std = avg_scores["平均スコア"].std()
    avg_scores["偏差値"] = avg_scores["平均スコア"].apply(lambda x: 50 + 10 * (x - mean) / std)

    if df_stats is not None:
        def evaluate(rate, thresholds):
            if rate >= thresholds[0]:
                return ("◎", 4)
            elif rate >= thresholds[1]:
                return ("〇", 3)
            elif rate >= thresholds[2]:
                return ("△", 2)
            else:
                return ("×", 1)

        def process_stats(row):
            marks = {}
            score = 0
            for col, thresholds in zip(["勝率", "連対率", "複勝率"],
                                       [(30, 20, 10), (50, 40, 30), (70, 60, 50)]):
                mark, pt = evaluate(row[col]*100, thresholds)
                marks[col] = mark
                score += pt
            return pd.Series([marks["勝率"], marks["連対率"], marks["複勝率"], score],
                             index=["勝率印", "連対率印", "複勝率印", "評価点"])

        df_stats[["勝率印", "連対率印", "複勝率印", "評価点"]] = df_stats.apply(process_stats, axis=1)

        # 偏差値 × 評価点 → 再スコア
        merged = pd.merge(avg_scores, df_stats[["馬名", "評価点"]], on="馬名", how="left")
        merged["評価点"] = merged["評価点"].fillna(0)
        merged["最終スコア"] = merged["偏差値"] * merged["評価点"]

        # 最終偏差値
        final_mean = merged["最終スコア"].mean()
        final_std = merged["最終スコア"].std()
        merged["最終偏差値"] = merged["最終スコア"].apply(lambda x: 50 + 10 * (x - final_mean) / final_std)

        top6 = merged.sort_values("最終偏差値", ascending=False).head(6)

        st.subheader("上位6頭（最終偏差値順）")
        st.write(top6[["馬名", "最終偏差値"]])

        st.subheader("全馬データ（詳細）")
        st.write(merged)
    else:
        st.warning("評価点付きの分析はスキップされました。")
