# 馬ごとのスコア標準偏差（安定性）と近走加重スコアを加味するコード
import pandas as pd
import numpy as np

import streamlit as st

# Streamlitファイルアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()
# アップロードファイルから読み込み
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"ファイル読み込みエラー: {e}")
    st.stop()

# --- 前提: アップロードシートに過去成績があり、必要カラムを計算する ---
# 必要カラム: 馬名, 頭数, グレード, 着順
col_req = ['馬名','頭数','グレード','着順']
if not all(c in df.columns for c in col_req):
    st.error(f"必須カラムが不足しています: {col_req}")
    st.stop()

# スコア計算（例：単純に着順と頭数からRawScore）
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,"オープン特別":4,
               "3勝クラス":3,"2勝クラス":2,"1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

def calculate_score(row):
    N, p = row['頭数'], row['着順']
    GP = GRADE_SCORE.get(row['グレード'],1)
    raw = GP * (N + 1 - p)
    return raw

# Score 列を追加
if 'Score' not in df.columns:
    df['Score'] = df.apply(calculate_score, axis=1)

# 必須列確認: 馬名, Score
required_cols = ['馬名','Score']
if not all(col in df.columns for col in required_cols):
    st.error(f"入力データに必要なカラムがありません: {required_cols}")
    st.stop()
required_cols = ['馬名', 'Score']
if not all(col in df.columns for col in required_cols):
    st.error(f"入力データに必要なカラムがありません: {required_cols}")
    st.stop()

# 1. スコア標準偏差の算出（安定性）（安定性）
score_std = df.groupby("馬名")["Score"].std().reset_index()
score_std.columns = ["馬名", "スコア標準偏差"]

# 2. 馬ごとのスコア平均と統合（仮に avg_scores があるとする）
avg_scores = df.groupby("馬名")["Score"].mean().reset_index()
avg_scores.columns = ["馬名", "平均スコア"]
avg_scores = pd.merge(avg_scores, score_std, on="馬名", how="left")

# 3. 偏差値計算（平均スコア）
mean = avg_scores["平均スコア"].mean()
std = avg_scores["平均スコア"].std()
avg_scores["偏差値"] = avg_scores["平均スコア"].apply(lambda x: 50 + 10 * (x - mean) / std)

# 4. 近走スコア加重（仮に日付順なし → 出現順で処理）
df["レース順"] = df.groupby("馬名").cumcount(ascending=False) + 1

# 重み付け（直近3走に 3, 2, 1）
weight_map = {1: 3, 2: 2, 3: 1}
df["重み"] = df["レース順"].map(weight_map).fillna(0)
df["加重スコア"] = df["Score"] * df["重み"]

# 加重平均スコアの算出
sum_weighted = df.groupby("馬名")["加重スコア"].sum()
sum_weights = df.groupby("馬名")["重み"].sum()
weighted_avg = (sum_weighted / sum_weights).reset_index(name="加重平均スコア")

# 偏差値化
mean_w = weighted_avg["加重平均スコア"].mean()
std_w = weighted_avg["加重平均スコア"].std()
weighted_avg["加重偏差値"] = weighted_avg["加重平均スコア"].apply(lambda x: 50 + 10 * (x - mean_w) / std_w)

# 5. 複合指標（例：加重スコア ÷ 標準偏差）
avg_scores = pd.merge(avg_scores, weighted_avg, on="馬名", how="left")
avg_scores["安定×加重"] = avg_scores["加重平均スコア"] / avg_scores["スコア標準偏差"]

# 結果表示（任意）
print(avg_scores.sort_values("加重偏差値", ascending=False))
