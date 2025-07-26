# 馬ごとのスコア標準偏差（安定性）と近走加重スコアを加味するコード
import pandas as pd
import numpy as np

# --- 前提: df には既に "馬名", "Score" カラムが存在している ---

# 1. スコア標準偏差の算出（安定性）
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
