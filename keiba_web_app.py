import pandas as pd
import numpy as np

# --- Weighted Score Analysis: 完成版コード ---
# df: DataFrame with columns ['馬名', 'Score'] already loaded
# 実運用では以下のようにExcelから読み込みます:
# df = pd.read_excel("data.xlsx")

# Example: if running standalone, uncomment and set file path
# file_path = "data.xlsx"
# df = pd.read_excel(file_path, usecols=["馬名", "Score"])

# 1. スコア標準偏差の算出（安定性）
score_std = df.groupby("馬名")["Score"].std().reset_index()
score_std.columns = ["馬名", "スコア標準偏差"]

# 2. 馬ごとの平均スコア算出と統合
avg_scores = df.groupby("馬名")["Score"].mean().reset_index()
avg_scores.columns = ["馬名", "平均スコア"]
avg_scores = pd.merge(avg_scores, score_std, on="馬名", how="left")

# 3. 偏差値計算（平均スコア）
mean_score = avg_scores["平均スコア"].mean()
std_score = avg_scores["平均スコア"].std()
avg_scores["偏差値"] = avg_scores["平均スコア"].apply(
    lambda x: 50 + 10 * (x - mean_score) / std_score
)

# 4. 近走加重スコア算出（仮: dfに重み列がある場合）
# ここでは既にdfに「レース順」列と「重み」列が存在すると仮定します
# 実運用時には以下の計算をばらして実装してください:
# df['レース順'] = df.groupby("馬名").cumcount(ascending=False) + 1
# weight_map = {1:3, 2:2, 3:1}
# df['重み'] = df['レース順'].map(weight_map).fillna(0)
# df['加重スコア'] = df['Score'] * df['重み']

# sum_weighted = df.groupby("馬名")["加重スコア"].sum()
# sum_weights = df.groupby("馬名")["重み"].sum()
# weighted_avg = (sum_weighted / sum_weights).reset_index(name="加重平均スコア")

# 5. 偏差値化
# mean_w = weighted_avg["加重平均スコア"].mean()
# std_w = weighted_avg["加重平均スコア"].std()
# weighted_avg["加重偏差値"] = weighted_avg["加重平均スコア"].apply(
#     lambda x: 50 + 10 * (x - mean_w) / std_w
# )

# 6. 複合指標作成
# final = pd.merge(avg_scores, weighted_avg, on="馬名", how="left")
# final["安定×加重"] = final["加重平均スコア"] / final["スコア標準偏差"]

# 結果出力
# print(final.sort_values("加重偏差値", ascending=False))

# シンプルに平均+偏差値のみ表示
print(avg_scores.sort_values("偏差値", ascending=False))
