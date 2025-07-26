import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ダミーデータ（本番ではdf_mergedとtop6をアプリから渡す）
data = {
    "馬名": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "最終偏差値": [60.2, 58.3, 57.1, 56.8, 55.0, 54.5, 52.3, 51.0, 50.1, 49.8],
    "偏差値": [55, 56, 52, 50, 49, 48, 50, 51, 47, 46],
    "評価点": [4, 3, 3, 2, 2, 2, 3, 1, 2, 1],
}
df = pd.DataFrame(data)

# 上位6頭を抽出
top6 = df.sort_values("最終偏差値", ascending=False).head(6)

# 棒グラフ：最終偏差値の上位6頭
plt.figure(figsize=(10, 6))
sns.barplot(x="最終偏差値", y="馬名", data=top6)
plt.title("最終偏差値（上位6頭）")
plt.xlabel("最終偏差値")
plt.ylabel("馬名")
plt.grid(True)
plt.tight_layout()
plt.show()

# 散布図：偏差値 × 評価点（全体）
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="偏差値", y="評価点", hue="馬名", s=100)
plt.title("偏差値 × 評価点 散布図（全頭）")
plt.xlabel("偏差値")
plt.ylabel("評価点")
plt.grid(True)
plt.tight_layout()
plt.show()
