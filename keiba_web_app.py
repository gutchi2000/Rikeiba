import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 日本語フォント読み込み
jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
plt.rcParams["font.family"] = jp_font.get_name()
sns.set(font=jp_font.get_name())

st.title("競馬スコア分析アプリ（拡張版）")

uploaded_file = st.file_uploader("Excelファイル（出走馬データ＋成績統計）をアップロード", type=["xlsx"])
if not uploaded_file:
    st.stop()

# シート1: 出走馬情報（必要カラム確認と手動入力対応）
df = pd.read_excel(uploaded_file, sheet_name=0)
col_req = ["馬名","頭数","グレード","着順","走破タイム","基準タイム","上がり3F","Ave-3F","単勝オッズ"]
# 不足カラム確認
missing = [c for c in col_req if c not in df.columns]
if missing:
    st.warning(f"以下のカラムが見つかりませんでした: {missing}\n手動で入力してください。")
    if "基準タイム" in missing:
        Tstd_manual = st.number_input("基準タイム(秒) を入力", value=60.0, step=0.1)
        df["基準タイム"] = Tstd_manual
    if "Ave-3F" in missing:
        U3std_manual = st.number_input("Ave-3F(秒) を入力", value=35.0, step=0.1)
        df["Ave-3F"] = U3std_manual
    for c in [c for c in missing if c not in ["基準タイム","Ave-3F"]]:
        st.error(f"必須カラム {c} がありません。アップロードファイルを確認してください。")
        st.stop()
# カラム順整備
df = df[col_req]

# 基準タイム確認表示
st.subheader("基準タイム一覧（手動またはファイル）")
st.write(df[["馬名","基準タイム"]])

# シート2: 成績統計データ読み込み
try:
    df_stats = pd.read_excel(uploaded_file, sheet_name=1, header=[0,1])
    df_stats.columns = ["馬名"] + [f"{c1}_{c2}" for c1, c2 in df_stats.columns[1:]]
except:
    df_stats = None

# グレードスコア定義
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN = 1
GP_MAX = 10

# 拡張スコア計算式（Raw_norm + Up3_norm）
def calculate_score_ext(row):
    N = row['頭数']
    p = row['着順']
    GP = GRADE_SCORE.get(row['グレード'], 1)
    U3i = row['上がり3F']
    U3std = row['Ave-3F']
    # 生スコア
    raw_score = GP * (N + 1 - p)
    # 正規化
    raw_norm = (raw_score - GP_MIN) / (GP_MAX * N - GP_MIN)
    up3_norm = U3std / U3i if U3i > 0 else 0
    # 最終スコア
    score = (raw_norm + up3_norm) / 2 * 100
    return score

# スコア列追加
df['Score'] = df.apply(calculate_score_ext, axis=1)

# 平均スコア・偏差値計算
avg_scores = df.groupby("馬名")["Score"].mean().reset_index()
avg_scores.columns = ["馬名","平均スコア"]
mean = avg_scores["平均スコア"].mean()
std = avg_scores["平均スコア"].std()
avg_scores["偏差値"] = avg_scores["平均スコア"].apply(lambda x: 50 + 10*(x-mean)/std)

# 安定性（標準偏差）
std_scores = df.groupby("馬名")["Score"].std().reset_index()
std_scores.columns = ["馬名","スコア標準偏差"]

# 直近3走加重平均偏差値
# 日付情報がない場合は出走順
df['レース順'] = df.groupby("馬名").cumcount(ascending=False)+1
df_recent = df.sort_values(["馬名","レース順"]).groupby("馬名").head(3)

def weighted_avg(x):
    w = np.array([3,2,1][:len(x)])
    return np.average(x[::-1],weights=w[::-1])

weighted = df_recent.groupby("馬名")["Score"].apply(weighted_avg).reset_index()
weighted.columns = ["馬名","加重平均スコア"]
mean_w = weighted["加重平均スコア"].mean()
std_w = weighted["加重平均スコア"].std()
weighted["加重平均偏差値"] = weighted["加重平均スコア"].apply(lambda x:50+10*(x-mean_w)/std_w)

# avg_scores結合
avg_scores = avg_scores.merge(std_scores,on="馬名").merge(weighted,on="馬名")

# 従来成績統計があれば評価点付与＆最終偏差値
if df_stats is not None:
    # ... 省略: 既存の印付け～最終偏差値算出処理 をここに挿入 ...
    pass

st.subheader("馬ごとのスコア一覧（平均・偏差値・安定性・加重偏差値）")
st.dataframe(avg_scores)
st.success("拡張スコア分析完了！")
