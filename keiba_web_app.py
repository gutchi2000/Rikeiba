import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# seaborn不要なので削除

# --- 日本語フォント設定 ---
font_path = "ipaexg.ttf"  # アプリフォルダに配置
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

st.title("競馬スコア分析アプリ（騎手有無対応版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み ---
df = pd.read_excel(uploaded_file)
cols = df.columns.tolist()

# 騎手あり/なし判定
has_jockey = '騎手' in cols

# 必要列設定
base_cols = ["馬名","頭数","クラス名","確定着順",
             "上がり3Fタイム","Ave-3F",
             "馬場状態","馬体重","増減","斤量","単勝オッズ"]
if has_jockey:
    base_cols.insert(1, '騎手')

# 列抽出
missing = set(base_cols) - set(cols)
if missing:
    st.error(f"必要な列が不足しています: {missing}")
    st.stop()

df = df[base_cols]

# 型変換
for c in ["頭数","確定着順","上がり3Fタイム","Ave-3F","馬体重","増減","斤量","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# グレードマップ
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# スコア計算
def calc_score(r):
    N, p = r['頭数'], r['確定着順']
    GP = GRADE_SCORE.get(r['クラス名'],1)
    raw = GP * (N + 1 - p)
    denom = GP_MAX*N - GP_MIN
    raw_norm = (raw - GP_MIN)/denom if denom>0 else 0
    up3_norm = r['Ave-3F']/r['上がり3Fタイム'] if r['上がり3Fタイム']>0 else 0
    odds_norm = 1/(1+np.log10(r['単勝オッズ'])) if r['単勝オッズ']>1 else 1
    jin_min, jin_max = df['斤量'].min(), df['斤量'].max()
    jin_norm = (jin_max - r['斤量'])/(jin_max-jin_min) if jin_max>jin_min else 0
    wdiff_norm = 1 - abs(r['増減'])/df['馬体重'].mean() if df['馬体重'].mean()>0 else 0
    bonus = 0.1 if has_jockey else 0
    s = raw_norm*8 + up3_norm*2 + odds_norm + jin_norm + wdiff_norm + bonus
    return s/13*100

# スコア適用
df['Score'] = df.apply(calc_score, axis=1)

# 馬別偏差値算出
df_out = df.groupby('馬名')['Score'].agg(['mean','std']).reset_index()
df_out.columns=['馬名','平均スコア','標準偏差']
mu, sigma = df_out['平均スコア'].mean(), df_out['平均スコア'].std()
df_out['偏差値'] = 50 + 10*(df_out['平均スコア']-mu)/sigma

# 結果表示
st.subheader('偏差値 上位6頭')
st.write(df_out.nlargest(6,'偏差値')[['馬名','偏差値']])

st.subheader('全馬スコア')
st.dataframe(df_out)
