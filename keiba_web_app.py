import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

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
df = pd.read_excel(uploaded_file, sheet_name=0)
cols = df.columns.tolist()

# 騎手あり/なし判定
has_jockey = '騎手' in cols

# 必要列設定
base_cols = ["馬名","頭数","クラス名","確定着順",
             "上がり3Fタイム","Ave-3F",
             "馬場状態","馬体重","増減","斤量","単勝オッズ"]
if has_jockey:
    base_cols.insert(1, '騎手')  # 騎手を2列目に

# 列抽出
try:
    df = df[base_cols]
except KeyError:
    st.error(f"必要な列が不足しています: {set(base_cols) - set(cols)}")
    st.stop()

# 型変換
num_cols = [c for c in base_cols if c not in ['馬名','騎手','クラス名','馬場状態']]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# パラメータ
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# スコア計算関数
def calc_score(row):
    N, p = row['頭数'], row['確定着順']
    GP = GRADE_SCORE.get(row['クラス名'],1)
    raw = GP*(N+1-p)
    denom = GP_MAX*N - GP_MIN
    raw_norm = (raw-GP_MIN)/denom if denom>0 else 0
    up3_norm = (row['Ave-3F']/row['上がり3Fタイム']) if row['上がり3Fタイム']>0 else 0
    odds_norm = 1/(1+np.log10(row['単勝オッズ'])) if row['単勝オッズ']>1 else 1
    jin_min, jin_max = df['斤量'].min(), df['斤量'].max()
    jin_norm = (jin_max-row['斤量'])/(jin_max-jin_min) if jin_max>jin_min else 0
    wdiff_norm = 1 - abs(row['増減'])/df['馬体重'].mean() if df['馬体重'].mean()>0 else 0
    # 騎手成績を考慮
    if has_jockey:
        # 仮に騎手別勝率が df_stats にある場合を想定
        # ここではランダム補正値として0.1加算例
        jockey_bonus = 0.1
    else:
        jockey_bonus = 0
    # 重み: raw8, up3 2, odds1, jin1, wdiff1
    s = raw_norm*8 + up3_norm*2 + odds_norm + jin_norm + wdiff_norm + jockey_bonus
    return s/13*100

# スコア適用
col_score = 'Score'
df[col_score] = df.apply(calc_score, axis=1)

# 馬別偏差値算出
df_out = df.groupby('馬名')[col_score].agg(['mean','std']).reset_index()
df_out.columns = ['馬名','平均スコア','標準偏差']
mu, sigma = df_out['平均スコア'].mean(), df_out['平均スコア'].std()
df_out['偏差値'] = 50+10*(df_out['平均スコア']-mu)/sigma

# 結果表示
st.subheader('偏差値 上位6頭')
st.write(df_out.nlargest(6,'偏差値')[['馬名','偏差値']])

st.subheader('全馬スコア')
st.dataframe(df_out)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

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
df = pd.read_excel(uploaded_file, sheet_name=0)
cols = df.columns.tolist()

# 騎手あり/なし判定
has_jockey = '騎手' in cols

# 必要列設定
base_cols = ["馬名","頭数","クラス名","確定着順",
             "上がり3Fタイム","Ave-3F",
             "馬場状態","馬体重","増減","斤量","単勝オッズ"]
if has_jockey:
    base_cols.insert(1, '騎手')  # 騎手を2列目に

# 列抽出
try:
    df = df[base_cols]
except KeyError:
    st.error(f"必要な列が不足しています: {set(base_cols) - set(cols)}")
    st.stop()

# 型変換
num_cols = [c for c in base_cols if c not in ['馬名','騎手','クラス名','馬場状態']]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# パラメータ
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# スコア計算関数
def calc_score(row):
    N, p = row['頭数'], row['確定着順']
    GP = GRADE_SCORE.get(row['クラス名'],1)
    raw = GP*(N+1-p)
    denom = GP_MAX*N - GP_MIN
    raw_norm = (raw-GP_MIN)/denom if denom>0 else 0
    up3_norm = (row['Ave-3F']/row['上がり3Fタイム']) if row['上がり3Fタイム']>0 else 0
    odds_norm = 1/(1+np.log10(row['単勝オッズ'])) if row['単勝オッズ']>1 else 1
    jin_min, jin_max = df['斤量'].min(), df['斤量'].max()
    jin_norm = (jin_max-row['斤量'])/(jin_max-jin_min) if jin_max>jin_min else 0
    wdiff_norm = 1 - abs(row['増減'])/df['馬体重'].mean() if df['馬体重'].mean()>0 else 0
    # 騎手成績を考慮
    jockey_bonus = 0.1 if has_jockey else 0
    s = raw_norm*8 + up3_norm*2 + odds_norm + jin_norm + wdiff_norm + jockey_bonus
    return s/13*100

# スコア適用
col_score = 'Score'
df[col_score] = df.apply(calc_score, axis=1)

# 馬別偏差値算出
df_out = df.groupby('馬名')[col_score].agg(['mean','std']).reset_index()
df_out.columns = ['馬名','平均スコア','標準偏差']
mu, sigma = df_out['平均スコア'].mean(), df_out['平均スコア'].std()
df_out['偏差値'] = 50+10*(df_out['平均スコア']-mu)/sigma

# 結果表示
st.subheader('偏差値 上位6頭')
st.write(df_out.nlargest(6,'偏差値')[['馬名','偏差値']])

st.subheader('全馬スコア')
st.dataframe(df_out)
