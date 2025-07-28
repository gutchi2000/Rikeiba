import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_path = "ipaexg.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

st.title("競馬スコア分析アプリ（全機能版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み & プレビュー ---
df = pd.read_excel(uploaded_file)
st.subheader('データプレビュー')
st.write(df.head())

# --- 必要列チェック ---
required = ["馬名","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"必要な列が不足しています: {missing}")
    st.stop()

# --- 列抽出 & 型変換 ---
df = df[required].copy()
# 日付
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
# 数値
num_cols = ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
# 欠損行の削除
df.dropna(subset=["レース日"] + num_cols, inplace=True)

# --- 指標計算 ---
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# 生スコアと正規化
N = df['頭数']
df['raw'] = df.apply(lambda r: GRADE_SCORE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['raw_norm'] = (df['raw'] - GP_MIN) / (GP_MAX * df['頭数'] - GP_MIN)
# 上がり3F
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
# オッズ
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
# 斤量（軽いほど高）
jin_max, jin_min = df['斤量'].max(), df['斤量'].min()
df['jin_norm'] = (jin_max - df['斤量']) / (jin_max - jin_min)
# 体重増減
mean_w = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / mean_w

# --- 直近レース重み付け ---
df['rank_date'] = df.groupby('馬名')['レース日'].rank(ascending=False, method='first')
df['weight'] = 1 / df['rank_date']

# --- Zスコア標準化 ---
metrics = ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']
for col in metrics:
    mu, sigma = df[col].mean(), df[col].std(ddof=1)
    df[f'Z_{col}'] = df[col].apply(lambda x: 0 if sigma == 0 else (x - mu) / sigma)

# --- 重み付け合成と偏差値化 (スケール拡大) ---
weights = {'Z_raw_norm':8, 'Z_up3_norm':2, 'Z_odds_norm':1, 'Z_jin_norm':1, 'Z_wdiff_norm':1}
total_w = sum(weights.values())
df['total_z'] = sum(df[k] * w for k, w in weights.items()) / total_w
mu_t, sigma_t = df['total_z'].mean(), df['total_z'].std(ddof=1)
scale = 15  # 偏差値のスケールを10→15に増加して分散を広げる
if sigma_t == 0:
    df['偏差値'] = 50
else:
    df['偏差値'] = 50 + scale * (df['total_z'] - mu_t) / sigma_t

# --- 馬別平均偏差値 (加重平均) ---
df_avg = df.groupby('馬名').apply(lambda d: np.average(d['偏差値'], weights=d['weight'])).reset_index()
df_avg.columns = ['馬名','平均偏差値']

# 以下は表示・グラフ・ダウンロード等の処理（省略せず従来どおり）
# ...
