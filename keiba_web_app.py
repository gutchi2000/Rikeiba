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

st.title("競馬スコア分析アプリ（完成版）")

# Excelアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# シート1: 過去成績データ（ヘッダー有無対応）
col_req = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F"]
try:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    if not all(c in df.columns for c in col_req):
        df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        df.columns = col_req
except Exception as e:
    st.error(f"シート1読み込み失敗: {e}")
    st.stop()
df = df[col_req]

# スコア計算パラメータ
GRADE_SCORE = {
    "GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
    "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
    "1勝クラス":1,"新馬":1,"未勝利":1
}
GP_MIN, GP_MAX = 1, 10

# — 0. GP（グレード点数）の説明 —
# GP = GⅠ:10, GⅡ:8, GⅢ:6, リステッド:5, オープン特別:4, 3勝クラス:3, 2勝クラス:2, 1勝クラス:1, 新馬・未勝利:1

# — 1. RawScore & Score —
GP_min, GP_max = 1, 10
# RawScore = GP × (頭数 +1 − 着順)
df['RawScore'] = df['Score'] = df.apply(lambda row: GRADE_SCORE.get(row['グレード'],1) * (row['頭数']+1-row['着順']), axis=1)
# Score = normalized 0–100
df['Score'] = (df['RawScore'] - GP_min) / (GP_max * df['頭数'] - GP_min) * 100

# — 2. 上がり3F スコア化 —
# (a) last3f fastest=100基準
T_min = df['上がり3F'].min()
df['S_L3F'] = T_min / df['上がり3F'] * 100
# (b) average3F 補正スコア
df['S_A3F'] = (df['上がり3F'].mean() - df['上がり3F']) / df['上がり3F'].mean() * 100

# — 3. 馬場状態補正 —
track_coef = {'良':1.00,'稍重':0.98,'重':0.95,'不良':0.92}
df['c_track'] = df['track_condition'].map(track_coef)

# — 4. 重み付け合成 —
w1, w2, w3 = 0.5, 0.3, 0.2
df['S_comb'] = w1*df['Score'] + w2*df['S_L3F'] + w3*df['S_A3F']
# 最終スコア
df['final_score'] = df['S_comb'] * df['c_track']

# — 5. 偏差値化 —
mean_fs = df['final_score'].mean()
std_fs = df['final_score'].std(ddof=0)
df['偏差値'] = 50 + 10*(df['final_score'] - mean_fs)/std_fs

# 馬別集計
avg = df.groupby('馬名')[['final_score','偏差値']].mean().reset_index()
avg.columns = ['馬名','平均最終スコア','偏差値']

# — 6. 出力 —
st.subheader('馬別スコア一覧')
st.dataframe(avg)

# 棒グラフ: 偏差値上位6頭
top6 = avg.sort_values('偏差値',ascending=False).head(6)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='偏差値',y='馬名',data=top6,ax=ax)
ax.set_title('偏差値 上位6頭',fontproperties=jp_font)
st.pyplot(fig)

# 散布図: 偏差値 vs 平均最終スコア
st.subheader('偏差値 × 平均最終スコア')
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.scatterplot(data=avg, x='偏差値', y='平均最終スコア', hue='馬名',s=100,ax=ax2)
ax2.set_title('偏差値 × 平均最終スコア',fontproperties=jp_font)
st.pyplot(fig2)

avg = df.groupby("馬名")["Score"].mean().reset_index()
avg.columns = ["馬名","平均スコア"]
m, s = avg["平均スコア"].mean(), avg["平均スコア"].std()
avg["偏差値"] = avg["平均スコア"].apply(lambda x: 50 + 10 * (x - m) / s)

# 安定性（標準偏差）
stds = df.groupby("馬名")["Score"].std().reset_index()
stds.columns = ["馬名","スコア標準偏差"]
avg = avg.merge(stds, on="馬名")

# 直近3走の加重偏差値
df["順"] = df.groupby("馬名").cumcount(ascending=False) + 1
recent = df.sort_values(["馬名","順"]).groupby("馬名").head(3)
def wavg(x): return np.average(x[::-1], weights=[3,2,1][:len(x)][::-1])
w = recent.groupby("馬名")["Score"].apply(wavg).reset_index()
w.columns = ["馬名","加重平均スコア"]

mw, sw = w["加重平均スコア"].mean(), w["加重平均スコア"].std()
w["加重平均偏差値"] = w["加重平均スコア"].apply(lambda x: 50 + 10 * (x - mw) / sw)
avg = avg.merge(w, on="馬名")

# — グラフを最初に表示 —
# 棒グラフ: 偏差値上位6頭
st.subheader("偏差値 上位6頭（棒グラフ）")
top6 = avg.sort_values("偏差値", ascending=False).head(6)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="偏差値", y="馬名", data=top6, ax=ax)
ax.set_title("偏差値 上位6頭", fontproperties=jp_font)
ax.set_xlabel("偏差値", fontproperties=jp_font)
ax.set_ylabel("馬名", fontproperties=jp_font)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=jp_font)
st.pyplot(fig)

# 散布図: 調子(加重偏差値)×安定性（馬名ラベル版）
st.subheader("調子(加重偏差値)×安定性（馬名ラベル）")
fig2, ax2 = plt.subplots(figsize=(10,6))
# 散布点は目立たない色・小さめサイズで描画
ax2.scatter(avg['偏差値'], avg['加重平均偏差値'], s=20, color='gray')
# 各点に馬名ラベルを付与
for _, row in avg.iterrows():
    # ラベルが重ならないようにオフセットして表示
    ax2.annotate(
        row['馬名'],
        (row['偏差値'], row['加重平均偏差値']),
        fontproperties=jp_font,
        fontsize=9,
        textcoords="offset points",
        xytext=(5,5),
        ha='left', va='bottom'
    )
# 軸ラベル・タイトル
ax2.set_title("調子(加重偏差値)×安定性", fontproperties=jp_font)
ax2.set_xlabel("偏差値", fontproperties=jp_font)
ax2.set_ylabel("加重平均偏差値", fontproperties=jp_font)
# 目盛フォント
for lbl in ax2.get_xticklabels(): lbl.set_fontproperties(jp_font)
for lbl in ax2.get_yticklabels(): lbl.set_fontproperties(jp_font)
# レジェンド削除
if ax2.get_legend():
    ax2.get_legend().remove()
st.pyplot(fig2)

# ——— EMA(指数移動平均)によるスコア ———
# 直近の Score を指数移動平均で平滑化（span=3）
df['rorder'] = df.groupby('馬名').cumcount(ascending=True) + 1
ema = (
    df.sort_values(['馬名','rorder'])
      .groupby('馬名')['Score']
      .apply(lambda x: x.ewm(span=3, adjust=False).mean().iloc[-1])
      .reset_index(name='emaスコア')
)
# ema偏差値化
m_ema, s_ema = ema['emaスコア'].mean(), ema['emaスコア'].std()
ema['ema偏差値'] = ema['emaスコア'].apply(lambda x: 50 + 10 * (x - m_ema) / s_ema)
# avg に結合
avg = avg.merge(ema, on='馬名')

# 最後にテーブル表示
st.subheader("馬別スコア一覧（テーブル）")
st.dataframe(avg)
st.subheader("馬別スコア一覧（テーブル）")
st.dataframe(avg)
