import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("競馬スコア分析アプリ（完成版 8:2:1:1:1 重み、列位置指定版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み（列位置指定） ---
# 0:馬名,1:頭数,2:クラス名,3:確定着順,4:上がり3Fタイム,5:Ave-3F,
# 6:馬場状態,7:馬体重,8:増減,9:斤量,10:単勝オッズ

df = pd.read_excel(uploaded_file, sheet_name=0, header=None)
# 必要列抽出
cols_idx = [0,1,2,3,4,5,6,7,8,9,10]
df = df.iloc[:, cols_idx]
# 列名設定
df.columns = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F",
              "track_condition","weight","weight_diff","jinryo","odds"]

# --- 型変換 ---
for c in ["頭数","着順","上がり3F","Ave-3F","weight","weight_diff","jinryo","odds"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- パラメータ ---
GRADE_SCORE = {"GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5,
               "オープン特別":4, "3勝クラス":3, "2勝クラス":2,
               "1勝クラス":1, "新馬":1, "未勝利":1}
GP_MIN, GP_MAX = 1, 10

# --- スコア計算 ---
def calc_score(row):
    N, p = row[1], row[3]
    GP = GRADE_SCORE.get(row[2], 1)
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN)
    up3_norm = row[5] / row[4] if row[4] > 0 else 0
    odds_norm = 1 / (1 + np.log10(row[10])) if row[10] > 1 else 1
    mean_jin = df[9].mean()
    jin_diff_norm = 1 - abs(row[9] - mean_jin) / mean_jin
    wdiff_norm = 1 - abs(row[8]) / df[7].mean()
    score = raw_norm*8 + up3_norm*2 + odds_norm + jin_diff_norm + wdiff_norm
    return score / 13 * 100

df['Score'] = df.apply(calc_score, axis=1)
# 馬名列をキーに平均取得
df.columns = ["馬名","頭数","グレード","着順","上がり3F","Ave-3F",
              "track_condition","weight","weight_diff","jinryo","odds","Score"]

# --- 偏差値計算 ---
df_avg = df.groupby('馬名')['Score'].mean().reset_index()
df_avg.columns = ['馬名','平均スコア']
mean_score = df_avg['平均スコア'].mean()
std_score = df_avg['平均スコア'].std()
df_avg['偏差値'] = df_avg['平均スコア'].apply(lambda x: 50 + 10*(x-mean_score)/std_score)

# --- 棒グラフ ---
top6 = df_avg.nlargest(6,'偏差値')
fig,ax = plt.subplots(figsize=(8,5))
sns.barplot(x='偏差値',y='馬名',data=top6,ax=ax)
ax.set_title('偏差値 上位6頭')
st.pyplot(fig)

# --- 散布図 ---
df_std = df.groupby('馬名')['Score'].std().reset_index()
df_std.columns=['馬名','標準偏差']
avg2 = df_avg.merge(df_std,on='馬名')
fig2,ax2 = plt.subplots(figsize=(10,6))
x0,y0 = avg2['偏差値'].mean(),avg2['標準偏差'].mean()
xmin,xmax=avg2['偏差値'].min(),avg2['偏差値'].max()
ymin,ymax=avg2['標準偏差'].min(),avg2['標準偏差'].max()
ax2.fill_betweenx([ymin,y0],xmin,x0,color='#dff0d8',alpha=0.3)
ax2.fill_betweenx([ymin,y0],x0,xmax,color='#fcf8e3',alpha=0.3)
ax2.fill_betweenx([y0,ymax],xmin,x0,color='#d9edf7',alpha=0.3)
ax2.fill_betweenx([y0,ymax],x0,xmax,color='#f2dede',alpha=0.3)
ax2.axvline(x0,color='gray',linestyle='--');ax2.axhline(y0,color='gray',linestyle='--')
ax2.scatter(avg2['偏差値'],avg2['標準偏差'],s=20)
for i,r in avg2.iterrows(): ax2.text(r['偏差値'],r['標準偏差']+(i%3)*0.1,r['馬名'],ha='center',va='bottom',fontsize=8)
ax2.text((x0+xmax)/2,(y0+ymin)/2,'本命候補',ha='center',va='center')
ax2.text((x0+xmax)/2,(y0+ymax)/2,'抑え・穴狙い',ha='center',va='center')
ax2.text((xmin+x0)/2,(y0+ymax)/2,'軽視ゾーン',ha='center',va='center')
ax2.text((xmin+x0)/2,(y0+ymin)/2,'堅軸ゾーン',ha='center',va='center')
st.pyplot(fig2)

# --- テーブル ---
st.dataframe(df_avg)
