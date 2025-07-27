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
# 列名ベースに修正した calc_score

def calc_score(row):
    # 頭数, 着順, グレード
    N = row['頭数']
    p = row['着順']
    GP = GRADE_SCORE.get(row['グレード'], 1)
    raw = GP * (N + 1 - p)
    raw_norm = (raw - GP_MIN) / (GP_MAX * N - GP_MIN)
    # 上がり3F正規化
    up3 = row['上がり3F']
    ave3 = row['Ave-3F']
    up3_norm = ave3 / up3 if up3 > 0 else 0
    # オッズ正規化
    odds = row['odds']
    odds_norm = 1 / (1 + np.log10(odds)) if odds > 1 else 1
    # 斤量差正規化
    jin = row['jinryo']
    mean_jin = df['jinryo'].mean()
    jin_diff_norm = 1 - abs(jin - mean_jin) / mean_jin
    # 馬体重増減正規化
    wdiff = row['weight_diff']
    mean_weight = df['weight'].mean()
    wdiff_norm = 1 - abs(wdiff) / mean_weight
    # 合成スコア (8+2+1+1+1=13)
    score = raw_norm*8 + up3_norm*2 + odds_norm*1 + jin_diff_norm*1 + wdiff_norm*1
    return score / 13 * 100

# スコア適用
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
