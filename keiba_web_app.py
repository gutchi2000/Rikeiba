import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import io

# --- ヘルパー関数 ---
def z_score(series: pd.Series) -> pd.Series:
    """
    与えられたSeriesを偏差値（平均50, 標準偏差10）に変換して返す
    """
    mu = series.mean()
    sigma = series.std(ddof=0)
    return 50 + 10 * (series - mu) / sigma


def grade_mark(z: float) -> str:
    """
    偏差値に基づき印（◎◯▲☆△×）を返す
    """
    if z >= 70:
        return "◎"
    if z >= 60:
        return "〇"
    if z >= 50:
        return "▲"
    if z >= 40:
        return "☆"
    if z >= 30:
        return "△"
    return "×"


def season_of(month: int) -> str:
    """
    月から四季を判定して文字列を返す
    """
    if 3 <= month <= 5:
        return '春'
    if 6 <= month <= 8:
        return '夏'
    if 9 <= month <= 11:
        return '秋'
    return '冬'

# --- サイドバー設定 ---
st.sidebar.header("パラメータ設定")
lambda_part = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)
hist_weight = 1.0 - orig_weight

st.sidebar.subheader("性別重み")
gender_w = {
    '牡': st.sidebar.slider('牡馬', 0.0, 2.0, 1.0, 0.1),
    '牝': st.sidebar.slider('牝馬', 0.0, 2.0, 1.0, 0.1),
    'セ': st.sidebar.slider('せん馬', 0.0, 2.0, 1.0, 0.1)
}

st.sidebar.subheader("脚質重み")
style_w = {
    '逃げ': st.sidebar.slider('逃げ', 0.0, 2.0, 1.0, 0.1),
    '先行': st.sidebar.slider('先行', 0.0, 2.0, 1.0, 0.1),
    '差し': st.sidebar.slider('差し', 0.0, 2.0, 1.0, 0.1),
    '追込': st.sidebar.slider('追込', 0.0, 2.0, 1.0, 0.1)
}

st.sidebar.subheader("四季重み")
season_w = {s: st.sidebar.slider(f'{s}', 0.0, 2.0, 1.0, 0.1) for s in ['春','夏','秋','冬']}

age_w = st.sidebar.number_input("年齢重み（全馬共通）", 0.0, 5.0, 1.0, 0.1)

st.sidebar.subheader("枠順重み")
frame_w = {str(i): st.sidebar.slider(f'{i}枠', 0.0, 2.0, 1.0, 0.1) for i in range(1,10)}

besttime_w = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0, 0.1)

total_budget = st.sidebar.slider("合計予算 (円)", 500, 50000, 10000, 500)
scenario = st.sidebar.selectbox("シナリオ", ['通常', 'ちょい余裕', '余裕'])

# --- メイン画面 ---
st.title("競馬予想アプリ")

# ファイルアップロード
st.subheader("ファイルアップロード")
excel_file = st.file_uploader("成績アップロード（Excel）", type=['xlsx'])
html_file = st.file_uploader("血統アップロード（HTML）", type=['html'])
if not excel_file or not html_file:
    st.info("Excel と HTML を両方アップロードしてください。")
    st.stop()

# Excelデータ読み込み
df = pd.read_excel(excel_file, sheet_name=0)

# HTML血統データ読み込み（pandas.read_htmlで直接抽出）
# read_htmlはファイル-likeオブジェクトも受け付ける
blood_tables = pd.read_html(html_file)
blood_df = blood_tables[0].iloc[:, :2]
blood_df.columns = ['馬名', '血統']

# データ結合
df = df.merge(blood_df, on='馬名', how='left')

# 血統キーワード入力
st.subheader("血統キーワード")
keywords = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bonus_point = st.slider("血統ボーナス点数", 0, 20, 5)

# 馬一覧編集
st.subheader("馬一覧と補正設定")
base_cols = ['枠', '馬名', '性別', '年齢']
df_edit = df[base_cols].copy()
df_edit['脚質'] = ''
df_edit['斤量'] = 0
edited = st.experimental_data_editor(df_edit, num_rows='dynamic')

# スコア計算（各レース行）
def calc_score(row):
    GP_map = {"GⅠ":10, "GⅡ":8, "GⅢ":6, "リステッド":5, "オープン特別":4,
              "3勝クラス":3, "2勝クラス":2, "1勝クラス":1, "新馬・未勝利":1}
    gp = GP_map.get(row['クラス名'], 1)
    N, p = row['頭数'], row['確定着順']
    raw = gp * (N + 1 - p) + lambda_part * gp
    # 季節判定
    date = pd.to_datetime(row['レース日'], errors='coerce')
    sw = season_w[season_of(date.month)]
    # 属性補正
    gw = gender_w.get(row['性別'], 1.0)
    stw = style_w.get(edited.loc[edited['馬名']==row['馬名'], '脚質'].values[0], 1.0)
    fw = frame_w.get(str(row['枠']), 1.0)
    aw = age_w
    bt = besttime_w
    # 血統補正
    blood_bonus = bonus_point if any(k in str(row['血統']) for k in keywords) else 0
    return raw * sw * gw * stw * fw * aw * bt + blood_bonus

# レース結果ごとにスコア計算
df['score_raw'] = df.apply(calc_score, axis=1)
# 正規化 → 偏差値化
df['score_norm'] = (df['score_raw'] - df['score_raw'].min()) / (df['score_raw'].max() - df['score_raw'].min()) * 100

# 馬ごとの統計（平均偏差値、標準偏差）
agg = df.groupby('馬名')['score_norm'].agg(['mean','std']).reset_index()
agg.columns = ['馬名', 'AvgZ', 'Stdev']
agg['Stability'] = -agg['Stdev']
agg['RankZ'] = z_score(agg['AvgZ'])

# 散布図描画
st.subheader("偏差値 vs 安定度 散布図")
fig, ax = plt.subplots()
ax.scatter(agg['RankZ'], agg['Stability'])
# 四象限線・ラベル
avg_st = agg['Stability'].mean()
ax.axvline(50, color='gray'); ax.axhline(avg_st, color='gray')
ax.text(60, avg_st + max(agg['Stability']) * 0.1, '一発警戒')
ax.text(40, avg_st + max(agg['Stability']) * 0.1, '警戒必須')
ax.text(60, avg_st - max(agg['Stability']) * 0.1, '鉄板級')
ax.text(40, avg_st - max(agg['Stability']) * 0.1, '堅実型')
st.pyplot(fig)

# 上位6頭印付け
top6 = agg.sort_values('RankZ', ascending=False).head(6)
top6['印'] = ['◎','〇','▲','☆','△','△']
st.subheader("上位6頭")
st.table(top6[['馬名', '印']])

# 資金配分 and 買い目生成
pur1 = total_budget * 0.25  # 単勝
pur2 = total_budget * 0.75  # 複勝
rem = total_budget - (pur1 + pur2)
if scenario == '通常':
    parts = ['馬連','ワイド','馬単']
elif scenario == 'ちょい余裕':
    parts = ['馬連','ワイド','馬単','三連複']
else:
    parts = ['馬連','ワイド','馬単','三連複','三連単']
bet_share = {p: rem / len(parts) for p in parts}

st.subheader("買い目と配分（円）")
st.write(f"単勝: {pur1:.0f}円, 複勝: {pur2:.0f}円")
st.table(pd.DataFrame.from_dict(bet_share, orient='index', columns=['金額']))

st.subheader("推奨買い目例")
st.write("単勝:", top6.iloc[0]['馬名'])
st.write("複勝:", top6.iloc[1]['馬名'])
st.write("馬連/ワイド/馬単:", f"{top6.iloc[0]['馬名']}-{top6.iloc[1]['馬名']}")
st.write("三連複:", f"{top6.iloc[0]['馬名']}-{','.join(top6.iloc[1:5]['馬名'])}")
st.write("三連単:", f"{top6.iloc[0]['馬名']} 軸→{','.join(top6.iloc[1:6]['馬名'])}")
