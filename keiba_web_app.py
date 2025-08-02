import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import altair as alt
from itertools import combinations
from itertools import product

# 日本語フォント設定（matplotlib用）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']

# --- ヘルパー関数 ---
def z_score(s: pd.Series) -> pd.Series:
    return 50 + 10 * (s - s.mean()) / s.std(ddof=0)

def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

# --- サイドバー設定 ---
st.sidebar.header("パラメータ設定")
lambda_part  = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight  = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)
hist_weight  = 1 - orig_weight

with st.sidebar.expander("性別重み", expanded=False):
    gender_w = {g: st.slider(g, 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
with st.sidebar.expander("脚質重み", expanded=False):
    style_w  = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['逃げ','先行','差し','追込']}
with st.sidebar.expander("四季重み", expanded=False):
    season_w = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
age_w        = st.sidebar.number_input("年齢重み", 0.0, 5.0, 1.0)
with st.sidebar.expander("枠順重み", expanded=False):
    frame_w = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0) for i in range(1,9)}
besttime_w   = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0)
weight_coeff = st.sidebar.slider("斤量効果強度", 0.0, 2.0, 1.0)
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)  # ステップを100円単位に変更
scenario     = st.sidebar.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])

# --- メイン画面 ---
st.title("競馬予想アプリ（完成版）")

# --- ファイルアップロード ---
st.subheader("ファイルアップロード")
excel_file = st.file_uploader("Excel (成績＆属性)", type='xlsx')
html_file  = st.file_uploader("HTML (血統)", type='html')
if not excel_file or not html_file:
    st.info("ExcelとHTMLを両方アップロードしてください。")
    st.stop()

# --- データ読み込み ---
df_score = pd.read_excel(excel_file, sheet_name=0)
sheet2   = pd.read_excel(excel_file, sheet_name=1)
# 重複行を削除して、出走メンバー18頭のみ取得
# "馬名"列で一意化（月並みに最初に出現した順）
sheet2 = sheet2.drop_duplicates(subset=sheet2.columns[2], keep='first').reset_index(drop=True)
# --- シート2から今回出走馬一覧を取得 ---
# 必要な列を位置で取得: 枠(0), 番(1), 馬名(2), 性別(3), 年齢(4)
sheet2 = pd.read_excel(excel_file, sheet_name=1)
attrs = sheet2.iloc[:, [0,1,2,3,4]].copy()
attrs.columns = ['枠','番','馬名','性別','年齢']
# 脚質の入力用列を追加
attrs['脚質'] = ''
attrs['斤量'] = np.nan

# --- 馬一覧編集 ---
st.subheader("馬一覧と脚質入力")
# 独自表: 枠, 番, 馬名, 性別, 年齢, 脚質
edited = st.data_editor(
    attrs,
    column_order=['枠','番','馬名','性別','年齢','脚質'],
    column_config={
        '脚質': st.column_config.SelectboxColumn(
            '脚質', options=['逃げ','先行','差し','追込']
        )
    },
    use_container_width=True,
    num_rows='static'
)
# 編集後のテーブルを horses に反映
horses = edited.copy()[['枠','番','馬名','性別','年齢','脚質']]

# --- 血統HTMLパース ---
cont = html_file.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont)
blood = []
for r in rows:
    c = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)
    if len(c) >= 2:
        blood.append((re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip()))
blood_df = pd.DataFrame(blood, columns=['馬名','血統'])

# --- データ結合 ---
df_score = (
    df_score
    .merge(horses, on='馬名', how='inner')
    .merge(blood_df, on='馬名', how='left')
)

# --- 血統キーワード入力 ---
st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

# --- スコア計算 ---

style_map = dict(zip(horses['馬名'], horses['脚質']))
def calc_score(r):
    GP = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
          '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
    g = GP.get(r['クラス名'], 1)
    raw = g * (r['頭数'] + 1 - r['確定着順']) + lambda_part * g
    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r['性別'], 1)
    stw = style_w.get(r['脚質'], 1)
    fw  = frame_w.get(str(r['枠']), 1)
    aw  = age_w
    bt  = besttime_w
    # 斤量補正なし
    weight_factor = 1
    # 血統ボーナス
    bonus = bp if any(k in str(r.get('血統', '')) for k in keys) else 0
    bonus = bp if any(k in str(r.get('血統', '')) for k in keys) else 0
    # 最終スコア
    return raw * sw * gw * stw * fw * aw * bt * weight_factor + bonus

# スコア適用
" + 
"df_score['score_raw']  = df_score.apply(calc_score, axis=1)
" +
"df_score['score_norm'] = (
" +
"    (df_score['score_raw'] - df_score['score_raw'].min()) /
" +
"    (df_score['score_raw'].max() - df_score['score_raw'].min()) * 100
)

" +
"# --- 過去5走重み設定 ---
" +
"w1 = st.sidebar.slider("1走前の重み", 0.0, 2.0, 1.0, 0.01)
" +
"w2 = st.sidebar.slider("2走前の重み", 0.0, 2.0, 1.0, 0.01)
" +
"w3 = st.sidebar.slider("3走前の重み", 0.0, 2.0, 1.0, 0.01)
" +
"w4 = st.sidebar.slider("4走前の重み", 0.0, 2.0, 1.0, 0.01)
" +
"w5 = st.sidebar.slider("5走前の重み", 0.0, 2.0, 1.0, 0.01)
" +
"weights = {1:w1, 2:w2, 3:w3, 4:w4, 5:w5}
" +
"weights_sum = w1 + w2 + w3 + w4 + w5

" +
"# --- 過去5走スコア重み付け ---
" +
"df_score['レース日'] = pd.to_datetime(df_score['レース日'])
" +
"df_score = df_score.sort_values(['馬名','レース日'], ascending=[True, False])
" +
"df_score['race_order'] = df_score.groupby('馬名').cumcount()+1
" +
"df_score['score_hist'] = df_score.apply(lambda r: r['score_norm']*weights.get(r['race_order'],0), axis=1)
" +
"
" +
"# --- 馬ごとの過去5走加重平均 ---
" +
"df_hist = (
" +
"    df_score[df_score['race_order']<=5]
" +
"    .groupby('馬名')['score_hist'].sum().reset_index()
" +
")
" +
"df_hist['HistAvg'] = df_hist['score_hist']/weights_sum

" +
"# --- 馬ごとの統計 ---
" +
"df_agg = (
" +
"    df_hist
" +
"    .merge(df_agg[['馬名']], on='馬名', how='right')  # 順序維持
" +
"    .fillna(0)
" +
"    .assign(AvgZ=lambda d: d['HistAvg'], Stdev=0)
" +
"    .loc[:, ['馬名','AvgZ','Stdev']]
" +
")
" +
"df_agg['Stability'] = -df_agg['Stdev']
" +
"df_agg['RankZ'] = z_score(df_agg['AvgZ']) (
    (df_score['score_raw'] - df_score['score_raw'].min()) /
    (df_score['score_raw'].max() - df_score['score_raw'].min()) * 100
)

# --- 馬ごとの統計 ---
df_agg = (
    df_score.groupby('馬名')['score_norm']
    .agg(['mean','std']).reset_index()
)
df_agg.columns     = ['馬名','AvgZ','Stdev']
df_agg['Stability'] = -df_agg['Stdev']
df_agg['RankZ']     = z_score(df_agg['AvgZ'])

# --- 散布図（Altair テキスト付き + 象限ラベル） ---
st.subheader("偏差値 vs 安定度 散布図")
# 四象限ラベル用データ
avg_st = df_agg['Stability'].mean()
quad_labels = pd.DataFrame([
    {'RankZ':75, 'Stability': avg_st + (df_agg['Stability'].max()-avg_st)/2, 'label':'一発警戒'},
    {'RankZ':25, 'Stability': avg_st + (df_agg['Stability'].max()-avg_st)/2, 'label':'警戒必須'},
    {'RankZ':75, 'Stability': avg_st - (avg_st-df_agg['Stability'].min())/2, 'label':'鉄板級'},
    {'RankZ':25, 'Stability': avg_st - (avg_st-df_agg['Stability'].min())/2, 'label':'堅実型'}
])
# 基本散布図
points = alt.Chart(df_agg).mark_circle(size=100).encode(
    x=alt.X('RankZ:Q', title='偏差値'),
    y=alt.Y('Stability:Q', title='安定度'),
    tooltip=['馬名','AvgZ','Stdev']
)
# 馬名テキスト
labels = alt.Chart(df_agg).mark_text(dx=5, dy=-5, fontSize=10, color='white').encode(
    x='RankZ:Q',
    y='Stability:Q',
    text='馬名:N'
)
# 象限ラベル
quad = alt.Chart(quad_labels).mark_text(fontSize=14, fontWeight='bold', color='white').encode(
    x='RankZ:Q',
    y='Stability:Q',
    text='label:N'
)
# 中心線
vline = alt.Chart(pd.DataFrame({'x':[50]})).mark_rule(color='gray').encode(x='x:Q')
hline = alt.Chart(pd.DataFrame({'y':[avg_st]})).mark_rule(color='gray').encode(y='y:Q')
# 合成
chart = (points + labels + quad + vline + hline)
st.altair_chart(chart.properties(width=600, height=400).interactive(), use_container_width=True)

# --- 偏差値フィルター ---
st.sidebar.subheader("偏差値フィルター")
z_cut = st.sidebar.slider("最低偏差値", float(df_agg['RankZ'].min()), float(df_agg['RankZ'].max()), 50.0)

# --- 散布図下に馬名＆偏差値テーブル ---
st.subheader("馬名と偏差値一覧（偏差値>=%0.1f）" % z_cut)
filtered = df_agg[df_agg['RankZ'] >= z_cut].sort_values('RankZ', ascending=False)
st.table(filtered[['馬名','RankZ']].rename(columns={'RankZ':'偏差値'}))

# --- 上位6頭印付け & 買い目生成 ---
top6 = df_agg.sort_values('RankZ', ascending=False).head(6)
top6['印'] = ['◎','〇','▲','☆','△','△']
st.subheader("上位6頭")
st.table(top6[['馬名','印']])

# --- サイドバーからの変数取得は省略 ---
# total_budget, scenario, top6, etc. が定義済みとします

# ——————————————
# ―― 資金配分 〜 最終買い目一覧 （完成形） ――
# ——————————————

# ◎／〇 の馬リスト準備
h1 = top6.iloc[0]['馬名']                  # ◎馬
h2 = top6.iloc[1]['馬名']                  # 〇馬
symbols = top6['印'].tolist()              # ['◎','〇','▲','☆','△','△']
names   = top6['馬名'].tolist()            # [h1,h2,h3,h4,h5,h6]
others_names   = names[1:]                 # ['〇馬名','▲馬名','☆馬名','△馬名','△馬名']
others_symbols = symbols[1:]               # ['〇','▲','☆','△','△']

# ——————————————
# ◎／〇 の馬リスト準備（省略）

# --- シナリオ別 券種リスト定義 ---
three = ['馬連','ワイド','馬単']
scenario_map = {
    '通常': three,
    'ちょい余裕': ['ワイド','三連複'],
    '余裕': ['ワイド','三連複','三連単']
}

# --- 資金配分計算 ---
main_share = 0.5
pur1 = int(round((total_budget * main_share * 1/4)  / 100) * 100)
pur2 = int(round((total_budget * main_share * 3/4)  / 100) * 100)
rem  = total_budget - (pur1 + pur2)

win_each   = int(round((pur1 / 2)  / 100) * 100)
place_each = int(round((pur2 / 2)  / 100) * 100)

st.subheader("■ 最終買い目一覧")
```
