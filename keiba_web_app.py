import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import altair as alt

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
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)  # 100円刻み
scenario     = st.sidebar.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])

st.title("競馬予想アプリ（完成版）")

# --- ファイルアップロード ---
excel_file = st.file_uploader("Excel (成績＆属性)", type='xlsx')
html_file  = st.file_uploader("HTML (血統)", type='html')
if not excel_file or not html_file:
    st.info("ExcelとHTMLを両方アップロードしてください。")
    st.stop()

# --- データ読み込み・前処理 ---
df_score = pd.read_excel(excel_file, sheet_name=0)
sheet2   = pd.read_excel(excel_file, sheet_name=1).drop_duplicates(subset=2, keep='first')
attrs = sheet2.iloc[:, [0,1,2,3,4]].copy()
attrs.columns = ['枠','番','馬名','性別','年齢']
attrs['脚質'] = ''
horses = st.data_editor(
    attrs, 
    column_order=['枠','番','馬名','性別','年齢','脚質'],
    column_config={'脚質': st.column_config.SelectboxColumn('脚質',
                     options=['逃げ','先行','差し','追込'])},
    use_container_width=True, num_rows='static'
)
horses = horses[['枠','番','馬名','性別','年齢','脚質']]

cont = html_file.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont)
blood = [(re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip())
         for r in rows
         for c in [re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)]
         if len(c)>=2]
blood_df = pd.DataFrame(blood, columns=['馬名','血統'])

df_score = df_score.merge(horses, on='馬名', how='inner').merge(blood_df, on='馬名', how='left')

# --- 血統ボーナス設定 ---
st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

# --- スコア計算 ---
def calc_score(r):
    GP = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
          '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
    g = GP.get(r['クラス名'], 1)
    raw = g * (r['頭数'] + 1 - r['確定着順']) + lambda_part * g
    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r['性別'], 1)
    stw = style_w.get(r['脚質'], 1)
    fw  = frame_w.get(str(r['枠']), 1)
    aw, bt = age_w, besttime_w
    bonus = bp if any(k in str(r.get('血統','')) for k in keys) else 0
    return raw * sw * gw * stw * fw * aw * bt + bonus

df_score['score_raw']  = df_score.apply(calc_score, axis=1)
df_score['score_norm'] = (df_score['score_raw'] - df_score['score_raw'].min()) / (df_score['score_raw'].max() - df_score['score_raw'].min()) * 100

# --- 統計・偏差値算出 ---
df_agg = df_score.groupby('馬名')['score_norm'].agg(['mean','std']).reset_index()
df_agg.columns = ['馬名','AvgZ','Stdev']
df_agg['Stability'] = -df_agg['Stdev']
df_agg['RankZ']     = z_score(df_agg['AvgZ'])

# --- 可視化・フィルター ---
st.subheader("偏差値 vs 安定度")
avg_st = df_agg['Stability'].mean()
# (略：Altair散布図コード as before)

st.sidebar.subheader("最低偏差値フィルター")
z_cut = st.sidebar.slider("最低偏差値", float(df_agg['RankZ'].min()), float(df_agg['RankZ'].max()), 50.0)
filtered = df_agg[df_agg['RankZ'] >= z_cut].sort_values('RankZ', ascending=False)
st.table(filtered[['馬名','RankZ']].rename(columns={'RankZ':'偏差値'}))

# --- 上位6頭 & 印づけ ---
top6 = df_agg.sort_values('RankZ', ascending=False).head(6)
top6['印'] = ['◎','〇','▲','☆','△','△']
st.subheader("上位6頭")
st.table(top6[['馬名','印']])

# --- 資金配分＆買い目生成(補正済) ---
# ① 単勝・複勝
main_share = 0.5
pur_win   = round((total_budget * main_share * 0.25) / 100) * 100
pur_place = round((total_budget * main_share * 0.75) / 100) * 100

# ② 残額をシナリオ別に
rem = total_budget - (pur_win + pur_place)
parts_map = {
    '通常':     ['choice'],
    'ちょい余裕': ['choice','trifecta'],
    '余裕':     ['choice','trifecta','trifecta_s'],
}
parts = parts_map[scenario]
share_each = rem / len(parts)

# ③ ◎, 〇, ▲, ☆, △①, △② を取得
tansho  = top6.iloc[0]['馬名']
fukusho = top6.iloc[1]['馬名']
others5 = list(top6.iloc[2:7]['馬名'])  # 3着以降５頭

# ④ 購入券種選択
with st.expander("馬連・ワイド・馬単から選択", expanded=False):
    choice = st.radio(
        "購入する券種を選択してください",
        ['馬連','ワイド','馬単'],
        index=1,
        key="purchase_choice"
    )
    st.write(f"▶︎ 選択：{choice}  に {int(round(share_each/100))*100}円")

# ⑤ 組み合わせリスト
combos = {
    '単勝':   [tansho, fukusho],
    '複勝':   [tansho, fukusho],
    'choice': [f"{tansho}-{m}" if choice!='馬単' else f"{tansho}>{m}" for m in others5],
    'trifecta':  [f"{tansho}-{fukusho}-{m}" for m in others5],
    'trifecta_s':[f"{tansho}>{fukusho}>{m}" for m in others5],
}

alloc = {'単勝': pur_win, '複勝': pur_place}
alloc.update({p: int(round(share_each/100))*100 for p in parts})

# ⑥ 配分・テーブル化
bets = []
# 単勝・複勝
for kind in ['単勝','複勝']:
    lst, total = combos[kind], alloc[kind]
    n = len(lst); amt = total//n//100*100; rem0 = total - amt*n
    for i, h in enumerate(lst):
        bets.append({"券種":kind,"組み合わせ":h,"金額":amt + (rem0 if i==0 else 0)})

# choice, trifecta, trifecta_s
for key in parts:
    lst, total = combos[key], alloc[key]
    n = len(lst); amt = total//n//100*100; rem0 = total - amt*n
    label = choice if key=='choice' else ('三連複' if key=='trifecta' else '三連単')
    for i, c in enumerate(lst):
        bets.append({"券種":label,"組み合わせ":c,"金額":amt + (rem0 if i==0 else 0)})

st.subheader("推奨買い目一覧と配分（円）")
st.table(pd.DataFrame(bets, columns=["券種","組み合わせ","金額"]))
