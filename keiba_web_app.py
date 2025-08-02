import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
import matplotlib.pyplot as plt
from itertools import combinations

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']

def z_score(s: pd.Series) -> pd.Series:
    return 50 + 10 * (s - s.mean()) / s.std(ddof=0)

def season_of(month: int) -> str:
    if 3 <= month <= 5:
        return '春'
    if 6 <= month <= 8:
        return '夏'
    if 9 <= month <= 11:
        return '秋'
    return '冬'

st.sidebar.header("パラメータ設定")
lambda_part = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)
scenario = st.sidebar.selectbox("シナリオ", ['通常', 'ちょい余裕', '余裕'])
lambda_decay = st.sidebar.slider("時系列減衰 λ", 0.0, 1.0, 0.1, 0.01)

with st.sidebar.expander("重み設定", expanded=False):
    gender_w = {g: st.slider(g, 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
    style_w  = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['逃げ','先行','差し','追込']}
    season_w = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
    age_w    = st.number_input("年齢重み", 0.0, 5.0, 1.0)
    frame_w  = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0) for i in range(1, 9)}
    besttime_w = st.slider("ベストタイム重み", 0.0, 2.0, 1.0)

st.title("競馬予想アプリ（完成版）")

st.subheader("ファイルアップロード")
excel_file = st.file_uploader("Excel (成績＆属性)", type='xlsx')
html_file  = st.file_uploader("HTML (血統)", type='html')
if not excel_file or not html_file:
    st.info("ExcelとHTMLを両方アップロードしてください。")
    st.stop()

df_score = pd.read_excel(excel_file, sheet_name=0)
sheet2   = pd.read_excel(excel_file, sheet_name=1)
attrs    = sheet2.drop_duplicates(subset=sheet2.columns[2], keep='first').iloc[:, [0,1,2,3,4]]
attrs.columns = ['枠','番','馬名','性別','年齢']
attrs['脚質'] = ''

st.subheader("馬一覧と脚質入力")
edited = st.data_editor(
    attrs,
    column_order=['枠','番','馬名','性別','年齢','脚質'],
    column_config={'脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込'])},
    use_container_width=True,
    num_rows='static'
)
horses = edited[['枠','番','馬名','性別','年齢','脚質']]

cont = html_file.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont)
blood = []
for r in rows:
    c = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)
    if len(c) >= 2:
        blood.append((re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip()))
blood_df = pd.DataFrame(blood, columns=['馬名','血統'])

df_score = df_score.merge(horses, on='馬名', how='inner').merge(blood_df, on='馬名', how='left')

st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

GP = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
def calc_score(r):
    g = GP.get(r['クラス名'], 1)
    raw = g * (r['頭数'] + 1 - r['確定着順']) + lambda_part * g
    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r['性別'], 1)
    stw = style_w.get(r['脚質'], 1)
    fw  = frame_w.get(str(r['枠']), 1)
    aw  = age_w
    bt  = besttime_w
    bonus = bp if any(k in str(r.get('血統','')) for k in keys) else 0
    return raw * sw * gw * stw * fw * aw * bt + bonus

df_score['レース日'] = pd.to_datetime(df_score['レース日'])
df_score['score_raw'] = df_score.apply(calc_score, axis=1)
min_raw, max_raw = df_score['score_raw'].min(), df_score['score_raw'].max()
df_score['score_norm'] = (df_score['score_raw'] - min_raw) / (max_raw - min_raw) * 100

df_score['days_ago'] = df_score.groupby('馬名')['レース日'].transform(lambda d: (d.max() - d).dt.days)
df_score['decay_w']  = np.exp(-lambda_decay * df_score['days_ago'])
df_score['score_tw'] = df_score['score_norm'] * df_score['decay_w']

df_agg = df_score.groupby('馬名').agg(AvgZ=('score_tw','mean'), Stdev=('score_tw','std')).reset_index()
df_agg['RankZ']     = z_score(df_agg['AvgZ'])
df_agg['Stability'] = -df_agg['Stdev']

st.subheader("偏差値 vs 安定度 散布図")
avg_st = df_agg['Stability'].mean()
points = alt.Chart(df_agg).mark_circle(size=100).encode(x='RankZ:Q', y='Stability:Q', tooltip=['馬名','AvgZ','Stdev'])
labels = alt.Chart(df_agg).mark_text(dx=5, dy=-5, fontSize=10, color='white').encode(x='RankZ:Q', y='Stability:Q', text='馬名:N')
vline  = alt.Chart(pd.DataFrame({'x':[50]})).mark_rule(color='gray').encode(x='x:Q')
hline  = alt.Chart(pd.DataFrame({'y':[avg_st]})).mark_rule(color='gray').encode(y='y:Q')
st.altair_chart((points+labels+vline+hline).properties(width=600, height=400).interactive(), use_container_width=True)

st.sidebar.subheader("偏差値フィルター")
z_cut    = st.sidebar.slider("最低偏差値", float(df_agg['RankZ'].min()), float(df_agg['RankZ'].max()), 50.0)
filtered = df_agg[df_agg['RankZ'] >= z_cut].sort_values('RankZ', ascending=False)
st.subheader(f"馬名と偏差値一覧（偏差値>={z_cut:.1f}）")
st.table(filtered[['馬名','RankZ']].rename(columns={'RankZ':'偏差値'}))

top6 = df_agg.sort_values('RankZ', ascending=False).head(6)
top6['印'] = ['◎','〇','▲','☆','△','△']
st.subheader("上位6頭")
st.table(top6[['馬名','印']])

h1, h2        = top6.iloc[0]['馬名'], top6.iloc[1]['馬名']
others        = top6['馬名'].tolist()[1:]
others_marks  = top6['印'].tolist()[1:]
three         = ['馬連','ワイド','馬単']
main_share    = 0.5
pur1          = int(round((total_budget*main_share*1/4)/100)*100)
pur2          = int(round((total_budget*main_share*3/4)/100)*100)
rem           = total_budget - (pur1+pur2)
win_each      = int(np.floor((pur1/2)/100)*100)
place_each    = int(np.floor((pur2/2)/100)*100)

st.subheader("■ 資金配分")
st.write(f"合計予算：{total_budget:,}円 ｜ 単勝：{pur1:,}円 ｜ 複勝：{pur2:,}円 ｜ 残り：{rem:,}円")

bets = [
    {'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
    {'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
    {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each},
    {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each},
]

if scenario == '通常':
    choice = st.radio("購入券種", options=three, index=1)
    n = len(others)
    base = int(np.floor((rem/n)/100)*100)
    total_assigned = base * n
    leftover = rem - total_assigned
    for i, (nm, mk) in enumerate(zip(others, others_marks)):
        amt = base + (leftover if i==0 else 0)
        bets.append({'券種':choice,'印':f'◎–{mk}','馬':h1,'相手':nm,'金額':amt})

elif scenario == 'ちょい余裕':
    n_w = len(others)
    n_t = len(list(combinations(others,2)))
    total = n_w + n_t
    base = int(np.floor((rem/total)/100)*100)
    leftover = rem - base*total
    for i,(nm,mk) in enumerate(zip(others,others_marks)):
        amt = base + (leftover if i==0 else 0)
        bets.append({'券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':amt})
    for pair in combinations(others,2):
        bets.append({'券種':'三連複','印':'◎-〇▲☆△△','馬':h1,'相手':'／'.join(pair),'金額':base})

else:
    n_w  = len(others)
    n_t3 = len(list(combinations(others,2)))
    second_opts = others[:2]
    combo3 = [(s,t) for s in second_opts for t in others if t!=s]
    total = n_w + n_t3 + len(combo3)
    base = int(np.floor((rem/total)/100)*100)
    leftover = rem - base*total
    for i,(nm,mk) in enumerate(zip(others,others_marks)):
        amt = base + (leftover if i==0 else 0)
        bets.append({'券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':amt})
    for pair in combinations(others,2):
        bets.append({'券種':'三連複','印':'◎-〇▲☆△△','馬':h1,'相手':'／'.join(pair),'金額':base})
    for i,(s,t) in enumerate(combo3):
        amt = base + (leftover if i==0 else 0)
        bets.append({'券種':'三連単フォーメーション','印':'◎-〇▲-...','馬':h1,'相手':f"{s}／{t}",'金額':amt})

df_bets = pd.DataFrame(bets)
summary = df_bets.groupby('券種')['金額'].sum().reset_index()
summary['金額'] = summary['金額'].map(lambda x: f"{x:,}円")
st.write("### 券種別合計")
st.table(summary)

df_bets['金額'] = df_bets['金額'].map(lambda x: f"{x:,}円" if x>0 else "")
st.subheader("■ 最終買い目一覧")
st.table(df_bets[['券種','印','馬','相手','金額']])
