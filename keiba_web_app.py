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
    return 50 + 10 * (s - s.mean()) / s.std(ddof=1)

def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

st.sidebar.header("パラメータ設定")
# メイン設定
lambda_part  = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight  = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)
weight_coeff = st.sidebar.slider("斤量効果強度", 0.0, 2.0, 1.0, 0.05)
besttime_w   = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0, 0.05)
lambda_decay = st.sidebar.slider("時系列減衰 λ", 0.0, 1.0, 0.1, 0.01)
# 詳細設定
with st.sidebar.expander("性別重み", expanded=True):
    gender_w = {g: st.slider(g, 0.0, 2.0, 1.0, 0.05) for g in ['牡','牝','セ']}
with st.sidebar.expander("脚質重み", expanded=True):
    style_w  = {s: st.slider(s, 0.0, 2.0, 1.0, 0.05) for s in ['逃げ','先行','差し','追込']}
with st.sidebar.expander("四季重み", expanded=True):
    season_w = {s: st.slider(s, 0.0, 2.0, 1.0, 0.05) for s in ['春','夏','秋','冬']}
age_w = st.sidebar.number_input("年齢重み", 0.0, 5.0, 1.0, 0.1)
with st.sidebar.expander("枠順重み", expanded=True):
    frame_w  = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0, 0.05) for i in range(1,9)}
# シナリオ
scenario     = st.sidebar.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])
# 予算は最後
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)

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
attrs = sheet2.drop_duplicates(subset=sheet2.columns[2], keep='first').iloc[:, :5]
attrs.columns=['枠','番','馬名','性別','年齢']; attrs['脚質']=''
edited = st.data_editor(
    attrs, column_order=['枠','番','馬名','性別','年齢','脚質'],
    column_config={'脚質': st.column_config.SelectboxColumn('脚質',options=['逃げ','先行','差し','追込'])},
    use_container_width=True, num_rows='static')
horses = edited[['枠','番','馬名','性別','年齢','脚質']]
cont = html_file.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont); blood=[]
for r in rows:
    c = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)
    if len(c)>=2: blood.append((re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip()))
blood_df = pd.DataFrame(blood, columns=['馬名','血統'])
df_score = df_score.merge(horses, on='馬名', how='inner').merge(blood_df, on='馬名', how='left')
# --- スコア計算 ---
st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)
GP={'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
def calc_score(r):
    g = GP.get(r['クラス名'],1)
    raw = g*(r['頭数']+1-r['確定着順']) + lambda_part*g
    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r['性別'],1); stw = style_w.get(r['脚質'],1)
    fw  = frame_w.get(str(r['枠']),1); aw=age_w; bt=besttime_w; wf=weight_coeff
    bonus = bp if any(k in str(r.get('血統','')) for k in keys) else 0
    return raw*sw*gw*stw*fw*aw*bt*wf + bonus
df_score['レース日'] = pd.to_datetime(df_score['レース日'])
df_score['score_raw'] = df_score.apply(calc_score, axis=1)
min_r, max_r = df_score['score_raw'].min(), df_score['score_raw'].max()
df_score['score_norm'] = (df_score['score_raw']-min_r)/(max_r-min_r)*100
# 時系列加重
df_score['days_ago'] = df_score.groupby('馬名')['レース日'] \
    .transform(lambda d: (d.max() - d).dt.days)
df_score['decay_w']  = np.exp(-lambda_decay * df_score['days_ago'])
df_score['score_tw'] = df_score['score_norm'] * df_score['decay_w']

# --- 集計 ---
df_agg = df_score.groupby('馬名').agg(AvgZ=('score_tw','mean'), Stdev=('score_tw','std')).reset_index()
df_agg['RankZ']     = z_score(df_agg['AvgZ']); df_agg['Stability'] = -df_agg['Stdev']
# --- 散布図 ---
st.subheader("偏差値 vs 安定度 散布図")
df_plot = df_agg.copy()
avg_rankz = df_plot['RankZ'].mean()
avg_stab = df_plot['Stability'].mean()
chart = alt.Chart(df_plot).mark_circle(size=150).encode(
    x=alt.X('RankZ:Q', title='偏差値'),
    y=alt.Y('Stability:Q', title='安定度'),
    color=alt.condition(
        (alt.datum.RankZ >= avg_rankz) & (alt.datum.Stability >= avg_stab),
        alt.value('red'), alt.value('steelblue')
    ),
    tooltip=[alt.Tooltip('馬名:N'), alt.Tooltip('AvgZ:Q', title='平均偏差値'), alt.Tooltip('Stdev:Q', title='偏差値標準偏差')]
).properties(width=600, height=400)
# 平均線
avg_lines = alt.Chart(pd.DataFrame({
    'x':[avg_rankz], 'y':[avg_stab]
})).mark_rule(color='gray', strokeDash=[4,4]).encode(
    x='x:Q', y='y:Q'
)
st.altair_chart(chart + avg_lines, use_container_width=True)

# --- 買い目生成 ---
top6 = df_agg.sort_values('RankZ', ascending=False).head(6)
top6['印'] = ['◎','〇','▲','☆','△','△']
others      = top6['馬名'].tolist()[1:]
others_marks= top6['印'].tolist()[1:]
h1, h2      = top6.iloc[0]['馬名'], top6.iloc[1]['馬名']
# 資金配分
main_share  = 0.5
pur1        = int(round((total_budget*main_share*1/4)/100)*100)
pur2        = int(round((total_budget*main_share*3/4)/100)*100)
rem         = total_budget - (pur1+pur2)
win_each    = int(np.floor((pur1/2)/100)*100)
place_each  = int(np.floor((pur2/2)/100)*100)

bets = [
    {'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
    {'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
    {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each},
    {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each},
]
parts_map = {'通常':['馬連','ワイド','馬単'], 'ちょい余裕':['ワイド','三連複'], '余裕':['ワイド','三連複','三連単']}
if scenario == '通常':
    choice = st.radio("購入券種", options=parts_map['通常'], index=1)
    n = len(others); base=int(np.floor((rem/n)/100)*100); left=rem-base*n
    for i,(nm, mk) in enumerate(zip(others, others_marks)):
        amt = base + (left if i==0 else 0)
        bets.append({'券種':choice,'印':f'◎–{mk}','馬':h1,'相手':nm,'金額':amt})
elif scenario == 'ちょい余裕':
    parts=['ワイド','三連複']; n_w=len(others); n_t=len(list(combinations(others,2)))
    total=n_w+n_t; base=int(np.floor((rem/total)/100)*100); left=rem-base*total
    for i,(nm, mk) in enumerate(zip(others, others_marks)):
        bets.append({'券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':base+(left if i==0 else 0)})
    for pair in combinations(others,2): bets.append({'券種':'三連複','印':'◎-〇▲☆△△','馬':h1,'相手':'/'.join(pair),'金額':base})
else:
    n_w=len(others); n_t3=len(list(combinations(others,2)))
    second=others[:2]; combo3=[(s,t) for s in second for t in others if t!=s]
    total=n_w+n_t3+len(combo3); base=int(np.floor((rem/total)/100)*100); left=rem-base*total
    for i,(nm, mk) in enumerate(zip(others, others_marks)):
        bets.append({'券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':base+(left if i==0 else 0)})
    for pair in combinations(others,2): bets.append({'券種':'三連複','印':'◎-〇▲☆△△','馬':h1,'相手':'/'.join(pair),'金額':base})
    for i,(s,t) in enumerate(combo3): bets.append({'券種':'三連単','印':'◎-〇▲-...','馬':h1,'相手':f"{s}/{t}",'金額':base+(left if i==0 else 0)})
# --- 表示 ---
df_bets=pd.DataFrame(bets)
df_bets['金額']=df_bets['金額'].map(lambda x:f"{x:,}円")
st.subheader("■ 最終買い目一覧")
st.table(df_bets[['券種','印','馬','相手','金額']])
