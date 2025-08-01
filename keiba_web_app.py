import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import altair as alt

# 日本語フォント設定（matplotlib用）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic','Meiryo','MS Gothic']

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
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)
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

# --- 出走馬一覧作成 （独自表） ---
attrs = sheet2.iloc[:, [0,1,2,3,4]].copy()
attrs.columns = ['枠','番','馬名','性別','年齢']
attrs['脚質'] = ''  # プルダウン入力用

# --- 馬一覧編集 ---
st.subheader("馬一覧と脚質入力")
edited = st.data_editor(
    attrs,
    column_order=['枠','番','馬名','性別','年齢','脚質'],
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質',
                                                 options=['逃げ','先行','差し','追込'])
    },
    use_container_width=True,
    num_rows='static'
)
horses = edited.copy()[['枠','番','馬名','性別','年齢','脚質']]

# --- 血統HTMLパース ---
cont = html_file.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?</tr>', cont)
blood=[]
for r in rows:
    c = re.findall(r'<t[dh][^>]*>([\s\S]*?)</[tdh]>', r)
    if len(c)>=2:
        blood.append((
            re.sub(r'<.*?>','',c[0]).strip(),
            re.sub(r'<.*?>','',c[1]).strip()
        ))
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
def calc_score(r):
    GP = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,
          'オープン特別':4,'3勝クラス':3,'2勝クラス':2,
          '1勝クラス':1,'新馬・未勝利':1}
    g   = GP.get(r['クラス名'],1)
    raw = g*(r['頭数'] + 1 - r['確定着順']) + lambda_part*g
    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r['性別'],1)
    stw = style_w.get(r['脚質'],1)
    fw  = frame_w.get(str(r['枠']),1)
    aw  = age_w
    bt  = besttime_w
    bonus = bp if any(k in str(r.get('血統','')) for k in keys) else 0
    return raw * sw * gw * stw * fw * aw * bt + bonus

df_score['score_raw']  = df_score.apply(calc_score, axis=1)
df_score['score_norm'] = (
    (df_score['score_raw'] - df_score['score_raw'].min()) /
    (df_score['score_raw'].max() - df_score['score_raw'].min())
) * 100

# --- 馬ごとの統計 ---
df_agg = df_score.groupby('馬名')['score_norm'] \
                .agg(['mean','std']) \
                .reset_index()
df_agg.columns      = ['馬名','AvgZ','Stdev']
df_agg['Stability'] = -df_agg['Stdev']
df_agg['RankZ']     = z_score(df_agg['AvgZ'])

# --- 散布図 ---
st.subheader("偏差値 vs 安定度 散布図")
avg_st = df_agg['Stability'].mean()
quad = pd.DataFrame([
    {'RankZ':75, 'Stability':avg_st+(df_agg['Stability'].max()-avg_st)/2, 'label':'一発警戒'},
    {'RankZ':25, 'Stability':avg_st+(df_agg['Stability'].max()-avg_st)/2, 'label':'警戒必須'},
    {'RankZ':75, 'Stability':avg_st-(avg_st-df_agg['Stability'].min())/2, 'label':'鉄板級'},
    {'RankZ':25, 'Stability':avg_st-(avg_st-df_agg['Stability'].min())/2, 'label':'堅実型'}
])
points     = alt.Chart(df_agg).mark_circle(size=100) \
                .encode(x='RankZ:Q', y='Stability:Q',
                        tooltip=['馬名','AvgZ','Stdev'])
labels     = alt.Chart(df_agg).mark_text(dx=5,dy=-5,fontSize=10,color='white') \
                .encode(x='RankZ:Q', y='Stability:Q', text='馬名:N')
quad_label = alt.Chart(quad).mark_text(fontSize=14,fontWeight='bold',color='white') \
                .encode(x='RankZ:Q', y='Stability:Q', text='label:N')
vline      = alt.Chart(pd.DataFrame({'x':[50]})).mark_rule(color='gray').encode(x='x:Q')
hline      = alt.Chart(pd.DataFrame({'y':[avg_st]})).mark_rule(color='gray').encode(y='y:Q')
st.altair_chart(
    (points + labels + quad_label + vline + hline)
     .properties(width=650, height=450)
     .interactive(),
    use_container_width=True
)

# --- 偏差値フィルター ---
st.sidebar.subheader("偏差値フィルター")
z_cut = st.sidebar.slider(
    "最低偏差値",
    float(df_agg['RankZ'].min()),
    float(df_agg['RankZ'].max()),
    50.0
)
st.subheader(f"馬名と偏差値一覧 (偏差値>={z_cut:.1f})")
filtered = df_agg[df_agg['RankZ'] >= z_cut] \
                .sort_values('RankZ', ascending=False)
st.table(filtered[['馬名','RankZ']].rename(columns={'RankZ':'偏差値'}))

# --- 上位6頭 ---
top6 = df_agg.sort_values('RankZ',ascending=False).head(6)
top6['印']=['◎','〇','▲','☆','△','△']
st.subheader("上位6頭")
st.table(top6[['馬名','印']])

# --- 資金配分 & 買い目一覧 ---
tansho  = top6.iloc[0]['馬名']
fukusho = top6.iloc[1]['馬名']
# 単複に50%、1:3で分配
pur1 = round((total_budget*0.5*0.25)/100)*100
pur2 = round((total_budget*0.5*0.75)/100)*100
rem   = total_budget - (pur1 + pur2)
others5    = list(top6.iloc[1:6]['馬名'])
umaren     = [f"{tansho}-{h}" for h in others5]
sanrenpuku = umaren.copy()
sanrentan  = umaren.copy()

bets = []
bets.append({"券種":"単勝","組み合わせ":tansho,"金額":pur1})
bets.append({"券種":"複勝","組み合わせ":fukusho,"金額":pur2})
for c in umaren:
    bets.append({"券種":"馬連","組み合わせ":c,         "金額":rem})
    bets.append({"券種":"ワイド","組み合わせ":c,       "金額":rem})
    bets.append({"券種":"馬単","組み合わせ":f"{tansho}>{c.split('-')[1]}", "金額":rem})
for c in sanrenpuku:
    bets.append({"券種":"三連複","組み合わせ":c,        "金額":rem})
for c in sanrentan:
    bets.append({"券種":"三連単マルチ","組み合わせ":c, "金額":rem})

df_bets = pd.DataFrame(bets, columns=["券種","組み合わせ","金額"])
st.subheader("推奨買い目一覧と配分（円）")
st.table(df_bets)
