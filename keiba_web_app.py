import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

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
lambda_part    = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight    = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)
hist_weight    = 1 - orig_weight

gender_w       = {g: st.sidebar.slider(g, 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
style_w        = {s: st.sidebar.slider(s, 0.0, 2.0, 1.0) for s in ['逃げ','先行','差し','追込']}
season_w       = {s: st.sidebar.slider(s, 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
age_w          = st.sidebar.number_input("年齢重み", 0.0, 5.0, 1.0)
frame_w        = {str(i): st.sidebar.slider(f"{i}枠",0.0,2.0,1.0) for i in range(1,9)}
besttime_w     = st.sidebar.slider("ベストタイム重み",0.0,2.0,1.0)
weight_coeff   = st.sidebar.slider("斤量効果強度",0.0,2.0,1.0)
total_budget   = st.sidebar.slider("合計予算",500,50000,10000,500)
scenario       = st.sidebar.selectbox("シナリオ",['通常','ちょい余裕','余裕'])

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
attrs    = sheet2.iloc[:, [0,2,5,3,4]].copy()
attrs.columns = ['枠','馬名','脚質','性別','年齢']

# --- 馬一覧編集 ---
st.subheader("馬一覧と脚質入力")
edited = st.data_editor(
    attrs,
    column_config={'脚質': st.column_config.SelectboxColumn('脚質', options=list(style_w.keys())),
                   '枠': st.column_config.NumberColumn('枠')},
    num_rows='static'
)
df_wt  = df_score[['馬名','斤量']].drop_duplicates().rename(columns={'斤量':'input_wt'})
horses = pd.merge(edited, df_wt, on='馬名', how='left')

# --- 血統HTMLパース ---
cont = html_file.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont)
blood=[]
for r in rows:
    c = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)
    if len(c)>=2:
        blood.append((re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip()))
blood_df = pd.DataFrame(blood, columns=['馬名','血統'])

df_score = df_score.merge(horses, on='馬名', how='inner').merge(blood_df, on='馬名', how='left')

# --- 血統キーワード入力 ---
st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

# --- スコア計算 ---
avg_wt    = horses['input_wt'].mean()
style_map = dict(zip(horses['馬名'], horses['脚質']))

def calc_score(r):
    GP = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
          '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬・未勝利':1}
    g = GP.get(r['クラス名'],1)
    raw = g*(r['頭数']+1-r['確定着順']) + lambda_part*g
    sw  = season_w[season_of(pd.to_datetime(r['レース日']).month)]
    gw  = gender_w.get(r['性別'],1)
    stw = style_w.get(style_map.get(r['馬名'],''),1)
    fw  = frame_w.get(str(r['枠']),1)
    aw  = age_w; bt = besttime_w
    wt  = r.get('input_wt', np.nan)
    wfac= ((wt/avg_wt)**weight_coeff) if not np.isnan(wt) and avg_wt>0 else 1
    bonus = bp if any(k in str(r.get('血統','')) for k in keys) else 0
    return raw*sw*gw*stw*fw*aw*bt*wfac + bonus

# スコア適用
df_score['score_raw']  = df_score.apply(calc_score, axis=1)
df_score['score_norm'] = ((df_score['score_raw'] - df_score['score_raw'].min()) / (df_score['score_raw'].max() - df_score['score_raw'].min()) * 100)

df_agg = df_score.groupby('馬名').agg(AvgZ=('score_norm','mean'),Stdev=('score_norm','std')).reset_index()
df_agg['Stability'] = -df_agg['Stdev']
df_agg['RankZ']     = z_score(df_agg['AvgZ'])

# --- 散布図 ---
st.subheader("偏差値 vs 安定度 散布図")
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(df_agg['RankZ'], df_agg['Stability'], s=50)
avg_st = df_agg['Stability'].mean()
# ラベル配置（四象限ごとにオフセット）
for x, y, name in zip(df_agg['RankZ'], df_agg['Stability'], df_agg['馬名']):
    if x >= 50 and y >= avg_st:
        dx, dy = 2, 2
    elif x < 50 and y >= avg_st:
        dx, dy = -8, 2
    elif x < 50 and y < avg_st:
        dx, dy = -8, -6
    else:
        dx, dy = 2, -6
    ax.text(x+dx, y+dy, name, fontsize=8)
# グリッドと中心線
ax.set_facecolor('#fafafa')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.axvline(50, color='gray', linewidth=1)
ax.axhline(avg_st, color='gray', linewidth=1)
# 四象限ラベル
ax.text(60, avg_st+0.1, '一発警戒')
ax.text(40, avg_st+0.1, '警戒必須')
ax.text(60, avg_st-0.1, '鉄板級')
ax.text(40, avg_st-0.1, '堅実型')
plt.tight_layout()
st.pyplot(fig)

# --- 上位6頭印付け & 買い目生成省略 ---
