import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations, permutations

# ── 日本語フォント設定 ──
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(fname='ipaexg.ttf').get_name()

st.set_page_config(layout="wide")
st.title('競馬予想アプリ（完成版）')

# ── サイドバー：重み設定 ──
with st.sidebar:
    st.header("重み設定")
    # 性別補正
    with st.expander("性別の重み", expanded=False):
        male_w    = st.number_input('牡の重み',   0.0, 2.0, 1.1, 0.01)
        female_w  = st.number_input('牝の重み',   0.0, 2.0, 1.0, 0.01)
        gelding_w = st.number_input('セの重み',   0.0, 2.0, 0.95,0.01)
    # 脚質補正
    with st.expander("脚質の重み", expanded=False):
        nige_w    = st.number_input('逃げの重み', 0.0, 2.0, 1.2, 0.01)
        senko_w   = st.number_input('先行の重み', 0.0, 2.0, 1.1, 0.01)
        sashi_w   = st.number_input('差しの重み', 0.0, 2.0, 1.0, 0.01)
        ooka_w    = st.number_input('追込の重み', 0.0, 2.0, 0.9, 0.01)
    # 四季補正
    with st.expander("四季の重み", expanded=False):
        spring_w  = st.number_input('春の重み',   0.0, 2.0, 1.0, 0.01)
        summer_w  = st.number_input('夏の重み',   0.0, 2.0, 1.1, 0.01)
        autumn_w  = st.number_input('秋の重み',   0.0, 2.0, 1.0, 0.01)
        winter_w  = st.number_input('冬の重み',   0.0, 2.0, 0.95,0.01)
    # その他指標
    w_jin     = st.number_input('斤量重み',      0.0, 1.0, 1.0, 0.1)
    w_best    = st.number_input('距離ベスト重み',0.0, 1.0, 1.0, 0.1)
    # 枠順補正
    with st.expander("枠順の重み (1〜8)", expanded=False):
        gate_weights = {}
        for gate in range(1,9):
            gate_weights[gate] = st.number_input(f'{gate}枠の重み', 0.0, 2.0, 1.0, 0.01)
    # 最終スコア重み
    with st.expander("最終スコア重み", expanded=True):
        weight_z    = st.slider('偏差値 の重み',         0.0, 1.0, 0.7, 0.05)
        weight_rb   = st.slider('RawBase偏差値 の重み',  0.0, 1.0, 0.2, 0.05)
        weight_gate = st.slider('枠偏差値 の重み',      0.0, 1.0, 0.1, 0.05)

# ── キャッシュクリア注意 ──
st.write("**注意：重み変更後は右上メニュー→Clear cacheで再実行してください。**")

# ── データアップロード ──
st.subheader("データアップロード")
c1, c2 = st.columns(2)
with c1:
    upload_xlsx = st.file_uploader('成績＆馬情報 (XLSX)', type='xlsx')
with c2:
    upload_html = st.file_uploader('血統表 (HTML)', type='html')
if not upload_xlsx:
    st.stop()

# ── データ読み込み ──
xls = pd.ExcelFile(upload_xlsx)
df  = xls.parse(0, parse_dates=['レース日'])
df['馬名'] = df['馬名'].astype(str).str.strip()

# ── 馬情報シート読み込み＆整形 ──
stats = xls.parse(1, header=1)
keys = ['馬名','性別','年齢','ベストタイム']
col_map = {}
for k in keys:
    for c in stats.columns:
        if k in str(c): col_map[c] = k; break
stats = stats.rename(columns=col_map)
for k in keys:
    if k not in stats.columns: stats[k] = np.nan
stats = stats[keys].drop_duplicates('馬名')
stats['馬名'] = stats['馬名'].astype(str).str.strip()
# 馬情報列に『枠-番-馬名』が入っている場合の分割
# 馬情報列に『枠-番-馬名』が入っている場合の安全分割
def split_frame(x):
    parts = str(x).split('-', 2)
    if len(parts)==3 and parts[0].isdigit() and parts[1].isdigit():
        return pd.Series({'枠':int(parts[0]), '番':int(parts[1]), '馬名':parts[2].strip()})
    else:
        return pd.Series({'枠':np.nan, '番':np.nan, '馬名':str(x).strip()})
stats = stats.join(stats['馬名'].apply(split_frame))
# NaN枠は1に置換
df = df.merge(
    stats[['枠','番','馬名','性別','年齢','best_dist_time']].fillna({'枠':1}),
    on='馬名', how='left'
)
(
    stats[['枠','番','馬名','性別','年齢','best_dist_time']],
    on='馬名', how='left'
)

# ── 血統表読み込み（未使用） ──
ped = None
if upload_html:
    try: ped = pd.read_html(upload_html.read())[0].set_index('馬名')
    except: ped = None

# ── 脚質・斤量設定 ──
st.subheader("脚質・本斤量設定")
equines = df['馬名'].unique()
inp = pd.DataFrame({'馬名':equines,'脚質':['差し']*len(equines),'本斤量':[56]*len(equines)})
edited = st.data_editor(inp, column_config={
    '脚質': st.column_config.SelectboxColumn('脚質',['逃げ','先行','差し','追込']),
    '本斤量': st.column_config.NumberColumn('本斤量',45,60,1)
}, use_container_width=True)
edited['馬名'] = edited['馬名'].astype(str).str.strip()
df = df.merge(edited.rename(columns={'本斤量':'today_weight'}), on='馬名', how='left')

# ── ファクター関数 ──
def ped_factor(r): return 1.0

def style_f(s): return {'逃げ':nige_w,'先行':senko_w,'差し':sashi_w,'追込':ooka_w}.get(s,1.0)
def age_f(a): return 1+0.2*(1-abs(a-5)/5)
def sex_f(sx): return {'牡':male_w,'牝':female_w,'セ':gelding_w}.get(sx,1.0)
def sea_f(dt): m=dt.month; return spring_w if m in [3,4,5] else summer_w if m in [6,7,8] else autumn_w if m in [9,10,11] else winter_w

def gate_factor(r):
    return gate_weights.get(r['枠'], 1.0)

# ── 生スコア計算（非線形グレード強調²乗） ──
base_grade = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,'3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
grade_score = {k:v**2 for k,v in base_grade.items()}

df['RawBase'] = df.apply(lambda r: grade_score.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['Raw'] = (
    df['RawBase']
    * df['脚質'].map(style_f)
    * df['年齢'].map(age_f)
    * df['性別'].map(sex_f)
    * df['レース日'].map(sea_f)
    * df.apply(gate_factor, axis=1)
)

# ── 指標の正規化＆Z化 ──
tmin,tmax = df['best_dist_time'].min(), df['best_dist_time'].max()
df['dist_n'] = (tmax - df['best_dist_time'])/(tmax - tmin)
df['up3_n']  = df['Ave-3F']/df['上がり3Fタイム']
df['odds_n'] = 1/(1 + np.log10(df['単勝オッズ']))
jmax,jmin  = df['斤量'].max(), df['斤量'].min()
df['jin_n']   = (jmax - df['斤量'])/(jmax - jmin)
df['today_n'] = (jmax - df['today_weight'])/(jmax - jmin)
wabs       = df['増減'].abs().mean()
df['wd_n']   = 1 - df['増減'].abs()/wabs
df['raw_n']  = (df['Raw'] - 1)/(10 * df['頭数'] - 1)

metrics = ['raw_n','up3_n','odds_n','jin_n','today_n','dist_n','wd_n']
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = ((df[m] - mu) / sd).fillna(0) if sd else 0
# 枠補正もZ化
mu_g, sd_g = df['枠'].map(lambda x: gate_weights.get(x,1.0)).mean(), df['枠'].map(lambda x: gate_weights.get(x,1.0)).std(ddof=1)
df['Z_gate'] = ((df['枠'].map(lambda x: gate_weights.get(x,1.0)) - mu_g) / sd_g).fillna(0) if sd_g else 0

# ── 合成スコア total_z ──
comp_w = {
    'Z_raw_n':8, 'Z_up3_n':2, 'Z_odds_n':1,
    'Z_jin_n':w_jin, 'Z_today_n':w_jin,
    'Z_dist_n':w_best, 'Z_wd_n':1,
    'Z_gate': weight_gate
}
totw = sum(comp_w.values())
df['total_z'] = sum(df[k]*w for k,w in comp_w.items())/totw

# ── 馬別集計＆偏差値化＋RawBase偏差値 ──
summary = df.groupby('馬名').agg(
    mean_z       = ('total_z','mean'),
    std_z        = ('total_z','std'),
    RawBase_mean = ('RawBase','mean')
).reset_index()
summary['std_z'] = summary['std_z'].fillna(0)

mz, MZ = summary['mean_z'].min(), summary['mean_z'].max()
summary['偏差値'] = 30 + (summary['mean_z'] - mz) / (MZ - mz) * 40 if MZ > mz else 50

rb_min, rb_max = summary['RawBase_mean'].min(), summary['RawBase_mean'].max()
summary['RawBase偏差値'] = 30 + (summary['RawBase_mean'] - rb_min) / (rb_max - rb_min) * 40 if rb_max > rb_min else 50

# ── 最終バランス ──
summary['バランス'] = (
    weight_z  * summary['偏差値'] +
    weight_rb * summary['RawBase偏差値'] +
    weight_gate * summary['枠'] -
    summary['std_z']
)

# ── 結果表示 ──
st.subheader("本日の予想6頭")
top6 = summary.nlargest(6, 'バランス').reset_index(drop=True)
tag_map = {1:'◎',2:'〇',3:'▲',4:'△',5:'△',6:'△'}
top6['印'] = top6.index.to_series().map(lambda i: tag_map[i+1])
st.table(top6[['印','馬名','偏差値','RawBase偏差値','バランス']])

