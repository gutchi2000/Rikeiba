import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations

# =========== 日本語フォントの用意 ==========
try:
    jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
except:
    jp_font = font_manager.FontProperties(fname="C:/Windows/Fonts/meiryo.ttc")  # Windowsの場合

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']

def z_score(s: pd.Series) -> pd.Series:
    if s.std(ddof=0) == 0 or pd.isna(s.std(ddof=0)):
        return pd.Series([50]*len(s), index=s.index)
    return 50 + 10 * (s - s.mean()) / s.std(ddof=0)

def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

# ========== サイドバー ==========
st.sidebar.header("パラメータ設定")
lambda_part  = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight  = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)  # いまは未使用
hist_weight  = 1 - orig_weight

with st.sidebar.expander("性別重み", expanded=False):
    gender_w = {g: st.slider(g, 0.0, 2.0, 1.0) for g in ['牡','牝','セ']}
with st.sidebar.expander("脚質重み", expanded=False):
    style_w  = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['逃げ','先行','差し','追込']}
with st.sidebar.expander("四季重み", expanded=False):
    season_w = {s: st.slider(s, 0.0, 2.0, 1.0) for s in ['春','夏','秋','冬']}
with st.sidebar.expander("年齢重み", expanded=False):
    age_w = {str(age): st.slider(f"{age}歳", 0.0, 2.0, 1.0, 0.05) for age in range(3, 11)}
with st.sidebar.expander("枠順重み", expanded=False):
    frame_w = {str(i): st.slider(f"{i}枠", 0.0, 2.0, 1.0) for i in range(1,9)}
besttime_w   = st.sidebar.slider("ベストタイム重み", 0.0, 2.0, 1.0)
with st.sidebar.expander("戦績率の重み（すべて芝を使用）", expanded=False):
    win_w  = st.slider("勝率(芝)の重み",   0.0, 5.0, 1.0, 0.1)
    quin_w = st.slider("連対率(芝)の重み", 0.0, 5.0, 0.7, 0.1)
    plc_w  = st.slider("複勝率(芝)の重み", 0.0, 5.0, 0.5, 0.1)
with st.sidebar.expander("各種ボーナス設定", expanded=False):
    grade_bonus = st.slider("重賞実績ボーナス（GⅠ・GⅡ・GⅢ）", 0, 20, 5)
    agari1_bonus = st.slider("上がり3F 1位ボーナス", 0, 10, 3)
    agari2_bonus = st.slider("上がり3F 2位ボーナス", 0, 5, 2)
    agari3_bonus = st.slider("上がり3F 3位ボーナス", 0, 3, 1)
    body_weight_bonus = st.slider("適正馬体重ボーナス", 0, 10, 3)
weight_coeff = st.sidebar.slider("斤量効果強度", 0.0, 2.0, 1.0)  # いまは未使用
total_budget = st.sidebar.slider("合計予算", 500, 50000, 10000, 100)
scenario     = st.sidebar.selectbox("シナリオ", ['通常','ちょい余裕','余裕'])

# ========== メイン ==========
st.title("競馬予想アプリ（完成版）")
st.subheader("ファイルアップロード")
excel_file = st.file_uploader("Excel (成績＆属性)", type='xlsx')
html_file  = st.file_uploader("HTML (血統)", type='html')
if not excel_file or not html_file:
    st.info("ExcelとHTMLを両方アップロードしてください。")
    st.stop()

# ---- データ読み込み ----
df_score = pd.read_excel(excel_file, sheet_name=0)
sheet2   = pd.read_excel(excel_file, sheet_name=1)
sheet2 = sheet2.drop_duplicates(subset=sheet2.columns[2], keep='first').reset_index(drop=True)
attrs = sheet2.iloc[:, [0,1,2,3,4]].copy()
attrs.columns = ['枠','番','馬名','性別','年齢']
attrs['脚質'] = ''
attrs['斤量'] = np.nan

# ===== [M1] 戦績率(芝)＆ベストタイム抽出 START =====
# 列名の正規化＆自動検出 → ダメなら手動選択にフォールバック
def norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'\s+', '', s)
    s = s.replace('（', '(').replace('）', ')').replace('％', '%')
    s = s.replace('ﾍﾞｽﾄ', 'ベスト').replace('ﾀｲﾑ', 'タイム')
    return s

col_map = {orig: norm_col(orig) for orig in sheet2.columns}

def find_col(patterns):
    for orig, normed in col_map.items():
        for pat in patterns:
            if re.search(pat, normed, flags=re.I):
                return orig
    return None

# 馬名列：見つからなければ3列目をフォールバック
name_col = find_col([r'馬名|名前|出走馬']) or sheet2.columns[2]

col_win  = find_col([r'勝率.*芝', r'芝.*勝率', r'^勝率(\(芝\))?$'])
col_quin = find_col([r'連対率.*芝', r'芝.*連対率', r'^連対率(\(芝\))?$'])
col_plc  = find_col([r'複勝率.*芝', r'芝.*複勝率', r'^複勝率(\(芝\))?$'])
col_bt   = find_col([r'ベスト.*タイム', r'Best.*Time', r'ﾍﾞｽﾄ.*ﾀｲﾑ', r'タイム.*(最速|ベスト)'])

if any(c is None for c in [col_win, col_quin, col_plc, col_bt]):
    st.warning("2枚目シートの列自動検出に失敗。手動で選んでください。")
    options = list(sheet2.columns)
    if col_win  is None:  col_win  = st.selectbox("勝率(芝)の列", options, key="wincol")
    if col_quin is None:  col_quin = st.selectbox("連対率(芝)の列", options, key="quincol")
    if col_plc  is None:  col_plc  = st.selectbox("複勝率(芝)の列", options, key="plccol")
    if col_bt   is None:  col_bt   = st.selectbox("ベストタイムの列", options, key="btcol")

rate = sheet2[[name_col, col_win, col_quin, col_plc, col_bt]].copy()
rate.columns = ['馬名','勝率_芝','連対率_芝','複勝率_芝','ベストタイム']

for c in ['勝率_芝','連対率_芝','複勝率_芝']:
    rate[c] = (
        rate[c].astype(str)
               .str.replace('%','', regex=False)
               .str.replace('％','', regex=False)
    )
    rate[c] = pd.to_numeric(rate[c], errors='coerce')

max_val = pd.concat([rate['勝率_芝'], rate['連対率_芝'], rate['複勝率_芝']], axis=1).max().max()
if pd.notna(max_val) and max_val <= 1.0:
    for c in ['勝率_芝','連対率_芝','複勝率_芝']:
        rate[c] = rate[c] * 100.0

def parse_time_to_sec(x):
    s = str(x).strip()
    m = re.match(r'^(\d+):(\d+)\.(\d+)$', s)
    if m:
        return int(m.group(1))*60 + int(m.group(2)) + float('0.'+m.group(3))
    m = re.match(r'^(\d+)[\.\:](\d+)[\.\:](\d+)$', s)
    if m:
        return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/10
    return pd.to_numeric(s, errors='coerce')

rate['ベストタイム秒'] = rate['ベストタイム'].apply(parse_time_to_sec)
# ===== [M1] 戦績率(芝)＆ベストタイム抽出 END =====

# --- 馬一覧＋脚質＋馬体重入力 ---
st.subheader("馬一覧・脚質・当日馬体重入力")
if '馬体重' not in attrs.columns:
    attrs['馬体重'] = np.nan

edited = st.data_editor(
    attrs,
    column_order=['枠','番','馬名','性別','年齢','脚質','馬体重'],
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込']),
        '馬体重': st.column_config.NumberColumn('馬体重', min_value=300, max_value=600, step=1)
    },
    use_container_width=True,
    num_rows='static'
)
horses = edited.copy()[['枠','番','馬名','性別','年齢','脚質','馬体重']]

# ---- 血統パース ----
cont = html_file.read().decode(errors='ignore')
rows = re.findall(r'<tr[\s\S]*?<\/tr>', cont)
blood = []
for r in rows:
    c = re.findall(r'<t[dh][^>]*>([\s\S]*?)<\/[tdh]>', r)
    if len(c) >= 2:
        blood.append((re.sub(r'<.*?>','',c[0]).strip(), re.sub(r'<.*?>','',c[1]).strip()))
blood_df = pd.DataFrame(blood, columns=['馬名','血統'])

# ===== [M2] df_score マージ（horses / blood / rate）START =====
df_score = (
    df_score
    .merge(horses, on='馬名', how='inner')
    .merge(blood_df, on='馬名', how='left')
    .merge(rate[['馬名','勝率_芝','連対率_芝','複勝率_芝','ベストタイム秒']], on='馬名', how='left')
)
# ===== [M2] df_score マージ（horses / blood / rate）END =====

# ===== [M3] ベストタイム正規化レンジ START =====
bt_min = df_score['ベストタイム秒'].min(skipna=True)
bt_max = df_score['ベストタイム秒'].max(skipna=True)
bt_span = (bt_max - bt_min) if pd.notna(bt_min) and pd.notna(bt_max) and (bt_max > bt_min) else 1.0
# ===== [M3] ベストタイム正規化レンジ END =====

st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

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
    aw  = age_w.get(str(r['年齢']), 1.0)
    weight_factor = 1

    # 血統ボーナス
    bloodline = str(r.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    blood_bonus = 0
    for k in keys:
        if k.strip() and k.strip().lower() in bloodline:
            blood_bonus = bp
            break

    # 重賞実績ボーナス（GⅠ・GⅡ・GⅢ のみ）
    grade_name = str(r.get('クラス名',''))
    grade_point = grade_bonus if grade_name in ['GⅠ','GⅡ','GⅢ'] else 0

    # 上がり3F順位ボーナス
    agari_bonus = 0
    agari_order = r.get('上3F順位', np.nan)
    try:
        agari_order = int(agari_order)
        if   agari_order == 1: agari_bonus = agari1_bonus
        elif agari_order == 2: agari_bonus = agari2_bonus
        elif agari_order == 3: agari_bonus = agari3_bonus
    except:
        pass

    # 馬体重適正ボーナス
    body_bonus = 0
    try:
        name = r['馬名']
        myhist = df_score[(df_score['馬名']==name) & (df_score['馬体重'].notna())]
        if len(myhist) > 0:
            best_row = myhist.loc[myhist['確定着順'].idxmin()]
            tekitai = best_row['馬体重']
            now_bw = r.get('馬体重', np.nan)
            if not pd.isna(now_bw) and abs(now_bw - tekitai) <= 10:
                body_bonus = body_weight_bonus
    except:
        pass

    # ===== [M4a] 芝の勝率/連対率/複勝率ボーナス START =====
    rate_bonus = 0.0
    try:
        if pd.notna(r.get('勝率_芝', np.nan)):   rate_bonus += win_w  * (float(r['勝率_芝'])  / 100.0)
        if pd.notna(r.get('連対率_芝', np.nan)): rate_bonus += quin_w * (float(r['連対率_芝']) / 100.0)
        if pd.notna(r.get('複勝率_芝', np.nan)): rate_bonus += plc_w  * (float(r['複勝率_芝'])  / 100.0)
    except:
        pass
    # ===== [M4a] 芝の勝率/連対率/複勝率ボーナス END =====

    # ===== [M4b] ベストタイム加点（速いほど+） START =====
    bt_bonus = 0.0
    try:
        if pd.notna(r.get('ベストタイム秒', np.nan)):
            bt_norm = (bt_max - float(r['ベストタイム秒'])) / bt_span  # 速いほど1に近い
            bt_norm = max(0.0, min(1.0, bt_norm))
            bt_bonus = besttime_w * bt_norm
    except:
        pass
    # ===== [M4b] ベストタイム加点（速いほど+） END =====

    total_bonus = blood_bonus + grade_point + agari_bonus + body_bonus + rate_bonus + bt_bonus
    return raw * sw * gw * stw * fw * aw * weight_factor + total_bonus

df_score['score_raw']  = df_score.apply(calc_score, axis=1)
if df_score['score_raw'].max() == df_score['score_raw'].min():
    df_score['score_norm'] = 50.0
else:
    df_score['score_norm'] = (
        (df_score['score_raw'] - df_score['score_raw'].min()) /
        (df_score['score_raw'].max() - df_score['score_raw'].min()) * 100
    )

df_agg = (
    df_score.groupby('馬名')['score_norm']
    .agg(['mean','std']).reset_index()
)
df_agg.columns     = ['馬名','AvgZ','Stdev']
df_agg['Stability'] = -df_agg['Stdev']
df_agg['RankZ']     = z_score(df_agg['AvgZ'])

def reason(row):
    base = f"平均スコア{row['AvgZ']:.1f}、安定度{row['Stdev']:.1f}。"
    if row['RankZ'] >= 65:
        base += "非常に高評価。"
    elif row['RankZ'] >= 55:
        base += "高水準。"
    if row['Stdev'] < 8:
        base += "安定感抜群。"
    elif row['Stdev'] < 13:
        base += "比較的安定。"
    if row['馬名'] in df_score[df_score['score_raw'] > df_score['score_raw'].mean() + 10]['馬名'].values:
        base += "血統・脚質等もプラス評価。"
    return base

df_agg['根拠'] = df_agg.apply(reason, axis=1)

# ========== 全頭散布図（偏差値 vs 安定度） ==========
import altair as alt

avg_st = df_agg['Stdev'].mean()
quad_labels = pd.DataFrame([
    {'RankZ':70, 'Stdev': avg_st - (avg_st-df_agg['Stdev'].min())/1.5, 'label':'鉄板・本命'},
    {'RankZ':70, 'Stdev': avg_st + (df_agg['Stdev'].max()-avg_st)/2, 'label':'波乱・ムラ馬'},
    {'RankZ':30, 'Stdev': avg_st - (avg_st-df_agg['Stdev'].min())/1.5, 'label':'堅実ヒモ'},
    {'RankZ':30, 'Stdev': avg_st + (df_agg['Stdev'].max()-avg_st)/2, 'label':'消し・大穴'},
])

points = alt.Chart(df_agg).mark_circle(size=100).encode(
    x=alt.X('RankZ:Q', title='偏差値'),
    y=alt.Y('Stdev:Q', title='標準偏差（小さいほど安定）'),
    tooltip=['馬名','AvgZ','Stdev']
)
labels = alt.Chart(df_agg).mark_text(dx=5, dy=-5, fontSize=10, color='white').encode(
    x='RankZ:Q',
    y='Stdev:Q',
    text='馬名:N'
)
quad = alt.Chart(quad_labels).mark_text(fontSize=14, fontWeight='bold', color='white').encode(
    x='RankZ:Q',
    y='Stdev:Q',
    text='label:N'
)
vline = alt.Chart(pd.DataFrame({'x':[50]})).mark_rule(color='gray').encode(x='x:Q')
hline = alt.Chart(pd.DataFrame({'y':[avg_st]})).mark_rule(color='gray').encode(y='y:Q')

chart = (points + labels + quad + vline + hline)
st.altair_chart(chart.properties(width=600, height=400).interactive(), use_container_width=True)

# ===== [T1] 上位馬（AvgZ>50）抽出 START =====
topN = df_agg[df_agg['AvgZ'] > 50].sort_values('RankZ', ascending=False).head(6).copy()
if len(topN) == 0:
    st.warning("平均スコア50超の馬がいません。閾値なしの上位6頭を暫定表示します。")
    topN = df_agg.sort_values('RankZ', ascending=False).head(6).copy()

topN['印'] = ['◎','〇','▲','☆','△','△'][:len(topN)]
st.subheader("上位馬（平均スコア>50のみ／根拠付き）")
st.table(topN[['馬名','印','根拠']])
# ===== [T1] 上位馬（AvgZ>50）抽出 END =====

with st.expander("▼『平均スコア』『安定度』の意味・基準を見る"):
    st.markdown("#### 平均スコア（AvgZ）")
    st.write(
        "- 過去成績（score_norm）から各馬ごとに平均を算出\n"
        "- 偏差値（RankZ）の算出基準にもなる\n"
        "- 数値が高いほど「安定して高成績」"
    )
    avg_mean = df_agg['AvgZ'].mean()
    avg_std = df_agg['AvgZ'].std()
    avg_med = df_agg['AvgZ'].median()
    st.write(f"【全体 平均: {avg_mean:.1f}　中央値: {avg_med:.1f}　標準偏差: {avg_std:.1f}】")
    fig, ax = plt.subplots()
    ax.hist(df_agg['AvgZ'], bins=10)
    ax.set_title("全馬の平均スコア分布", fontproperties=jp_font)
    ax.set_xlabel("平均スコア", fontproperties=jp_font)
    ax.set_ylabel("馬の数", fontproperties=jp_font)
    st.pyplot(fig)

    st.markdown("#### 安定度（Stdev）")
    st.write(
        "- 過去成績の「ばらつき」の大きさ（標準偏差）\n"
        "- 小さいほど「安定している」\n"
        "- 今回は安定度=マイナス標準偏差（大きいほど安定）で比較"
    )
    st.write(f"【全体 平均: {df_agg['Stdev'].mean():.1f}　中央値: {df_agg['Stdev'].median():.1f}　標準偏差: {df_agg['Stdev'].std():.1f}】")
    fig2, ax2 = plt.subplots()
    ax2.hist(df_agg['Stdev'], bins=10)
    ax2.set_title("全馬の安定度（標準偏差）分布", fontproperties=jp_font)
    ax2.set_xlabel("安定度（標準偏差）", fontproperties=jp_font)
    ax2.set_ylabel("馬の数", fontproperties=jp_font)
    st.pyplot(fig2)

# ========== 展開ロケーション（全頭・馬番） ==========
df_map = horses.copy()
df_map['印'] = df_map['馬名'].map(dict(zip(topN['馬名'], topN['印'])))

df_map['番'] = df_map['番'].astype(str).str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
df_map['番'] = pd.to_numeric(df_map['番'], errors='coerce')
df_map = df_map.dropna(subset=['番'])
df_map['番'] = df_map['番'].astype(int)

df_map['脚質'] = pd.Categorical(df_map['脚質'], categories=['逃げ','先行','差し','追込'], ordered=True)
df_map = df_map.sort_values(['番'])

fig, ax = plt.subplots(figsize=(10,3))
colors = {'逃げ':'red', '先行':'orange', '差し':'green', '追込':'blue'}

for i, row in df_map.iterrows():
    x = row['番']
    y = ['逃げ','先行','差し','追込'].index(row['脚質']) if row['脚質'] in ['逃げ','先行','差し','追込'] else np.nan
    if np.isnan(y): continue
    ax.scatter(x, y, color=colors.get(row['脚質'], 'gray'), s=200)
    label = f"{row['馬名']} {row['印'] if pd.notnull(row['印']) else ''}"
    ax.text(
        x, y, label,
        ha='center', va='center', color='white', fontsize=9, weight='bold',
        bbox=dict(facecolor=colors.get(row['脚質'], 'gray'), alpha=0.7, boxstyle='round'),
        fontproperties=jp_font
    )

ax.set_yticks([0,1,2,3])
ax.set_yticklabels(['逃げ','先行','差し','追込'], fontproperties=jp_font)
ax.set_xticks(sorted(df_map['番'].unique()))
ax.set_xticklabels([f"{i}番" for i in sorted(df_map['番'].unique())], fontproperties=jp_font)
ax.set_xlabel("馬番", fontproperties=jp_font)
ax.set_ylabel("脚質", fontproperties=jp_font)
ax.set_title("展開ロケーション（脚質×馬番／全頭）", fontproperties=jp_font)
st.pyplot(fig)

# --- 脚質カウント＆ペース/記号判定 ---
kakusitsu = ['逃げ','先行','差し','追込']
counter = df_map['脚質'].value_counts().reindex(kakusitsu, fill_value=0)

def pace_and_favor(counter):
    nige = counter['逃げ']
    sengo = counter['先行']
    pace = "ミドルペース"
    mark = {'逃げ':'△', '先行':'△', '差し':'△', '追込':'△'}
    if nige >= 3 or (nige==2 and sengo>=4):
        pace = "ハイペース"
        mark = {'逃げ':'△', '先行':'△', '差し':'◎', '追込':'〇'}
    elif nige == 1 and sengo <= 2:
        pace = "スローペース"
        mark = {'逃げ':'◎', '先行':'〇', '差し':'△', '追込':'×'}
    elif nige <= 1:
        pace = "ややスローペース"
        mark = {'逃げ':'〇', '先行':'◎', '差し':'△', '追込':'×'}
    else:
        pace = "ミドルペース"
        mark = {'逃げ':'〇', '先行':'◎', '差し':'〇', '追込':'△'}
    return pace, mark

pace_type, mark = pace_and_favor(counter)

st.markdown(
    "#### 脚質内訳｜" + "｜".join([f"{k}:{counter[k]}頭" for k in kakusitsu])
)
st.markdown(
    f"**【展開想定】{pace_type}**"
)

# --- 脚質ごとに該当馬（スコア順）を横並びで表示 ---
cols = st.columns(4)
for i, k in enumerate(kakusitsu):
    temp = df_map[df_map['脚質'] == k].copy()
    if not temp.empty:
        temp = temp.merge(df_agg[['馬名','AvgZ']], on='馬名', how='left')
        temp = temp.sort_values('AvgZ', ascending=False)
        names = "<br>".join(temp['馬名'].tolist())
    else:
        names = "該当馬なし"
    cols[i].markdown(f"**{k}　{mark[k]}**<br>{names}", unsafe_allow_html=True)

# 印と脚質・血統情報を全頭にマージ
印map = dict(zip(topN['馬名'], topN['印']))
horses = horses.merge(df_agg[['馬名','AvgZ','Stdev']], on='馬名', how='left')
horses['印'] = horses['馬名'].map(印map).fillna('')

# 血統情報もマージ
horses = horses.merge(blood_df, on='馬名', how='left', suffixes=('', '_血統'))

# コメント列
def ai_comment(row):
    base = ""
    if row['印'] == '◎':
        base += "本命評価。"
        if row['Stdev'] <= 8:
            base += "高い安定感で信頼度抜群。"
        else:
            base += "能力最上位もムラあり。"
    elif row['印'] == '〇':
        base += "対抗評価。"
        if row['Stdev'] <= 10:
            base += "近走安定しており軸候補。"
        else:
            base += "展開ひとつで逆転も。"
    elif row['印'] in ['▲','☆']:
        base += "上位グループの一角。"
        if row['Stdev'] > 15:
            base += "ムラがあり一発タイプ。"
        else:
            base += "安定型で堅実。"
    elif row['印'] == '△':
        base += "押さえ候補。"
        if row['Stdev'] < 12:
            base += "堅実だが勝ち切るまでは？"
        else:
            base += "展開次第で浮上も。"
    else:
        if pd.notna(row['AvgZ']) and pd.notna(row['Stdev']):
            if row['AvgZ'] >= 55 and row['Stdev'] < 13:
                base += "実力十分。ヒモ穴候補。"
            elif row['AvgZ'] < 45:
                base += "実績からは厳しい。"
            else:
                base += "決定打に欠ける。"

    if pd.notna(row['Stdev']) and row['Stdev'] >= 18:
        base += "波乱含み。"
    elif pd.notna(row['Stdev']) and row['Stdev'] <= 8:
        base += "非常に安定。"

    bloodtxt = str(row.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    bloodword = ""
    for k in keys:
        if k.strip() and k.strip().lower() in bloodtxt:
            bloodword = k.strip()
            break
    if bloodword:
        base += f"血統的にも注目（{bloodword}系統）。"

    style = str(row.get('脚質','')).strip()
    if style == '逃げ':
        base += "ハナを奪えれば粘り込み十分。"
    elif style == '先行':
        base += "先行力を活かして上位争い。"
    elif style == '差し':
        base += "展開が向けば末脚強烈。"
    elif style == '追込':
        base += "直線勝負の一撃に期待。"
    return base

horses['短評'] = horses.apply(ai_comment, axis=1)

# ===== [G] 重賞好走履歴（G1:1-5 / G2:1-4 / G3:1-3） START =====
# このExcelでは列名が『クラス名』『確定着順』『レース日』『レース名』
race_col = 'レース名'

def normalize_grade(x: str) -> str:
    """ 'G 3' / 'Ｇ３' / 'GⅢ' / 'GIII' / 'JpnIII' などを G1/G2/G3 に統一 """
    s = str(x).strip().upper()
    # 全角→半角
    trans = str.maketrans('Ｇ１２３４５６７８９０　', 'G1234567890 ')
    s = s.translate(trans)
    # 余分な空白を削除
    s = re.sub(r'\s+', '', s)          # 例: 'G 3' → 'G3'
    # ローマ数字→英字
    s = s.replace('Ⅲ', 'III').replace('Ⅱ', 'II').replace('Ⅰ', 'I')
    # Jpn をGに
    s = re.sub(r'JPNIII', 'GIII', s)
    s = re.sub(r'JPNII',  'GII',  s)
    s = re.sub(r'JPNI',   'GI',   s)
    # GI/GII/GIII → G1/G2/G3
    s = s.replace('GIII', 'G3').replace('GII', 'G2').replace('GI', 'G1')
    # 既に G1/G2/G3 ならそのまま。OP(L) 等は対象外
    return s if s in {'G1','G2','G3'} else s

def parse_pos(x) -> float:
    """ '1', '1着', '1位' などから先頭の数字だけを取る """
    if pd.isna(x): return np.nan
    m = re.search(r'\d+', str(x))
    return float(m.group()) if m else np.nan

gr = df_score.copy()
gr['GradeN']   = gr['クラス名'].apply(normalize_grade)
gr['着順num']  = gr['確定着順'].apply(parse_pos)

# G1/G2/G3 のみ対象
gr = gr[gr['GradeN'].isin(['G1','G2','G3'])].copy()

# 閾値マップ
thr_map = {'G1':5, 'G2':4, 'G3':3}
gr['閾値'] = gr['GradeN'].map(thr_map)

# 好走のみ
gr = gr[gr['着順num'].notna() & (gr['着順num'] <= gr['閾値'])].copy()

# 日付整形（2025.7.6 などもOK）
gr['_date']     = pd.to_datetime(gr['レース日'], errors='coerce')
gr['_date_str'] = gr['_date'].dt.strftime('%Y.%m.%d').fillna('日付不明')

def one_line(row):
    race = row[race_col] if pd.notna(row.get(race_col)) else 'レース名不明'
    return f"{race}　{row['GradeN']}　{int(row['着順num'])}着　{row['_date_str']}"

gr = gr.sort_values('_date', ascending=False)
grade_highlights = gr.groupby('馬名').apply(lambda d: [one_line(r) for _, r in d.iterrows()]).to_dict()

def highlight_text(name):
    lines = grade_highlights.get(name, [])
    return "重賞経験なし" if len(lines) == 0 else "\n".join(lines)

horses['重賞実績'] = horses['馬名'].apply(highlight_text)
# ===== [G] 重賞好走履歴（G1:1-5 / G2:1-4 / G3:1-3） END =====

st.subheader("■ 全頭AI診断コメント")
st.dataframe(horses[['馬名','印','脚質','血統','短評','AvgZ','Stdev','重賞実績']])

with st.expander("各馬の重賞好走履歴（クリックで展開）", expanded=False):
    for _, row in horses[['馬名']].iterrows():
        name = row['馬名']
        st.markdown(f"**{name}**")
        txt = grade_highlights.get(name, None)
        if not txt:
            st.write("　重賞経験なし")
        else:
            for line in grade_highlights[name]:
                st.write("　" + line)

# ===== [B1] 買い目生成＆資金配分 START =====
h1 = topN.iloc[0]['馬名'] if len(topN) >= 1 else None
h2 = topN.iloc[1]['馬名'] if len(topN) >= 2 else None

symbols = topN['印'].tolist()
names   = topN['馬名'].tolist()
others_names   = names[1:] if len(names) > 1 else []
others_symbols = symbols[1:] if len(symbols) > 1 else []

three = ['馬連','ワイド','馬単']
scenario_map = {
    '通常': three,
    'ちょい余裕': ['ワイド','三連複'],
    '余裕': ['ワイド','三連複','三連単']
}

main_share = 0.5
pur1 = int(round((total_budget * main_share * 1/4)  / 100) * 100)
pur2 = int(round((total_budget * main_share * 3/4)  / 100) * 100)
rem  = total_budget - (pur1 + pur2)

win_each   = int(round((pur1 / 2)  / 100) * 100)
place_each = int(round((pur2 / 2)  / 100) * 100)

st.subheader("■ 資金配分")
st.write(f"合計予算：{total_budget:,}円  単勝：{pur1:,}円  複勝：{pur2:,}円  残：{rem:,}円")

bets = []
if h1 is not None:
    bets += [
        {'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
        {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each},
    ]
if h2 is not None:
    bets += [
        {'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
        {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each},
    ]

parts = scenario_map[scenario]

if scenario == '通常':
    with st.expander("馬連・ワイド・馬単 から１券種を選択", expanded=True):
        choice = st.radio("購入券種", options=three, index=1)
        st.write(f"▶ {choice} に残り {rem:,}円 を充当")
    if len(others_names) > 0 and h1 is not None:
        share_each = int(round(rem / len(others_names) / 100) * 100)
        for nm, mk in zip(others_names, others_symbols):
            bets.append({'券種': choice, '印': f'◎–{mk}', '馬': h1, '相手': nm, '金額': share_each})
    else:
        st.info("相手がいないため連系はスキップ。")

elif scenario == 'ちょい余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 で消費します")
    n_w = len(others_names)
    n_t = len(list(combinations(others_names, 2)))
    total_line = n_w + n_t
    if total_line > 0 and h1 is not None:
        share_each = int(round(rem / total_line / 100) * 100)
        for nm, mk in zip(others_names, others_symbols):
            bets.append({'券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':share_each})
        for pair in combinations(others_names, 2):
            bets.append({'券種':'三連複','印':'◎-〇▲☆△△','馬':h1,'相手':'／'.join(pair),'金額':share_each})
    else:
        st.info("相手が足りないため連系はスキップ。")

elif scenario == '余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 ＋ 三連単フォーメーション で消費します")
    n_w     = len(others_names)
    n_tri3  = len(list(combinations(others_names,2)))
    second_opts = others_names[:2]
    combo3 = [(s,t) for s in second_opts for t in others_names if t!=s]
    n_tri1  = len(combo3)
    total_line = n_w + n_tri3 + n_tri1
    if total_line > 0 and h1 is not None:
        share_each = int(round(rem / total_line / 100) * 100)
        for nm, mk in zip(others_names, others_symbols):
            bets.append({'券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':share_each})
        for pair in combinations(others_names,2):
            bets.append({'券種':'三連複','印':'◎-〇▲☆△△','馬':h1,'相手':'／'.join(pair),'金額':share_each})
        for s,t in combo3:
            bets.append({'券種':'三連単フォーメーション','印':'◎-〇▲-〇▲☆△△','馬':h1,'相手':f"{s}／{t}",'金額':share_each})
    else:
        st.info("相手が足りないため連系はスキップ。")

df_bets = pd.DataFrame(bets)
df_bets['金額'] = df_bets['金額'].map(lambda x: f"{x:,}円" if x and x>0 else "")

unique_types = df_bets['券種'].unique().tolist() if len(df_bets)>0 else []
tabs = st.tabs(['サマリー'] + unique_types) if len(unique_types)>0 else st.tabs(['サマリー'])
with tabs[0]:
    st.subheader("■ 最終買い目一覧（全券種まとめ）")
    if len(df_bets)==0:
        st.info("現在、買い目はありません。")
    else:
        st.table(df_bets[['券種','印','馬','相手','金額']])

for i, typ in enumerate(unique_types, start=1):
    with tabs[i]:
        df_this = df_bets[df_bets['券種'] == typ]
        if len(df_this) == 0:
            st.info(f"{typ} の買い目はありません。")
        else:
            st.subheader(f"{typ} 買い目一覧")
            st.table(df_this[['券種','印','馬','相手','金額']])
# ===== [B1] 買い目生成＆資金配分 END =====
