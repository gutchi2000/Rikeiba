import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations
import altair as alt

# =========== 日本語フォント ==========
try:
    jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
except:
    jp_font = font_manager.FontProperties(fname="C:/Windows/Fonts/meiryo.ttc")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']

def z_score(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([50]*len(s), index=s.index)
    return 50 + 10 * (s - s.mean()) / std

def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

# ========== サイドバー ==========
st.sidebar.header("パラメータ設定")
lambda_part  = st.sidebar.slider("出走ボーナス λ", 0.0, 1.0, 0.5, 0.05)
orig_weight  = st.sidebar.slider("OrigZ の重み", 0.0, 1.0, 0.5, 0.05)  # （現状未使用）
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

weight_coeff = st.sidebar.slider("斤量効果強度", 0.0, 2.0, 1.0)  # （現状未使用）
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
# ===== [M1] END =====

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

# ===== [M2] マージ =====
df_score = (
    df_score
    .merge(horses, on='馬名', how='inner')
    .merge(blood_df, on='馬名', how='left')
    .merge(rate[['馬名','勝率_芝','連対率_芝','複勝率_芝','ベストタイム秒']], on='馬名', how='left')
)

# ===== [M3] ベストタイム正規化レンジ =====
bt_min = df_score['ベストタイム秒'].min(skipna=True)
bt_max = df_score['ベストタイム秒'].max(skipna=True)
bt_span = (bt_max - bt_min) if pd.notna(bt_min) and pd.notna(bt_max) and (bt_max > bt_min) else 1.0

# ===== スコア算出 =====
st.subheader("血統キーワードとボーナス")
keys = st.text_area("系統名を1行ずつ入力", height=100).splitlines()
bp   = st.slider("血統ボーナス点数", 0, 20, 5)

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

    # 血統ボーナス
    bloodline = str(r.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    blood_bonus = 0
    for k in keys:
        if k.strip() and k.strip().lower() in bloodline:
            blood_bonus = bp
            break

    # 重賞名ボーナス（GⅠ〜GⅢのみ）
    grade_point = grade_bonus if str(r.get('クラス名','')) in ['GⅠ','GⅡ','GⅢ'] else 0

    # 上がり3F順位ボーナス
    agari_bonus = 0
    try:
        ao = int(r.get('上3F順位', np.nan))
        if   ao == 1: agari_bonus = agari1_bonus
        elif ao == 2: agari_bonus = agari2_bonus
        elif ao == 3: agari_bonus = agari3_bonus
    except: pass

    # 馬体重適正
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
    except: pass

    # 芝の勝率/連対率/複勝率
    rate_bonus = 0.0
    try:
        if pd.notna(r.get('勝率_芝', np.nan)):   rate_bonus += win_w  * (float(r['勝率_芝'])  / 100.0)
        if pd.notna(r.get('連対率_芝', np.nan)): rate_bonus += quin_w * (float(r['連対率_芝']) / 100.0)
        if pd.notna(r.get('複勝率_芝', np.nan)): rate_bonus += plc_w  * (float(r['複勝率_芝'])  / 100.0)
    except: pass

    # ベストタイム（速いほど+）
    bt_bonus = 0.0
    try:
        if pd.notna(r.get('ベストタイム秒', np.nan)):
            bt_norm = (bt_max - float(r['ベストタイム秒'])) / bt_span
            bt_norm = max(0.0, min(1.0, bt_norm))
            bt_bonus = besttime_w * bt_norm
    except: pass

    total_bonus = blood_bonus + grade_point + agari_bonus + body_bonus + rate_bonus + bt_bonus
    return raw * sw * gw * stw * fw * aw + total_bonus

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
    if row['RankZ'] >= 65: base += "非常に高評価。"
    elif row['RankZ'] >= 55: base += "高水準。"
    if row['Stdev'] < 8: base += "安定感抜群。"
    elif row['Stdev'] < 13: base += "比較的安定。"
    if row['馬名'] in df_score[df_score['score_raw'] > df_score['score_raw'].mean() + 10]['馬名'].values:
        base += "血統・脚質等もプラス評価。"
    return base

df_agg['根拠'] = df_agg.apply(reason, axis=1)

# ========== 散布図 ==========
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
    x='RankZ:Q', y='Stdev:Q', text='馬名:N'
)
vline = alt.Chart(pd.DataFrame({'x':[50]})).mark_rule(color='gray').encode(x='x:Q')
hline = alt.Chart(pd.DataFrame({'y':[avg_st]})).mark_rule(color='gray').encode(y='y:Q')
st.altair_chart((points + labels + vline + hline + alt.Chart(quad_labels).mark_text(fontSize=14, fontWeight='bold', color='white').encode(x='RankZ:Q', y='Stdev:Q', text='label:N')).properties(width=600, height=400).interactive(), use_container_width=True)

# ===== 上位馬抽出 =====
topN = df_agg[df_agg['AvgZ'] > 50].sort_values('RankZ', ascending=False).head(6).copy()
if len(topN) == 0:
    st.warning("平均スコア50超の馬がいません。閾値なしの上位6頭を暫定表示します。")
    topN = df_agg.sort_values('RankZ', ascending=False).head(6).copy()
topN['印'] = ['◎','〇','▲','☆','△','△'][:len(topN)]
st.subheader("上位馬（平均スコア>50のみ／根拠付き）")
st.table(topN[['馬名','印','根拠']])

# ========== 展開ロケーション ==========
df_map = horses.copy()
df_map['印'] = df_map['馬名'].map(dict(zip(topN['馬名'], topN['印'])))
df_map['番'] = pd.to_numeric(df_map['番'].astype(str).str.translate(str.maketrans('０１２３４５６７８９', '0123456789')), errors='coerce')
df_map = df_map.dropna(subset=['番']).astype({'番':int})
df_map['脚質'] = pd.Categorical(df_map['脚質'], categories=['逃げ','先行','差し','追込'], ordered=True)
df_map = df_map.sort_values(['番'])

fig, ax = plt.subplots(figsize=(10,3))
colors = {'逃げ':'red', '先行':'orange', '差し':'green', '追込':'blue'}
for _, row in df_map.iterrows():
    if row['脚質'] in colors:
        x = row['番']; y = ['逃げ','先行','差し','追込'].index(row['脚質'])
        ax.scatter(x, y, color=colors[row['脚質']], s=200)
        label = f"{row['馬名']} {row['印'] if pd.notnull(row['印']) else ''}"
        ax.text(x, y, label, ha='center', va='center', color='white', fontsize=9, weight='bold',
                bbox=dict(facecolor=colors[row['脚質']], alpha=0.7, boxstyle='round'),
                fontproperties=jp_font)
ax.set_yticks([0,1,2,3]); ax.set_yticklabels(['逃げ','先行','差し','追込'], fontproperties=jp_font)
ax.set_xticks(sorted(df_map['番'].unique())); ax.set_xticklabels([f"{i}番" for i in sorted(df_map['番'].unique())], fontproperties=jp_font)
ax.set_xlabel("馬番", fontproperties=jp_font); ax.set_ylabel("脚質", fontproperties=jp_font)
ax.set_title("展開ロケーション（脚質×馬番／全頭）", fontproperties=jp_font)
st.pyplot(fig)

kakusitsu = ['逃げ','先行','差し','追込']
counter = df_map['脚質'].value_counts().reindex(kakusitsu, fill_value=0)
def pace_and_favor(counter):
    nige = counter['逃げ']; sengo = counter['先行']
    if nige >= 3 or (nige==2 and sengo>=4): return "ハイペース", {'逃げ':'△','先行':'△','差し':'◎','追込':'〇'}
    if nige == 1 and sengo <= 2:           return "スローペース", {'逃げ':'◎','先行':'〇','差し':'△','追込':'×'}
    if nige <= 1:                           return "ややスローペース", {'逃げ':'〇','先行':'◎','差し':'△','追込':'×'}
    return "ミドルペース", {'逃げ':'〇','先行':'◎','差し':'〇','追込':'△'}
pace_type, mark = pace_and_favor(counter)
st.markdown("#### 脚質内訳｜" + "｜".join([f"{k}:{counter[k]}頭" for k in kakusitsu]))
st.markdown(f"**【展開想定】{pace_type}**")

cols = st.columns(4)
for i, k in enumerate(kakusitsu):
    temp = df_map[df_map['脚質'] == k].merge(df_agg[['馬名','AvgZ']], on='馬名', how='left').sort_values('AvgZ', ascending=False)
    names = "<br>".join(temp['馬名'].tolist()) if not temp.empty else "該当馬なし"
    cols[i].markdown(f"**{k}　{mark[k]}**<br>{names}", unsafe_allow_html=True)

# ===== horses に情報付与 =====
印map = dict(zip(topN['馬名'], topN['印']))
horses = horses.merge(df_agg[['馬名','AvgZ','Stdev']], on='馬名', how='left')
horses['印'] = horses['馬名'].map(印map).fillna('')
horses = horses.merge(blood_df, on='馬名', how='left', suffixes=('', '_血統'))

def ai_comment(row):
    base = ""
    if row['印'] == '◎':
        base += "本命評価。"; base += "高い安定感で信頼度抜群。" if row['Stdev']<=8 else "能力最上位もムラあり。"
    elif row['印'] == '〇':
        base += "対抗評価。"; base += "近走安定しており軸候補。" if row['Stdev']<=10 else "展開ひとつで逆転も。"
    elif row['印'] in ['▲','☆']:
        base += "上位グループの一角。"; base += "ムラがあり一発タイプ。" if row['Stdev']>15 else "安定型で堅実。"
    elif row['印'] == '△':
        base += "押さえ候補。"; base += "堅実だが勝ち切るまでは？" if row['Stdev']<12 else "展開次第で浮上も。"
    else:
        if pd.notna(row['AvgZ']) and pd.notna(row['Stdev']):
            base += "実力十分。ヒモ穴候補。" if (row['AvgZ']>=55 and row['Stdev']<13) else ("実績からは厳しい。" if row['AvgZ']<45 else "決定打に欠ける。")
    if pd.notna(row['Stdev']) and row['Stdev'] >= 18: base += "波乱含み。"
    elif pd.notna(row['Stdev']) and row['Stdev'] <= 8: base += "非常に安定。"
    bloodtxt = str(row.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    for k in keys:
        if k.strip() and k.strip().lower() in bloodtxt:
            base += f"血統的にも注目（{k.strip()}系統）。"; break
    style = str(row.get('脚質','')).strip()
    base += {"逃げ":"ハナを奪えれば粘り込み十分。","先行":"先行力を活かして上位争い。","差し":"展開が向けば末脚強烈。","追込":"直線勝負の一撃に期待。"}.get(style,"")
    return base

horses['短評'] = horses.apply(ai_comment, axis=1)

# ===== [G0] 重賞好走抽出（G1:1-5 / G2:1-4 / G3:1-3 ＋ 上がり3F） =====
# 列推定
race_col = next((c for c in ['レース名','競走名','レース','名称'] if c in df_score.columns), None)
ag_col   = next((c for c in ['上がり3Fタイム','上がり3F','上がり３Ｆ','上3Fタイム','上3F'] if c in df_score.columns), None)
grade_col_guess = next((c for c in ['クラス名','グレード','格','条件','クラス','レースグレード'] if c in df_score.columns), None)

# 文字列から G1/G2/G3 を抽出（クラス列でもレース名でもOK）
def extract_grade_any(s: str | float | int) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)): 
        return None
    x = str(s)
    # 全角/ローマ数字を正規化
    x = x.replace('Ｇ','G').replace('（','(').replace('）',')')
    x = x.replace('Ⅰ','I').replace('Ⅱ','II').replace('Ⅲ','III')
    # G + ローマ数字 → G1/2/3
    x = re.sub(r'G\s*III', 'G3', x, flags=re.I)
    x = re.sub(r'G\s*II',  'G2', x, flags=re.I)
    x = re.sub(r'G\s*I',   'G1', x, flags=re.I)
    # どこかに G1/2/3 があれば拾う（括弧内でも可）
    m = re.search(r'G\s*([123])', x, flags=re.I)
    return f"G{m.group(1)}" if m else None

# 1行から等級を決める（クラス列→ダメならレース名）
def grade_from_row(row) -> str | None:
    g = extract_grade_any(row[grade_col_guess]) if grade_col_guess in row else None
    if not g and race_col:
        g = extract_grade_any(row[race_col])
    return g

# 着順列
finish_col = '確定着順'  # ここはあなたのファイルで既に使えているので固定
df_g = df_score.copy()
df_g['GradeN']   = df_g.apply(grade_from_row, axis=1)
df_g['着順num']  = pd.to_numeric(df_g[finish_col], errors='coerce')
df_g['_date']    = pd.to_datetime(df_g['レース日'], errors='coerce')
df_g['_date_str']= df_g['_date'].dt.strftime('%Y.%m.%d').fillna('日付不明')

# 閾値
thr_map = {'G1':5, 'G2':4, 'G3':3}
df_g = df_g[df_g['GradeN'].isin(thr_map.keys()) & df_g['着順num'].notna()].copy()
df_g = df_g[df_g.apply(lambda r: r['着順num'] <= thr_map[r['GradeN']], axis=1)]
df_g = df_g.sort_values(['馬名','_date'], ascending=[True, False])

# 1行表示文字列（上がり3F付）
def make_line(r):
    race = r[race_col] if race_col else 'レース名不明'
    ag   = f"　上がり3F {r[ag_col]}" if (ag_col and pd.notna(r[ag_col])) else ""
    return f"{race}　{r['GradeN']}　{int(r['着順num'])}着　{r['_date_str']}{ag}"

grade_highlights = df_g.groupby('馬名').apply(
    lambda d: [make_line(row) for _, row in d.iterrows()]
).to_dict()

# ===== ハイライト表示（上位は展開、その他は折り畳み） =====
st.subheader("■ 重賞好走ハイライト（上がり3F付き）")
top_names = topN['馬名'].tolist()

st.markdown("##### 上位馬（展開済み）")
for name in top_names:
    lines = grade_highlights.get(name, [])
    st.markdown(f"**{name}**")
    if not lines:
        st.write("　重賞経験なし")
    else:
        st.markdown("・" + "<br>・".join(lines), unsafe_allow_html=True)

rest_names = horses.loc[~horses['馬名'].isin(top_names), '馬名']
if len(rest_names) > 0:
    st.markdown("##### その他の馬（必要なら開く）")
    for name in rest_names:
        with st.expander(name, expanded=False):
            lines = grade_highlights.get(name, [])
            if not lines:
                st.write("重賞経験なし")
            else:
                st.markdown("・" + "<br>・".join(lines), unsafe_allow_html=True)

# ===== ハイライト表示（上位は展開、その他は折り畳み） =====
st.subheader("■ 重賞好走ハイライト（上がり3F付き）")
top_names = topN['馬名'].tolist()

st.markdown("##### 上位馬（展開済み）")
for name in top_names:
    lines = grade_highlights.get(name, [])
    st.markdown(f"**{name}**")
    if not lines:
        st.write("　重賞経験なし")
    else:
        st.markdown("・" + "<br>・".join(lines), unsafe_allow_html=True)

rest_names = horses.loc[~horses['馬名'].isin(top_names), '馬名']
if len(rest_names) > 0:
    st.markdown("##### その他の馬（必要なら開く）")
    for name in rest_names:
        with st.expander(name, expanded=False):
            lines = grade_highlights.get(name, [])
            if not lines:
                st.write("重賞経験なし")
            else:
                st.markdown("・" + "<br>・".join(lines), unsafe_allow_html=True)

# ===== 買い目生成＆資金配分 =====
h1 = topN.iloc[0]['馬名'] if len(topN) >= 1 else None
h2 = topN.iloc[1]['馬名'] if len(topN) >= 2 else None

symbols = topN['印'].tolist()
names   = topN['馬名'].tolist()
others_names   = names[1:] if len(names) > 1 else []
others_symbols = symbols[1:] if len(symbols) > 1 else []

three = ['馬連','ワイド','馬単']
scenario_map = {'通常': three, 'ちょい余裕': ['ワイド','三連複'], '余裕': ['ワイド','三連複','三連単']}

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
    bets += [{'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
             {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each}]
if h2 is not None:
    bets += [{'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
             {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each}]

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
    n_w = len(others_names); n_t = len(list(combinations(others_names, 2)))
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
    n_w = len(others_names); n_tri3 = len(list(combinations(others_names,2)))
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
    if len(df_bets)==0: st.info("現在、買い目はありません。")
    else: st.table(df_bets[['券種','印','馬','相手','金額']])
for i, typ in enumerate(unique_types, start=1):
    with tabs[i]:
        df_this = df_bets[df_bets['券種'] == typ]
        st.subheader(f"{typ} 買い目一覧")
        st.table(df_this[['券種','印','馬','相手','金額']]) if len(df_this)>0 else st.info(f"{typ} の買い目はありません。")
