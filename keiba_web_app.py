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
    return 50 + 10 * (s - s.mean()) / s.std(ddof=0)

def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

# ========== サイドバー ==========
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
weight_coeff = st.sidebar.slider("斤量効果強度", 0.0, 2.0, 1.0)
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
# --- 2枚目シートから芝の率＋ベストタイム列を自動検出 ---
name_col = sheet2.columns[2]  # 馬名の列（あなたのファイルでは3列目）

def find_col(patterns):
    for pat in patterns:
        for c in sheet2.columns:
            if re.search(pat, str(c)):
                return c
    return None

col_win  = find_col([r'勝率.*芝', r'芝.*勝率', r'^勝率$'])
col_quin = find_col([r'連対率.*芝', r'芝.*連対率', r'^連対率$'])
col_plc  = find_col([r'複勝率.*芝', r'芝.*複勝率', r'^複勝率$'])
col_bt   = find_col([r'ベスト.*タイム', r'Best.*Time', r'ﾍﾞｽﾄ.*ﾀｲﾑ'])

# 必須列が見つからない場合は列名を教えて（or ここに手動指定）
rate = sheet2[[name_col, col_win, col_quin, col_plc, col_bt]].copy()
rate.columns = ['馬名','勝率_芝','連対率_芝','複勝率_芝','ベストタイム']

# 数値化とタイム秒化
for c in ['勝率_芝','連対率_芝','複勝率_芝']:
    rate[c] = pd.to_numeric(rate[c], errors='coerce')

def parse_time_to_sec(x):
    s = str(x)
    m = re.match(r'^\s*(\d+)[\.\:](\d+)[\.\:](\d+)\s*$', s)  # 例: 1.07.3 → 1分07秒3
    if m:
        return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/10
    return pd.to_numeric(s, errors='coerce')  # 既に秒ならそのまま

rate['ベストタイム秒'] = rate['ベストタイム'].apply(parse_time_to_sec)

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

df_score = (
    df_score
    .merge(horses, on='馬名', how='inner')
    .merge(blood_df, on='馬名', how='left')
    .merge(rate[['馬名','勝率_芝','連対率_芝','複勝率_芝','ベストタイム秒']], on='馬名', how='left')
)

# ベストタイム正規化用のレンジ
bt_min = df_score['ベストタイム秒'].min(skipna=True)
bt_max = df_score['ベストタイム秒'].max(skipna=True)
bt_span = (bt_max - bt_min) if pd.notna(bt_min) and pd.notna(bt_max) and (bt_max > bt_min) else 1.0

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
        if agari_order == 1:   agari_bonus = agari1_bonus
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

    # 芝の勝率/連対率/複勝率で加点（0〜 win_w+quin_w+plc_w）
    rate_bonus = 0.0
    try:
        if pd.notna(r.get('勝率_芝', np.nan)):   rate_bonus += win_w  * (float(r['勝率_芝'])  / 100.0)
        if pd.notna(r.get('連対率_芝', np.nan)): rate_bonus += quin_w * (float(r['連対率_芝']) / 100.0)
        if pd.notna(r.get('複勝率_芝', np.nan)): rate_bonus += plc_w  * (float(r['複勝率_芝'])  / 100.0)
    except:
        pass

    # ベストタイム加点（速いほど+）0〜besttime_w
    bt_bonus = 0.0
    try:
        if pd.notna(r.get('ベストタイム秒', np.nan)):
            bt_norm = (bt_max - float(r['ベストタイム秒'])) / bt_span  # 速いほど1に近い
            bt_norm = max(0.0, min(1.0, bt_norm))
            bt_bonus = besttime_w * bt_norm
    except:
        pass

    total_bonus = blood_bonus + grade_point + agari_bonus + body_bonus + rate_bonus + bt_bonus
    return raw * sw * gw * stw * fw * aw * weight_factor + total_bonus

df_score['score_raw']  = df_score.apply(calc_score, axis=1)
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

df_agg['Stdev'] = df_agg['Stdev']  # 標準偏差そのまま

# 四象限ラベル
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

# 注釈文も下に表示
with st.expander("▶ 散布図の見方（クリックで開く）"):
    st.markdown("""
- この散布図は「右下ほど本命級」
- 横軸：偏差値（高いほど能力が高い）
- 縦軸：標準偏差（小さいほど安定）
- 右下：鉄板・本命
- 右上：波乱・ムラ馬
- 左下：堅実ヒモ
- 左上：消し・大穴
- 的中率重視なら右下、本命党は右下重視！
""")

# ========== 上位馬（平均スコア>50のみ／根拠付き） ==========
topN = df_agg[df_agg['AvgZ'] > 50].sort_values('RankZ', ascending=False).head(6).copy()
topN['印'] = ['◎','〇','▲','☆','△','△'][:len(topN)]
st.subheader("上位馬（平均スコア>50のみ／根拠付き）")
st.table(topN[['馬名','印','根拠']])

if len(topN) == 0:
    st.warning("平均スコア50超の馬がいません。")
    st.stop()

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

    st.info(
        "- **平均スコア**が高い＝「実力が高い」\n"
        "- **安定度（標準偏差）**が小さい＝「ムラが少なく信頼できる」\n"
        "- これらを両方見て、上位6頭や印の優先度を決めています"
    )

# ========== 展開ロケーション（全頭・馬番） ==========
df_map = horses.copy()
df_map['印'] = df_map['馬名'].map(dict(zip(topN['馬名'], topN['印'])))

# --- 馬番の安全変換（全角→半角, 数値化, 欠損除去）---
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
    if np.isnan(y): continue  # 脚質未入力はスキップ
    ax.scatter(x, y, color=colors.get(row['脚質'], 'gray'), s=200)
    label = f"{row['馬名']} {row['印'] if pd.notnull(row['印']) else ''}"
    ax.text(
        x, y, label, 
        ha='center', va='center', color='white', fontsize=9, weight='bold',
        bbox=dict(facecolor=colors.get(row['脚質'], 'gray'), alpha=0.7, boxstyle='round'),
        fontproperties=jp_font  # ←コレが重要！
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
out = ""
for k in kakusitsu:
    # 脚質ごとに該当馬（平均スコア順）
    temp = df_map[df_map['脚質'] == k].copy()
    if not temp.empty:
        # df_aggにスコア（AvgZ）をJOIN
        temp = temp.merge(df_agg[['馬名','AvgZ']], on='馬名', how='left')
        temp = temp.sort_values('AvgZ', ascending=False)
        names = "、".join(temp['馬名'].tolist())
    else:
        names = "該当馬なし"
    out += f"<b>{k} {mark[k]}</b><br>{names}<br><br>"

# --- 横並び（4列）っぽくするためカラムレイアウトで表示 ---
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

# 血統情報もくっつけておく（すでにhorsesに血統が入っていればこの行は不要）
horses = horses.merge(blood_df, on='馬名', how='left', suffixes=('', '_血統'))

# コメント列
def ai_comment(row):
    base = ""
    # ◎本命、〇対抗など印ごとに診断
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
        if row['AvgZ'] >= 55 and row['Stdev'] < 13:
            base += "実力十分。ヒモ穴候補。"
        elif row['AvgZ'] < 45:
            base += "実績からは厳しい。"
        else:
            base += "決定打に欠ける。"
    # ムラ・波乱
    if row['Stdev'] >= 18:
        base += "波乱含み。"
    elif row['Stdev'] <= 8:
        base += "非常に安定。"

    # 血統キーワード判定
    bloodtxt = str(row.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    bloodword = ""
    for k in keys:
        if k.strip() and k.strip().lower() in bloodtxt:
            bloodword = k.strip()
            break
    if bloodword:
        base += f"血統的にも注目（{bloodword}系統）。"

    # 脚質コメント
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

st.subheader("■ 全頭AI診断コメント")
# 必要なら表示項目を増減してください
st.dataframe(horses[['馬名','印','脚質','血統','短評','AvgZ','Stdev']])

# ========== 買い目生成＆資金配分 ==========
h1 = topN.iloc[0]['馬名']
h2 = topN.iloc[1]['馬名']
symbols = topN['印'].tolist()
names   = topN['馬名'].tolist()
others_names   = names[1:]
others_symbols = symbols[1:]

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
bets += [
    {'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
    {'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
    {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each},
    {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each},
]

parts = scenario_map[scenario]

if scenario == '通常':
    with st.expander("馬連・ワイド・馬単 から１券種を選択", expanded=True):
        choice = st.radio("購入券種", options=three, index=1)
        st.write(f"▶ {choice} に残り {rem:,}円 を充当")
    share_each = int(round(rem / len(others_names) / 100) * 100)
    for nm, mk in zip(others_names, others_symbols):
        bets.append({
            '券種': choice,
            '印':   f'◎–{mk}',
            '馬':    h1,
            '相手':  nm,
            '金額':  share_each
        })

elif scenario == 'ちょい余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 で消費します")
    n_w = len(others_names)
    n_t = len(list(combinations(others_names, 2)))
    share_each = int(round(rem / (n_w + n_t) / 100) * 100)
    for nm, mk in zip(others_names, others_symbols):
        bets.append({
            '券種':'ワイド',
            '印':  f'◎–{mk}',
            '馬':   h1,
            '相手': nm,
            '金額': share_each
        })
    for pair in combinations(others_names, 2):
        bets.append({
            '券種':'三連複',
            '印':  '◎-〇▲☆△△',
            '馬':   h1,
            '相手':'／'.join(pair),
            '金額': share_each
        })

elif scenario == '余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 ＋ 三連単フォーメーション で消費します")
    n_w     = len(others_names)
    n_tri3  = len(list(combinations(others_names,2)))
    second_opts = others_names[:2]
    combo3 = [(s,t) for s in second_opts for t in others_names if t!=s]
    n_tri1  = len(combo3)
    total_line = n_w + n_tri3 + n_tri1
    share_each = int(round(rem / total_line / 100) * 100)
    for nm, mk in zip(others_names, others_symbols):
        bets.append({
            '券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':share_each
        })
    for pair in combinations(others_names,2):
        bets.append({
            '券種':'三連複','印':'◎-〇▲☆△△','馬':h1,
            '相手':'／'.join(pair),'金額':share_each
        })
    for s,t in combo3:
        bets.append({
            '券種':'三連単フォーメーション','印':'◎-〇▲-〇▲☆△△',
            '馬':h1,'相手':f"{s}／{t}",'金額':share_each
        })

df_bets = pd.DataFrame(bets)
df_bets['金額'] = df_bets['金額'].map(lambda x: f"{x:,}円" if x>0 else "")
# ---------- タブで券種ごとに表示 ----------
unique_types = df_bets['券種'].unique().tolist()
tabs = st.tabs(['サマリー'] + unique_types)
for i, typ in enumerate([''] + unique_types):
    with tabs[i]:
        if i == 0:
            st.subheader("■ 最終買い目一覧（全券種まとめ）")
            st.table(df_bets[['券種','印','馬','相手','金額']])
        else:
            df_this = df_bets[df_bets['券種'] == typ]
            if len(df_this) == 0:
                st.info(f"{typ} の買い目はありません。")
            else:
                st.subheader(f"{typ} 買い目一覧")
                st.table(df_this[['券種','印','馬','相手','金額']])

