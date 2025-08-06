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

st.subheader("馬一覧と脚質入力")
edited = st.data_editor(
    attrs,
    column_order=['枠','番','馬名','性別','年齢','脚質'],
    column_config={
        '脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込'])
    },
    use_container_width=True,
    num_rows='static'
)
horses = edited.copy()[['枠','番','馬名','性別','年齢','脚質']]

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
)

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
    bt  = besttime_w
    weight_factor = 1
    # 血統ボーナス（空白/全角スペース/大文字小文字対応）
    bloodline = str(r.get('血統','')).replace('\u3000',' ').replace('\n',' ').lower()
    bonus = 0
    for k in keys:
        if k.strip() and k.strip().lower() in bloodline:
            bonus = bp
            break
    return raw * sw * gw * stw * fw * aw * bt * weight_factor + bonus

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

# ========== 上位6頭＆説明 ==========
st.subheader("上位6頭（根拠付き）")
top6 = df_agg.sort_values('RankZ', ascending=False).head(6)
top6['印'] = ['◎','〇','▲','☆','△','△']
st.table(top6[['馬名','印','根拠']])

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
df_map['印'] = df_map['馬名'].map(dict(zip(top6['馬名'], top6['印'])))

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

# --- 脚質カウント表示 ---
kakusitsu = ['逃げ','先行','差し','追込']
counter = df_map['脚質'].value_counts().reindex(kakusitsu, fill_value=0)
st.markdown(
    "#### 脚質内訳"
    + "｜".join([f"{k}:{counter[k]}頭" for k in kakusitsu])
)

# --- 展開コメント自動生成 ---
def pace_comment(counter):
    nige = counter['逃げ']
    sengo = counter['先行']
    if nige >= 3:
        return "逃げ馬が多くハイペース濃厚。差し・追込有利な流れになりそう。"
    elif nige == 2:
        return "逃げ馬が複数いてペースは流れやすい。差し馬の台頭に注意。"
    elif nige == 1:
        return "逃げ馬が1頭で単騎逃げ濃厚。前残りの可能性高い。"
    else:
        return "純粋な逃げ馬不在。先行・差しが早めに動く持久戦か、意外な馬の逃げ残りも。"

pace_text = pace_comment(counter)
st.markdown(f"**【展開傾向】** {pace_text}")


# ========== 買い目生成＆資金配分 ==========
h1 = top6.iloc[0]['馬名']
h2 = top6.iloc[1]['馬名']
symbols = top6['印'].tolist()
names   = top6['馬名'].tolist()
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
st.subheader("■ 最終買い目一覧")
st.table(df_bets[['券種','印','馬','相手','金額']])
