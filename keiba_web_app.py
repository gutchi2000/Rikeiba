import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager
import altair as alt
from itertools import combinations

# =========== 日本語フォントの用意 ==========
# ipaexg.ttf（IPAexゴシック）がある場合
try:
    jp_font = font_manager.FontProperties(fname="ipaexg.ttf")
except:
    jp_font = font_manager.FontProperties(fname="C:/Windows/Fonts/meiryo.ttc") # Windowsの場合

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAPGothic', 'Meiryo', 'MS Gothic']

# ---- ヘルパー ----
def z_score(s: pd.Series) -> pd.Series:
    return 50 + 10 * (s - s.mean()) / s.std(ddof=0)

def season_of(month: int) -> str:
    if 3 <= month <= 5: return '春'
    if 6 <= month <= 8: return '夏'
    if 9 <= month <= 11: return '秋'
    return '冬'

# ------------- サイドバー -------------
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

# ---- メイン ----
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
    aw  = age_w
    bt  = besttime_w
    weight_factor = 1
    bonus = bp if any(k in str(r.get('血統', '')) for k in keys) else 0
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

# ========== グラフ系：ここを日本語ラベル＆フォント指定 ==========
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
# --- サイドバーからの変数取得は省略 ---
# total_budget, scenario, top6, etc. が定義済みとします

# ——————————————
# ―― 資金配分 〜 最終買い目一覧 （完成形） ――
# ——————————————

# ◎／〇 の馬リスト準備
h1 = top6.iloc[0]['馬名']                  # ◎馬
h2 = top6.iloc[1]['馬名']                  # 〇馬
symbols = top6['印'].tolist()              # ['◎','〇','▲','☆','△','△']
names   = top6['馬名'].tolist()            # [h1,h2,h3,h4,h5,h6]
others_names   = names[1:]                 # ['〇馬名','▲馬名','☆馬名','△馬名','△馬名']
others_symbols = symbols[1:]               # ['〇','▲','☆','△','△']

# ——————————————
# ◎／〇 の馬リスト準備（省略）

# --- シナリオ別 券種リスト定義 ---
three = ['馬連','ワイド','馬単']
scenario_map = {
    '通常': three,
    'ちょい余裕': ['ワイド','三連複'],
    '余裕': ['ワイド','三連複','三連単']
}

# --- 資金配分計算 ---
main_share = 0.5
pur1 = int(round((total_budget * main_share * 1/4)  / 100) * 100)
pur2 = int(round((total_budget * main_share * 3/4)  / 100) * 100)
rem  = total_budget - (pur1 + pur2)

win_each   = int(round((pur1 / 2)  / 100) * 100)
place_each = int(round((pur2 / 2)  / 100) * 100)

st.subheader("■ 資金配分")
st.write(f"合計予算：{total_budget:,}円  単勝：{pur1:,}円  複勝：{pur2:,}円  残：{rem:,}円")

bets = []
# 単勝・複勝（◎／〇 各2頭ずつ）
bets += [
    {'券種':'単勝','印':'◎','馬':h1,'相手':'','金額':win_each},
    {'券種':'単勝','印':'〇','馬':h2,'相手':'','金額':win_each},
    {'券種':'複勝','印':'◎','馬':h1,'相手':'','金額':place_each},
    {'券種':'複勝','印':'〇','馬':h2,'相手':'','金額':place_each},
]

# シナリオごとの残予算割当
parts = scenario_map[scenario]

# — 通常 —  
if scenario == '通常':
    with st.expander("馬連・ワイド・馬単 から１券種を選択", expanded=True):
        choice = st.radio("購入券種", options=three, index=1)
        st.write(f"▶ {choice} に残り {rem:,}円 を充当")
    # 選択した1種を均等割り
    # ここは「◎–相手」一行ずつ
    share_each = int(round(rem / len(others_names) / 100) * 100)
    for nm, mk in zip(others_names, others_symbols):
        bets.append({
            '券種': choice,
            '印':   f'◎–{mk}',
            '馬':    h1,
            '相手':  nm,
            '金額':  share_each
        })

# — ちょい余裕 —  
elif scenario == 'ちょい余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 で消費します")
    # ワイドの行数
    n_w = len(others_names)
    # 三連複の組み合わせ数 C(5,2)=10
    n_t = len(list(combinations(others_names, 2)))
    # 合計行数で等分
    share_each = int(round(rem / (n_w + n_t) / 100) * 100)
    # ワイド
    for nm, mk in zip(others_names, others_symbols):
        bets.append({
            '券種':'ワイド',
            '印':  f'◎–{mk}',
            '馬':   h1,
            '相手': nm,
            '金額': share_each
        })
    # 三連複
    for pair in combinations(others_names, 2):
        bets.append({
            '券種':'三連複',
            '印':  '◎-〇▲☆△△',
            '馬':   h1,
            '相手':'／'.join(pair),
            '金額': share_each
        })


# — 余裕 —  
elif scenario == '余裕':
    st.write("▶ 残り予算を ワイド ＋ 三連複 ＋ 三連単フォーメーション で消費します")
    # 各券種の組み合わせ数を求める
    n_w     = len(others_names)
    n_tri3  = len(list(combinations(others_names,2)))
    # 三連単フォーメーションの8通り
    second_opts = others_names[:2]
    combo3 = [(s,t) for s in second_opts for t in others_names if t!=s]
    n_tri1  = len(combo3)  # =8
    total_line = n_w + n_tri3 + n_tri1
    share_each = int(round(rem / total_line / 100) * 100)
    # ワイド
    for nm, mk in zip(others_names, others_symbols):
        bets.append({
            '券種':'ワイド','印':f'◎–{mk}','馬':h1,'相手':nm,'金額':share_each
        })
    # 三連複
    for pair in combinations(others_names,2):
        bets.append({
            '券種':'三連複','印':'◎-〇▲☆△△','馬':h1,
            '相手':'／'.join(pair),'金額':share_each
        })
    # 三連単フォーメーション
    for s,t in combo3:
        bets.append({
            '券種':'三連単フォーメーション','印':'◎-〇▲-〇▲☆△△',
            '馬':h1,'相手':f"{s}／{t}",'金額':share_each
        })

# --- 最終テーブル表示 ---
df_bets = pd.DataFrame(bets)
df_bets['金額'] = df_bets['金額'].map(lambda x: f"{x:,}円" if x>0 else "")
st.subheader("■ 最終買い目一覧")
st.table(df_bets[['券種','印','馬','相手','金額']])
