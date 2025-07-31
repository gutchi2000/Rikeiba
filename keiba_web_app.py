import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import combinations, permutations

# 日本語フォント設定
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(fname='ipaexg.ttf').get_name()

st.title('競馬予想アプリ（完成版）')

# サイドバー設定
st.sidebar.subheader('パラメータ設定')
male_weight = st.sidebar.number_input('牡の重み', min_value=0.0, value=1.1, step=0.01, format="%.2f")
female_weight = st.sidebar.number_input('牝の重み', min_value=0.0, value=1.0, step=0.01, format="%.2f")
gelding_weight = st.sidebar.number_input('せんの重み', min_value=0.0, value=0.95, step=0.01, format="%.2f")
summer_weight = st.sidebar.number_input('夏季重み', min_value=0.0, value=1.1, step=0.01, format="%.2f")

# --- データアップロード: レース成績シートと馬情報シート ---
# Excelファイルの1枚目を成績、2枚目を馬情報として読み込む
uploaded_file = st.file_uploader('成績&馬情報データをアップロード (Excel)', type=['xlsx'])
if not uploaded_file:
    st.stop()
# pandas で読み込み
xls = pd.ExcelFile(uploaded_file)
# 成績データ (1枚目)
df = xls.parse(sheet_name=0, parse_dates=['レース日'])
# 馬情報 (2枚目): 馬名, 性別, 年齢 のみ使用
stats = xls.parse(sheet_name=1, usecols=['馬名','性別','年齢'])
# 馬情報を成績データに結合
df = df.merge(stats, on='馬名', how='left')(stats, on='馬名', how='left')

# --- 脚質入力テーブル ---
equine_list = df['馬名'].unique().tolist()
style_df = pd.DataFrame({'馬名': equine_list, '脚質': ['差し']*len(equine_list)})
edited = st.data_editor(
    style_df,
    column_config={'脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込'])},
    use_container_width=True,
    key='editor'
)
style_map = dict(zip(edited['馬名'], edited['脚質']))

# --- 血統HTMLアップロード ---
ped_df = None
html_file = st.file_uploader('血統表をアップロード (HTML)', type=['html'])
if html_file:
    try:
        ped_df = pd.read_html(html_file.read())[0].set_index('馬名')
    except:
        ped_df = None

# --- 強調種牡馬リスト ---
ped_input = st.text_area('強調種牡馬リストを入力 (カンマ区切り)', '')
priority_sires = [s.strip() for s in ped_input.split(',') if s.strip()]

# --- 必要列チェック ---
required_cols = ['馬名','年齢','性別','レース日','頭数','クラス名','確定着順','上がり3Fタイム',
                 'Ave-3F','馬場状態','斤量','増減','単勝オッズ']
missing = set(required_cols) - set(df.columns)
if missing:
    st.error(f"不足列: {missing}")
    st.stop()

# --- ファクター関数 ---
def eval_pedigree(row):
    sire = ped_df.at[row['馬名'], '父馬'] if ped_df is not None and row['馬名'] in ped_df.index else ''
    return 1.2 if sire in priority_sires else 1.0

def style_factor(row):
    return {'逃げ':1.2,'先行':1.1,'差し':1.0,'追込':0.9}.get(style_map.get(row['馬名'], ''), 1.0)

def age_factor(age):
    peak = 5
    return 1 + 0.2*(1 - abs(age-peak)/peak)

def sex_factor(sex):
    # 性別: 牡, 牝, セ の略記に対応
    return {'牡': male_weight, '牝': female_weight, 'セ': gelding_weight}.get(sex, 1.0)

def summer_factor(date):
    return summer_weight if date.month in [6,7,8] else 1.0

# --- 各ファクター適用 ---
df['pedigree_factor'] = df.apply(eval_pedigree, axis=1)
df['style_factor'] = df.apply(style_factor, axis=1)
df['age_factor'] = df['年齢'].apply(age_factor)
df['sex_factor'] = df['性別'].apply(sex_factor)
df['summer_factor'] = df['レース日'].apply(summer_factor)

# --- 基本スコア ---
GRADE = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
         '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
raw = df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['raw'] = raw * df['pedigree_factor'] * df['style_factor'] * df['age_factor'] * df['sex_factor'] * df['summer_factor']

# --- 正規化 ---
jmax,jmin = df['斤量'].max(), df['斤量'].min()
df['raw_norm'] = (df['raw'] - 1) / (10 * df['頭数'] - 1)
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
df['jin_norm'] = (jmax - df['斤量']) / (jmax - jmin)
wmean = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / wmean

# --- Zスコア化 ---
metrics = ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm',
           'pedigree_factor','style_factor','age_factor','sex_factor','summer_factor']
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = (df[m] - mu) / sd if sd != 0 else 0

# --- 合成偏差値化 ---
weights = {'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,
           'Z_jin_norm':1,'Z_wdiff_norm':1,'Z_pedigree_factor':3,
           'Z_age_factor':2,'Z_style_factor':2,'Z_sex_factor':2,'Z_summer_factor':2}
tot_w = sum(weights.values())
df['total_z'] = sum(df[k] * w for k, w in weights.items()) / tot_w
zmin, zmax = df['total_z'].min(), df['total_z'].max()
df['偏差値'] = 30 + (df['total_z'] - zmin) / (zmax - zmin) * 40

# --- 馬別集計 ---
summary = df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
summary.columns = ['馬名','平均偏差値','安定性']
summary['バランススコア'] = summary['平均偏差値'] - summary['安定性']

# --- 表示 ---
st.subheader('馬別 評価一覧')
st.dataframe(summary.sort_values('バランススコア',ascending=False).reset_index(drop=True))

# --- 上位10頭表示 ---
st.subheader('偏差値上位10頭')
top10 = summary.sort_values('平均偏差値',ascending=False).head(10)
st.table(top10[['馬名','平均偏差値']])

# --- 本日の予想6頭表示 ---
combined = summary.sort_values('バランススコア',ascending=False).head(6)
st.subheader('本日の予想6頭')
st.table(combined)

# --- 可視化1: バランススコア ---
fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.barh(combined['馬名'], combined['バランススコア'])
ax1.invert_yaxis()
ax1.set_xlabel('バランススコア')
st.pyplot(fig1)

# --- 可視化2: 偏差値 vs 安定性 ---
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.scatter(summary['平均偏差値'], summary['安定性'])
ax2.axvline(summary['平均偏差値'].mean(), linestyle='--')
ax2.axhline(summary['安定性'].mean(), linestyle='--')
for _, r in summary.iterrows(): ax2.text(r['平均偏差値'], r['安定性'], r['馬名'], fontsize=8)
ax2.set_xlabel('平均偏差値')
ax2.set_ylabel('安定性')
st.pyplot(fig2)

# --- 予想タグ ---
tag_map = {1:'◎',2:'〇',3:'▲',4:'☆',5:'△',6:'△'}
pred = combined.reset_index(drop=True).copy()
pred['タグ'] = pred.index.map(lambda i: tag_map.get(i+1,''))
st.subheader('本日の予想タグ')
st.table(pred[['馬名','タグ','平均偏差値']])

# --- ベット設定 ---
scenarios = {'通常':{'単勝':8,'複勝':22,'ワイド':40,'馬連':20,'三連複':0,'三連単':0},
             '余裕':{'単勝':5,'複勝':15,'ワイド':20,'馬連':15,'三連複':25,'三連単':5}}

@st.cache_data
def allocate_budget(budget, perc):
    raw = {k: budget * v / 100 for k, v in perc.items()}
    rnd = {k: int(v // 100) * 100 for k, v in raw.items()}
    diff = budget - sum(rnd.values())
    if diff:
        rnd[max(perc, key=perc.get)] += diff
    return rnd

with st.expander('ベット設定'):
    sc = st.selectbox('シナリオ', list(scenarios.keys()))
    bd = st.number_input('予算 (円)', min_value=1000, step=1000, value=10000)
    al = allocate_budget(bd, scenarios[sc])
    st.write(f"シナリオ：{sc} / 予算：{bd:,}円")
    tbl = pd.DataFrame.from_dict(al, orient='index', columns=['金額']).reset_index()
    tbl.columns = ['券種','金額(円)']
    st.table(tbl)
    detail = st.selectbox('詳細', tbl['券種'])
    names = combined['馬名'].tolist()
    axis = names[0] if names else ''
    amt = al.get(detail, 0)
    if detail in ['単勝','複勝']:
        st.write(f"{detail}：軸馬 {axis} に {amt:,}円")
    else:
        if detail in ['馬連','ワイド']:
            combos = [f"{a}-{b}" for a, b in combinations(names, 2)]
        elif detail == '馬連':
            combos = [f"{axis}->{o}" for o in names if o != axis]
        elif detail == '三連複':
            combos = [f"{axis}-{o1}-{o2}" for o1, o2 in combinations(names[1:], 2)]
        elif detail == '三連単':
            combos = ["->".join(p) for p in permutations(names, 3)]
        else:
            combos = []
        cnt = len(combos)
        if cnt > 0:
            unit = (amt // cnt) // 100 * 100
            rem = amt - unit * cnt
            amounts = [unit + 100 if i < rem // 100 else unit for i in range(cnt)]
            df_det = pd.DataFrame({'組合せ':combos, '金額':amounts})
            st.dataframe(df_det)
        else:
            st.write('対象の買い目がありません')
