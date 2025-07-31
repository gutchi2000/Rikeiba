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
st.sidebar.subheader('1. 性別の重み')
male_weight    = st.sidebar.number_input('牡の重み', min_value=0.0, value=1.1, step=0.01, format="%.2f")
female_weight  = st.sidebar.number_input('牝の重み', min_value=0.0, value=1.0, step=0.01, format="%.2f")
gelding_weight = st.sidebar.number_input('セの重み', min_value=0.0, value=0.95, step=0.01, format="%.2f")

st.sidebar.subheader('2. 脚質の重み')
nige_weight  = st.sidebar.number_input('逃げの重み', min_value=0.0, value=1.2, step=0.01, format="%.2f")
senko_weight = st.sidebar.number_input('先行の重み', min_value=0.0, value=1.1, step=0.01, format="%.2f")
sashi_weight = st.sidebar.number_input('差しの重み', min_value=0.0, value=1.0, step=0.01, format="%.2f")
ooka_weight = st.sidebar.number_input('追込の重み', min_value=0.0, value=0.9, step=0.01, format="%.2f")

st.sidebar.subheader('3. 四季の重み')
spring_weight = st.sidebar.number_input('春の重み', min_value=0.0, value=1.0, step=0.01, format="%.2f")
summer_weight = st.sidebar.number_input('夏の重み', min_value=0.0, value=1.1, step=0.01, format="%.2f")
autumn_weight = st.sidebar.number_input('秋の重み', min_value=0.0, value=1.0, step=0.01, format="%.2f")
winter_weight = st.sidebar.number_input('冬の重み', min_value=0.0, value=0.95, step=0.01, format="%.2f")

st.sidebar.subheader('4. 斤量の重み')
w_jin = st.sidebar.number_input('Z_斤量正規化 重み', min_value=0.0, value=1.0, step=0.1, format="%.1f")

# --- データアップロード ---
uploaded_file = st.file_uploader('成績&馬情報データをアップロード (Excel)', type=['xlsx'])
if not uploaded_file:
    st.stop()
# シート読み込み
xls = pd.ExcelFile(uploaded_file)
df = xls.parse(sheet_name=0, parse_dates=['レース日'])
stats = xls.parse(sheet_name=1, header=1)
if 'Unnamed: 0' in stats.columns and '馬名' not in stats.columns:
    stats = stats.rename(columns={'Unnamed: 0':'馬名'})
stats = stats[['馬名','性別','年齢']]
df = df.merge(stats, on='馬名', how='left')

# --- ファクター関数 ---
def eval_pedigree(row, ped_df, priority_sires):
    sire = ''
    try:
        sire = ped_df.at[row['馬名'], '父馬']
    except:
        pass
    return 1.2 if sire in priority_sires else 1.0

def style_factor(style):
    return {'逃げ':nige_weight,'先行':senko_weight,'差し':sashi_weight,'追込':ooka_weight}.get(style, 1.0)

def age_factor(age):
    peak = 5
    return 1 + 0.2 * (1 - abs(age - peak) / peak)

def sex_factor(sex):
    return {'牡':male_weight,'牝':female_weight,'セ':gelding_weight}.get(sex, 1.0)

def seasonal_factor(date):
    m = date.month
    if m in [3,4,5]: return spring_weight
    if m in [6,7,8]: return summer_weight
    if m in [9,10,11]: return autumn_weight
    return winter_weight

# 血統表（HTML）アップロード
html_file = st.file_uploader('血統表をアップロード (HTML)', type=['html'])
ped_df = None
if html_file:
    try:
        ped_df = pd.read_html(html_file.read())[0].set_index('馬名')
    except:
        ped_df = None

# 強調種牡馬リスト
ped_input = st.text_area('強調種牡馬リストを入力 (カンマ区切り)', '')
priority_sires = [s.strip() for s in ped_input.split(',') if s.strip()]

# --- 各ファクター適用 ---
df['pedigree_factor'] = df.apply(lambda r: eval_pedigree(r, ped_df, priority_sires), axis=1)
df['style_factor']    = df['脚質'] .apply(style_factor)
df['age_factor']      = df['年齢'] .apply(age_factor)
df['sex_factor']      = df['性別'] .apply(sex_factor)
df['seasonal_factor'] = df['レース日'] .apply(seasonal_factor)

# --- 基本スコア計算 ---
GRADE={'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
       '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
raw = df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['raw'] = raw * df['pedigree_factor'] * df['style_factor'] * df['age_factor'] * df['sex_factor'] * df['seasonal_factor']

# --- 正規化 ---
jmax,jmin=df['斤量'].max(),df['斤量'].min()
df['raw_norm'] = (df['raw'] - 1) / (10 * df['頭数'] - 1)
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
df['jin_norm']  = (jmax - df['斤量']) / (jmax - jmin)
wmean = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / wmean

# --- Zスコア化 ---
metrics=['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm',
         'pedigree_factor','style_factor','age_factor','sex_factor','seasonal_factor']
for m in metrics:
    mu,sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = (df[m] - mu) / sd if sd!=0 else 0

# --- 合成偏差値化 ---
weights={'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,
         'Z_jin_norm':w_jin,'Z_wdiff_norm':1,
         'Z_pedigree_factor':3,'Z_age_factor':2,'Z_style_factor':2,'Z_seasonal_factor':2}
tot_w=sum(weights.values())
df['total_z']=sum(df[k]*w for k,w in weights.items())/tot_w
zmin,zmax=df['total_z'].min(),df['total_z'].max()
df['偏差値']=30+(df['total_z']-zmin)/(zmax-zmin)*40

# --- 馬別集計・表示以下省略（既存コードを継続） ---
