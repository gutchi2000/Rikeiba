import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="ipaexg.ttf").get_name()

st.title("競馬予想アプリ（完成版）")

# --- 入力: 成績データ ---
uploaded_file = st.file_uploader("成績データをアップロード (Excel)", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# --- 入力: 馬名／年齢／脚質（テーブル編集） ---
equine_list = df['馬名'].unique().tolist()
equine_df = pd.DataFrame({'馬名': equine_list, '年齢': [5]*len(equine_list), '脚質': ['差し']*len(equine_list)})
edited = st.data_editor(
    equine_df,
    column_config={
        '年齢': st.column_config.NumberColumn("年齢", min_value=1, max_value=10, step=1),
        '脚質': st.column_config.SelectboxColumn("脚質", options=['逃げ','先行','差し','追込'])
    },
    use_container_width=True,
    key="equine_editor"
)
# マッピング
age_map = dict(zip(edited['馬名'], edited['年齢']))
style_map = dict(zip(edited['馬名'], edited['脚質']))

# --- 入力: 血統表 (HTML) ---
html_file = st.file_uploader("血統表をアップロード (HTML)", type=["html"])

# --- 必要列チェック ---
cols = ["馬名","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
if any(c not in df.columns for c in cols):
    st.error(f"必要列が不足しています: {set(cols)-set(df.columns)}")
    st.stop()

# --- 前処理 ---
df = df[cols].copy()
for c in ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
df.dropna(subset=cols, inplace=True)

# --- 血統評価 ---
def eval_pedigree(mn):
    if html_file:
        try:
            tables = pd.read_html(html_file.getvalue())
            ped = tables[0]
            sire = ped.set_index('馬名').get('父馬', {}).get(mn, '')
            if sire in ["サクラバクシンオー","スウェプトオーヴァード"]:
                return 1.2
            if sire in ["マンハッタンカフェ","フジキセキ"]:
                return 1.1
        except Exception:
            return 1.0
    return 1.0

df['pedigree_factor'] = df['馬名'].map(eval_pedigree)

# --- 脚質評価 ---
def style_factor(mn):
    return {'逃げ':1.2,'先行':1.1,'差し':1.0,'追込':0.9}.get(style_map.get(mn,''),1.0)

df['style_factor'] = df['馬名'].map(style_factor)

# --- 年齢評価 ---
def age_factor(a):
    peak = 5
    return 1 + 0.2 * (1 - abs(a - peak) / peak)

df['年齢'] = df['馬名'].map(lambda mn: age_map.get(mn, 5))
df['age_factor'] = df['年齢'].apply(age_factor)

# --- 基本指標計算 ---
GRADE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,"オープン特別":4,"3勝クラス":3,"2勝クラス":2,"1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

df['raw'] = df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1) * df['pedigree_factor']
df['raw_norm'] = (df['raw'] - GP_MIN) / (GP_MAX * df['頭数'] - GP_MIN)
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
jmax, jmin = df['斤量'].max(), df['斤量'].min()
df['jin_norm'] = (jmax - df['斤量']) / (jmax - jmin)
wmean = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / wmean

# --- Zスコア化 ---
metrics = ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = df[m].apply(lambda x: 0 if sd == 0 else (x - mu) / sd)
for m in ['pedigree_factor','age_factor','style_factor']:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = df[m].apply(lambda x: 0 if sd == 0 else (x - mu) / sd)

# --- 合成偏差値化 ---
weights = {'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,'Z_jin_norm':1,'Z_wdiff_norm':1,
           'Z_pedigree_factor':3,'Z_age_factor':2,'Z_style_factor':2}
total_w = sum(weights.values())
df['total_z'] = sum(df[k]*w for k,w in weights.items()) / total_w
# 偏差値計算
z = df['total_z']
zmin, zmax = z.min(), z.max()
df['偏差値'] = 30 + (z - zmin) / (zmax - zmin) * 40
