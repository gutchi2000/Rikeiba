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

# 分割
import pandas as pd as pd_funcs

def split_frame(x):
    parts = str(x).split('-', 2)
    if len(parts)==3 and parts[0].isdigit() and parts[1].isdigit():
        return pd.Series({'枠':int(parts[0]), '番':int(parts[1]), '馬名':parts[2].strip()})
    else:
        return pd.Series({'枠':1, '番':np.nan, '馬名':str(x).strip()})

splits = stats['馬名'].apply(split_frame)
stats = pd.concat([stats.drop(columns='馬名'), splits], axis=1)

df = df.merge(stats[['枠','番','馬名','性別','年齢']], on='馬名', how='left')

# (以下省略：同じ実装が続きます)

# 最後のテーブル表示
st.table(top6[['印','馬名','偏差値','RawBase偏差値','バランス']])(top6[['印','馬名','偏差値','RawBase偏差値','バランス']])

