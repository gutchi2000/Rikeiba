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

# --- 入力 ---
uploaded_file = st.file_uploader("成績データをアップロード (Excel)", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# --- 馬名／年齢／脚質入力（Excelライク） ---
equine_list = df['馬名'].unique().tolist()
equine_df = pd.DataFrame({'馬名': equine_list, '年齢': ['']*len(equine_list), '脚質': ['']*len(equine_list)})
edited = st.data_editor(
    equine_df,
    column_config={
        '年齢': st.column_config.NumberColumn("年齢", help="1歳～10歳を入力", min_value=1, max_value=10, step=1),
        '脚質': st.column_config.SelectboxColumn("脚質", help="脚質を選択", options=['逃げ','先行','差し','追込'])
    },
    use_container_width=True,
    key="equine_editor"
)
# 編集結果をマッピング
age_map = dict(zip(edited['馬名'], edited['年齢']))
style_map = dict(zip(edited['馬名'], edited['脚質']))

# --- 血統表入力 ---
html_file = st.file_uploader("血統表をアップロード (HTML)", type=["html"])
# 必要列チェック
cols = ["馬名","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
