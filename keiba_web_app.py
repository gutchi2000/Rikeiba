import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import font_manager

# 日本語フォント設定
font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams['font.family'] = font_manager.FontProperties(fname='ipaexg.ttf').get_name()

st.title('競馬予想アプリ（完成版）')

# --- 成績データアップロード ---
uploaded_file = st.file_uploader('成績データをアップロード (Excel)', type=['xlsx'])
if not uploaded_file:
    st.stop()
df = pd.read_excel(uploaded_file)

# --- 馬名／年齢／脚質入力テーブル ---
equine_list = df['馬名'].unique().tolist()
equine_df = pd.DataFrame({
    '馬名': equine_list,
    '年齢': [5] * len(equine_list),
    '脚質': ['差し'] * len(equine_list)
})
edited = st.data_editor(
    equine_df,
    column_config={
        '年齢': st.column_config.NumberColumn('年齢', min_value=1, max_value=10, step=1),
        '脚質': st.column_config.SelectboxColumn('脚質', options=['逃げ','先行','差し','追込'])
    },
    use_container_width=True,
    key='editor'
)
age_map = dict(zip(edited['馬名'], edited['年齢']))
style_map = dict(zip(edited['馬名'], edited['脚質']))

# --- 血統HTMLアップロード ---
html_file = st.file_uploader('血統表をアップロード (HTML)', type=['html'])

# --- 必要列チェック ---
required_cols = ['馬名','レース日','頭数','クラス名','確定着順','上がり3Fタイム',
                 'Ave-3F','馬場状態','斤量','増減','単勝オッズ']
if any(c not in df.columns for c in required_cols):
    st.error(f"不足列: {set(required_cols) - set(df.columns)}")
    st.stop()

# --- 前処理 ---
df = df[required_cols].copy()
for col in ['頭数','確定着順','上がり3Fタイム','Ave-3F','斤量','増減','単勝オッズ']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
df.dropna(subset=required_cols, inplace=True)

# --- 血統評価ファクター ---
def eval_pedigree(mn):
    if html_file:
        try:
            html_bytes = html_file.read()
            tables = pd.read_html(html_bytes)
            ped = tables[0]
            sire = ped.set_index('馬名').get('父馬', {}).get(mn, '')
            if sire in ['サクラバクシンオー','スウェプトオーヴァード']:
                return 1.2
            if sire in ['マンハッタンカフェ','フジキセキ']:
                return 1.1
        except:
            pass
    return 1.0
df['pedigree_factor'] = df['馬名'].map(eval_pedigree)

# --- 脚質評価ファクター ---
def style_factor(mn):
    return {'逃げ':1.2,'先行':1.1,'差し':1.0,'追込':0.9}.get(style_map.get(mn,''),1.0)
df['style_factor'] = df['馬名'].map(style_factor)

# --- 年齢評価ファクター ---
def age_factor(age):
    peak = 5
    return 1 + 0.2 * (1 - abs(age - peak) / peak)
df['年齢'] = df['馬名'].map(lambda m: age_map.get(m,5))
df['age_factor'] = df['年齢'].apply(age_factor)

# --- 基本指標計算 ---
GRADE = {'GⅠ':10,'GⅡ':8,'GⅢ':6,'リステッド':5,'オープン特別':4,
         '3勝クラス':3,'2勝クラス':2,'1勝クラス':1,'新馬':1,'未勝利':1}
GP_MIN, GP_MAX = 1, 10
df['raw'] = df.apply(lambda r: GRADE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['raw'] *= df['pedigree_factor']
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
    df[f'Z_{m}'] = (df[m] - mu) / sd if sd != 0 else 0
for m in ['pedigree_factor','style_factor','age_factor']:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = (df[m] - mu) / sd if sd != 0 else 0

# --- 合成偏差値化 ---
weights = {'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,'Z_jin_norm':1,'Z_wdiff_norm':1,
           'Z_pedigree_factor':3,'Z_age_factor':2,'Z_style_factor':2}
total_w = sum(weights.values())
df['total_z'] = sum(df[k] * w for k, w in weights.items()) / total_w
zmin, zmax = df['total_z'].min(), df['total_z'].max()
df['偏差値'] = 30 + (df['total_z'] - zmin) / (zmax - zmin) * 40

# --- 馬別集計 ---
summary = df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
summary.columns = ['馬名','平均偏差値','安定性']
summary['バランススコア'] = summary['平均偏差値'] - summary['安定性']

# --- 表示 ---
st.subheader('馬別 評価一覧')
st.dataframe(summary.sort_values('バランススコア', ascending=False).reset_index(drop=True))
# 上位抽出
top10 = summary.sort_values('平均偏差値', ascending=False).head(10)
st.subheader('偏差値上位10頭')
st.table(top10[['馬名','平均偏差値']])
combined = top10.sort_values('バランススコア', ascending=False).head(6)
st.subheader('偏差値10頭中の安定&調子上位6頭')
st.table(combined)

# --- 可視化 ---
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x='バランススコア', y='馬名', data=combined, palette='viridis', ax=ax)
st.pyplot(fig)

# --- 散布図: 調子 × 安定性 ---
fig2, ax2 = plt.subplots(figsize=(10,6))
# mean, std
df_out = summary.copy()
x0 = df_out['平均偏差値'].mean()
y0 = df_out['安定性'].mean()
ax2.scatter(df_out['平均偏差値'], df_out['安定性'], color='black')
ax2.axvline(x0, linestyle='--', color='gray')
ax2.axhline(y0, linestyle='--', color='gray')
for _, r in df_out.iterrows():
    ax2.text(r['平均偏差値'], r['安定性'], r['馬名'], fontsize=8)
ax2.set_xlabel('平均偏差値')
ax2.set_ylabel('安定性')
st.pyplot(fig2)

# --- ベット設定 ---
# シナリオ定義
scenarios = {
    '通常（堅め）': {'単勝':8,'複勝':22,'ワイド等':70,'三連複':0,'三連単マルチ':0},
    'ちょい余裕':   {'単勝':6,'複勝':19,'ワイド等':50,'三連複':25,'三連単マルチ':0},
    '余裕（攻め）': {'単勝':5,'複勝':15,'ワイド等':35,'三連複':25,'三連単マルチ':20},
}
# 資金配分関数
@st.cache_data(ttl=600)
def allocate_budget(budget, percents):
    raw = {k: budget * v / 100 for k, v in percents.items()}
    rounded = {k: int(v // 100) * 100 for k, v in raw.items()}
    diff = budget - sum(rounded.values())
    if diff != 0:
        main = max(percents, key=lambda k: percents[k])
        rounded[main] += diff
    return rounded

# --- ベット設定 ---
# シナリオ定義
scenarios = {
    '通常（堅め）': {'単勝':8,'複勝':22,'ワイド等':70,'三連複':0,'三連単マルチ':0},
    'ちょい余裕':   {'単勝':6,'複勝':19,'ワイド等':50,'三連複':25,'三連単マルチ':0},
    '余裕（攻め）': {'単勝':5,'複勝':15,'ワイド等':35,'三連複':25,'三連単マルチ':20},
}
# 資金配分関数
@st.cache_data(ttl=600)
def allocate_budget(budget, percents):
    raw = {k: budget * v / 100 for k, v in percents.items()}
    rounded = {k: int(v // 100) * 100 for k, v in raw.items()}
    diff = budget - sum(rounded.values())
    if diff != 0:
        main = max(percents, key=lambda k: percents[k])
        rounded[main] += diff
    return rounded

with st.expander('ベット設定'):
    # シナリオ選択
    scenario = st.selectbox('資金配分シナリオ', list(scenarios.keys()), key='scenario')
    budget = st.number_input('予算 (円)', min_value=1000, step=1000, value=10000, key='budget')
    # 配分計算
    alloc = allocate_budget(budget, scenarios[scenario])
    st.write(f"**シナリオ：{scenario}**, **予算：{budget:,}円**")
    # 配分結果テーブル
    alloc_df = pd.DataFrame.from_dict(alloc, orient='index', columns=['金額']).reset_index()
    alloc_df.columns = ['券種','金額(円)']
    st.table(alloc_df)
    # 詳細買い目
    detail = st.selectbox('券種別詳細', alloc_df['券種'].tolist(), key='detail')
    amt = alloc.get(detail, 0)
    if detail in ['単勝','複勝']:
        st.write(f"{detail}：軸馬 {combined['馬名'].iat[0]} に {amt:,}円")
    else:
        # 組み合わせ生成
        names = combined['馬名'].tolist()
        if detail in ['馬連','ワイド']:
            combos = [f"{names[0]}-{n}" for n in names[1:]]
        elif detail == '三連複':
            combos = ["-".join(sorted([names[0], b, c])) for b in names[1:3] for c in names[3:]]
        else:  # 三連単
            combos = [f"{names[0]}→{i}→{j}" for i in names[1:4] for j in names[1:4] if i != j]
        if combos:
            per = amt // len(combos) // 100 * 100
            dfb = pd.DataFrame({'券種': detail, '組合せ': combos, '金額': [per]*len(combos)})
            st.dataframe(dfb)
        else:
            st.write('対象の買い目がありません')
