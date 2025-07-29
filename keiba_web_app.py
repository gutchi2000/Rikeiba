import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import font_manager

font_manager.fontManager.addfont("ipaexg.ttf")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="ipaexg.ttf").get_name()

st.title("競馬スコア分析アプリ（完成版）")

uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)
cols = ["馬名","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
if any(c not in df.columns for c in cols):
    st.error("必要な列が不足しています")
    st.stop()

df = df[cols].copy()
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
for c in ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df.dropna(subset=cols, inplace=True)

GRADE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,"オープン特別":4,"3勝クラス":3,"2勝クラス":2,"1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

df['raw'] = df.apply(lambda r: GRADE.get(r['クラス名'],1) * (r['頭数'] + 1 - r['確定着順']), axis=1)
df['raw_norm'] = (df['raw'] - GP_MIN) / (GP_MAX * df['頭数'] - GP_MIN)
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
jmax, jmin = df['斤量'].max(), df['斤量'].min()
df['jin_norm'] = (jmax - df['斤量']) / (jmax - jmin)
wmean = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / wmean

df['rank_date'] = df.groupby('馬名')['レース日'].rank(method='first', ascending=False)
df['weight'] = 1 / df['rank_date']
metrics = ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']
for m in metrics:
    mu, sd = df[m].mean(), df[m].std(ddof=1)
    df[f'Z_{m}'] = df[m].apply(lambda x: 0 if sd == 0 else (x - mu) / sd)
wmap = {'Z_raw_norm':8,'Z_up3_norm':2,'Z_odds_norm':1,'Z_jin_norm':1,'Z_wdiff_norm':1}
df['total_z'] = sum(df[k] * v for k, v in wmap.items()) / sum(wmap.values())
z = df['total_z']; zmin, zmax = z.min(), z.max()
df['偏差値'] = 30 + (z - zmin) / (zmax - zmin) * 40

df_avg = df.groupby('馬名').apply(lambda d: np.average(d['偏差値'], weights=d['weight'])).reset_index(name='平均偏差値')
st.subheader('全馬 偏差値一覧')
st.dataframe(df_avg.sort_values('平均偏差値', ascending=False))

df_out = df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
df_out.columns = ['馬名','mean_z','std_z']
cand = df_out.copy()
cand['comp'] = cand['mean_z'] - cand['std_z']
cand = cand.merge(df_avg, on='馬名')
top6 = cand.nlargest(6, 'comp')[['馬名','平均偏差値']]
st.subheader('総合スコア 上位6頭')
st.table(top6)

import seaborn as sns
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.barplot(x='平均偏差値', y='馬名', data=top6, palette=sns.color_palette('hsv', len(top6)), ax=ax1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10,6))
x0 = df_out['mean_z'].mean() + df_out['mean_z'].std(ddof=1)
y0 = df_out['std_z'].mean() + df_out['std_z'].std(ddof=1)
xmin, xmax = df_out['mean_z'].min(), df_out['mean_z'].max()
ymin, ymax = 9, df_out['std_z'].max()
ax2.fill_betweenx([y0, ymax], xmin, x0, alpha=0.3)
ax2.fill_betweenx([ymin, y0], xmin, x0, alpha=0.3)
ax2.fill_betweenx([y0, ymax], x0, xmax, alpha=0.3)
ax2.fill_betweenx([ymin, y0], x0, xmax, alpha=0.3)
ax2.axvline(x0, linestyle='--')
ax2.axhline(y0, linestyle='--')
ax2.scatter(df_out['mean_z'], df_out['std_z'], s=50)
for _, r in df_out.iterrows():
    ax2.text(r['mean_z'], r['std_z'], r['馬名'], fontsize=9)
ax2.plot([xmin, xmax], [ymax, ymin], linestyle=':', color='gray')
ax2.set_xlabel('平均偏差値')
ax2.set_ylabel('安定性')
ax2.set_ylim(9, ymax)
st.pyplot(fig2)

output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    df_avg.to_excel(writer, index=False, sheet_name='偏差値一覧')
st.download_button('偏差値一覧をExcelでダウンロード', data=output.getvalue(), file_name='score_list.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

res_file = st.file_uploader('実際の着順Excelをアップロードしてください', type=['xlsx'], key='result')
if res_file:
    resdf = pd.read_excel(res_file, usecols=['馬名','確定着順']).rename(columns={'確定着順':'着順'})
    merged = top6.merge(resdf, on='馬名', how='left')
    merged['ポイント'] = merged['着順'].apply(lambda x: 10 if x <= 3 else -5)
    st.subheader('予想結果と獲得ポイント')
    st.dataframe(merged[['馬名','ポイント']])
    st.success(f"本日の合計ポイント: {merged['ポイント'].sum()}")

with st.expander('券種別買い目候補と予算配分'):
    horses = top6['馬名'].tolist()
    axis = horses[0]
    others = horses[1:]
    umaren = [f"{axis}-{h}" for h in others]
    wide = umaren.copy()
    first, second, third = horses[0], horses[1:3], horses[3:]
    sanrenpuku = ["-".join(sorted([first, b, c])) for b in second for c in third]
    fixed = horses[:4]
    sanrentan = [f"{fixed[0]}→{o1}→{o2}" for o1 in fixed[1:] for o2 in fixed[1:] if o1 != o2]
    bet_choice = st.radio('馬連 or ワイド', ['馬連','ワイド'])
    types = ['単勝','複勝', bet_choice, '三連複', '三連単']
    total_budget = st.number_input('総ベット予算（円）', min_value=1000, step=1000, value=10000)
    base = total_budget // len(types)
    alloc = {t: (base // 100) * 100 for t in types[:-1]}
    alloc[types[-1]] = ((total_budget - sum(alloc.values())) // 100) * 100
    alloc_rows = []
    for t,amt in alloc.items():
        if t == '単勝':
            amount = (amt * 1/4) // 100 * 100
            alloc_rows.append({'券種':'単勝','組合せ':axis,'金額':int(amount)})
        elif t == '複勝':
            amount = (amt * 3/4) // 100 * 100
            alloc_rows.append({'券種':'複勝','組合せ':axis,'金額':int(amount)})
        elif t == '馬連':
            p = (amt // 5) // 100 * 100
            for combo in umaren:
                alloc_rows.append({'券種':'馬連','組合せ':combo,'金額':int(p)})
        elif t == 'ワイド':
            p = (amt // 5) // 100 * 100
            for combo in wide:
                alloc_rows.append({'券種':'ワイド','組合せ':combo,'金額':int(p)})
        elif t == '三連複':
            p = (amt // len(sanrenpuku)) // 100 * 100
            for combo in sanrenpuku:
                alloc_rows.append({'券種':'三連複','組合せ':combo,'金額':int(p)})
        elif t == '三連単':
            p = (amt // len(sanrentan)) // 100 * 100
            for combo in sanrentan:
                alloc_rows.append({'券種':'三連単','組合せ':combo,'金額':int(p)})
    st.dataframe(pd.DataFrame(alloc_rows))
