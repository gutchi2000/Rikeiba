import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_path = "ipaexg.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

st.title("競馬スコア分析アプリ（全機能版）")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- データ読み込み & プレビュー ---
df = pd.read_excel(uploaded_file)
st.subheader('データプレビュー')
st.write(df.head())

# --- 必要列チェック ---
required = ["馬名","レース日","頭数","クラス名","確定着順","上がり3Fタイム","Ave-3F","馬場状態","斤量","増減","単勝オッズ"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"必要な列が不足しています: {missing}")
    st.stop()

# --- 列抽出 & 型変換 ---
df = df[required].copy()
# 日付
df['レース日'] = pd.to_datetime(df['レース日'], errors='coerce')
# 数値
num_cols = ["頭数","確定着順","上がり3Fタイム","Ave-3F","斤量","増減","単勝オッズ"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
# 欠損行の削除
df.dropna(subset=["レース日"] + num_cols, inplace=True)

# --- 指標計算 ---
GRADE_SCORE = {"GⅠ":10,"GⅡ":8,"GⅢ":6,"リステッド":5,
               "オープン特別":4,"3勝クラス":3,"2勝クラス":2,
               "1勝クラス":1,"新馬":1,"未勝利":1}
GP_MIN, GP_MAX = 1, 10

# 生スコアと正規化
N = df['頭数']
df['raw'] = df.apply(lambda r: GRADE_SCORE.get(r['クラス名'],1)*(r['頭数']+1-r['確定着順']), axis=1)
df['raw_norm'] = (df['raw'] - GP_MIN) / (GP_MAX * df['頭数'] - GP_MIN)
# 上がり3F
df['up3_norm'] = df['Ave-3F'] / df['上がり3Fタイム']
# オッズ
df['odds_norm'] = 1 / (1 + np.log10(df['単勝オッズ']))
# 斤量（軽いほど高）
jin_max, jin_min = df['斤量'].max(), df['斤量'].min()
df['jin_norm'] = (jin_max - df['斤量']) / (jin_max - jin_min)
# 体重増減
mean_w = df['増減'].abs().mean()
df['wdiff_norm'] = 1 - df['増減'].abs() / mean_w

# --- 直近レース重み付け ---
df['rank_date'] = df.groupby('馬名')['レース日'].rank(ascending=False, method='first')
df['weight'] = 1 / df['rank_date']

# --- Zスコア標準化 ---
metrics = ['raw_norm','up3_norm','odds_norm','jin_norm','wdiff_norm']
for col in metrics:
    mu, sigma = df[col].mean(), df[col].std(ddof=1)
    df[f'Z_{col}'] = df[col].apply(lambda x: 0 if sigma == 0 else (x - mu) / sigma)

# --- 重み付け合成と偏差値化 (スケール拡大) ---
weights = {'Z_raw_norm':8, 'Z_up3_norm':2, 'Z_odds_norm':1, 'Z_jin_norm':1, 'Z_wdiff_norm':1}
total_w = sum(weights.values())
df['total_z'] = sum(df[k] * w for k, w in weights.items()) / total_w
mu_t, sigma_t = df['total_z'].mean(), df['total_z'].std(ddof=1)
scale = 15  # 偏差値のスケールを10→15に増加して分散を広げる
if sigma_t == 0:
    df['偏差値'] = 50
else:
    df['偏差値'] = 50 + scale * (df['total_z'] - mu_t) / sigma_t

# --- 馬別平均偏差値 (加重平均) ---
df_avg = df.groupby('馬名').apply(lambda d: np.average(d['偏差値'], weights=d['weight'])).reset_index()
df_avg.columns = ['馬名','平均偏差値']

# --- 表示: 全馬偏差値一覧 ---
st.subheader('全馬 偏差値一覧')
st.dataframe(df_avg.sort_values('平均偏差値', ascending=False).reset_index(drop=True))

# --- 表示: 上位6頭 （調子×安定性による総合評価） ---
# 散布図用の df_out を定義（mean_z, std_z の算出）
df_out = df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
df_out.columns = ['馬名','mean_z','std_z']
# 調子(mean_z)と安定性(std_z)の差分で総合スコアを計算
candidate = df_out.copy()
# 調子(mean_z)と安定性(std_z)の差分で総合スコアを計算
candidate = df_out.copy()
candidate['composite'] = candidate['mean_z'] - candidate['std_z']
# 平均偏差値列を結合
candidate = candidate.merge(df_avg, on='馬名')
# composite 上位6頭を取得
top6 = candidate.nlargest(6, 'composite')[[ '馬名','平均偏差値','composite' ]].reset_index(drop=True)
st.subheader('総合スコア 上位6頭')
st.write(top6[['馬名','平均偏差値']])

# タグ付け
# composite をもとにランク付け
tag_map = {1: '◎', 2: '〇', 3: '▲', 4: '☆', 5: '△', 6: '△'}
top6['タグ'] = top6.index.map(lambda i: tag_map.get(i+1, ''))

# --- 棒グラフ (タグ別色分け) ---
fig1, ax1 = plt.subplots(figsize=(8,5))
import seaborn as sns
palette = {'◎':'#e31a1c','〇':'#1f78b4','▲':'#33a02c','☆':'#ff7f00','△':'#6a3d9a'}
sns.barplot(x='平均偏差値', y='馬名', hue='タグ', data=top6, dodge=False, palette=palette, ax=ax1)
ax1.set_xlabel('平均偏差値')
ax1.set_ylabel('馬名')
ax1.legend(title='タグ')
st.pyplot(fig1)

# --- 散布図: 調子×安定性 ---: 調子×安定性 ---
df_out = df.groupby('馬名')['偏差値'].agg(['mean','std']).reset_index()
df_out.columns = ['馬名','mean_z','std_z']
fig2, ax2 = plt.subplots(figsize=(10,6))
# 基準線 (mean+1σ)
x0 = df_out['mean_z'].mean() + df_out['mean_z'].std(ddof=1)
y0 = df_out['std_z'].mean() + df_out['std_z'].std(ddof=1)
# 軸範囲
xmin, xmax = df_out['mean_z'].min(), df_out['mean_z'].max()
ymin, ymax = df_out['std_z'].min(), df_out['std_z'].max()
# 背景4象限塗り分け
ax2.fill_betweenx([y0, ymax], xmin, x0, alpha=0.3)
ax2.fill_betweenx([ymin, y0], xmin, x0, alpha=0.3)
ax2.fill_betweenx([y0, ymax], x0, xmax, alpha=0.3)
ax2.fill_betweenx([ymin, y0], x0, xmax, alpha=0.3)
# 基準線描画
ax2.axvline(x0, linestyle='--')
ax2.axhline(y0, linestyle='--')
# 散布プロット
ax2.scatter(df_out['mean_z'], df_out['std_z'], s=50)
for _, r in df_out.iterrows():
    ax2.text(r['mean_z'], r['std_z'], r['馬名'], fontsize=9)
# ラベル配置
ax2.text((xmin+x0)/2, (y0+ymax)/2, '軽視', ha='center', va='center')
ax2.text((xmin+x0)/2, (ymin+y0)/2, '堅軸', ha='center', va='center')
ax2.text((x0+xmax)/2, (y0+ymax)/2, '抑え穴', ha='center', va='center')
ax2.text((x0+xmax)/2, (ymin+y0)/2, '本命', ha='center', va='center')
ax2.set_xlabel('平均偏差値')
ax2.set_ylabel('安定性 (偏差値標準偏差)')

# --- 参考線: 対角線 (負の相関目安) ---
# 左上 -> 右下 のライン
ax2.plot([xmin, xmax], [ymax, ymin], linestyle=':', linewidth=1, color='gray', label='対角線')
ax2.legend()

st.pyplot(fig2)

# --- Excel ダウンロード ---
download_df = df_avg.sort_values('平均偏差値', ascending=False)
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    download_df.to_excel(writer, index=False, sheet_name='偏差値一覧')
processed = output.getvalue()
st.download_button('偏差値一覧をExcelでダウンロード', data=processed,
                   file_name='score_list.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- 予想結果履歴CSV ダウンロード ---
res_file = st.file_uploader('実際の着順Excelをアップロードしてください', type=['xlsx'], key='result')
if res_file:
    res_df = pd.read_excel(res_file, usecols=["馬名","確定着順"]).rename(columns={"確定着順":"着順"})
    merged = top6.merge(res_df, on='馬名', how='left')
    merged['ポイント'] = merged['着順'].apply(lambda x: 10 if x<=3 else -5)
    st.subheader('予想結果と獲得ポイント')
    st.write(merged[['馬名','タグ','着順','ポイント']])
    total = merged['ポイント'].sum()
    st.success(f'本日の合計ポイント: {total}')
    csv = merged.to_csv(index=False).encode('utf-8-sig')
    st.download_button('予想結果CSVをダウンロード', data=csv,
                       file_name='prediction_history.csv', mime='text/csv')

# --- 予算からのベット配分推奨（100円単位） ---
st.subheader('予算からの推奨ベット配分')
budget = st.number_input('総ベット予算（円）を入力してください', min_value=1000, step=1000, value=10000)
scores = top6['平均偏差値']
ratios = scores / scores.sum()
bet_amounts = (ratios * budget / 100).round(0).astype(int) * 100
recommend = pd.DataFrame({'馬名': top6['馬名'], 'タグ': top6['タグ'], '推奨ベット額': bet_amounts})
st.write(recommend)
st.write(f"合計推奨ベット額: {bet_amounts.sum():,}円")

# --- 券種別買い目出力 & 予算内訳配分 ---
with st.expander('券種別買い目候補と予算配分'):
    st.write('◎→〇▲☆△を軸にした各券種サンプル買い目')
    # 上位6頭リスト作成
    horses = list(top6['馬名'])
    axis = horses[0]
    others = horses[1:]
    # 各券種買い目生成
    umaren = [f"{axis}-{h}" for h in others]
    wide = umaren.copy()
    first, second, third = horses[0], horses[1:3], horses[3:]
    sanrenpuku = ["-".join(sorted([first, b, c])) for b in second for c in third]
    fixed = horses[:4]
    sanrentan = [f"{fixed[0]}→{o1}→{o2}" for o1 in fixed[1:] for o2 in fixed[1:] if o1 != o2]

    # 予算と使用券種判定
    total_budget = st.number_input('券種合計ベット予算（円）を入力', min_value=1000, step=1000, value=10000)
    if total_budget <= 10000:
        types = ['単勝・複勝', '馬連', '三連複', '三連単']
    else:
        types = ['単勝・複勝', 'ワイド', '三連複', '三連単']

    # 各券種表示
    for t in types:
        if t == '単勝・複勝':
            st.markdown(f'**{t}** → {axis}')
        elif t == '馬連':
            st.markdown('**馬連**（軸：◎ → 相手：〇▲☆△）')
            st.write(umaren)
        elif t == 'ワイド':
            st.markdown('**ワイド**（軸：◎ → 流し）')
            st.write(wide)
        elif t == '三連複':
            st.markdown('**三連複フォーメーション**')
            st.write(sanrenpuku)
        elif t == '三連単':
            st.markdown('**三連単マルチ**（軸：◎ → 相手：〇▲☆）')
            st.write(sanrentan)

    # 予算配分（均等切り捨て）
    n = len(types)
    rates = {t: 1/n for t in types}
    alloc = {}
    remaining = total_budget
    for i, t in enumerate(types):
        if i == len(types) - 1:
            alloc_amt = remaining // 100 * 100
        else:
            alloc_amt = int(total_budget * rates[t]) // 100 * 100
            remaining -= alloc_amt
        alloc[t] = alloc_amt

    # 一買い目あたり金額算出
    alloc_per = {}
    for t, amt in alloc.items():
        if t == '単勝・複勝':
            win_amt = int(round(amt * 1/4) // 100 * 100)
            place_amt = int(round(amt * 3/4) // 100 * 100)
            alloc_per['単勝'] = win_amt
            alloc_per['複勝'] = place_amt
        elif t == '馬連':
            for combo in umaren:
                alloc_per[f'馬連: {combo}'] = amt // len(umaren)
        elif t == 'ワイド':
            for combo in wide:
                alloc_per[f'ワイド: {combo}'] = amt // len(wide)
        elif t == '三連複':
            for combo in sanrenpuku:
                alloc_per[f'三連複: {combo}'] = amt // len(sanrenpuku)
        elif t == '三連単':
            for combo in sanrentan:
                alloc_per[f'三連単: {combo}'] = amt // len(sanrentan)

    # 表示
    st.write('**券種別予算配分**')
    st.write(pd.DataFrame([{t: v for t, v in alloc.items()}]).T.rename(columns={0:'配分額(円)'}))
    st.write('**一買い目あたり推奨金額**')
    st.dataframe(pd.DataFrame([{k: v for k, v in alloc_per.items()}]).T.rename(columns={0:'推奨金額(円)'}))
    st.caption('※100円単位で切り捨てています。')
