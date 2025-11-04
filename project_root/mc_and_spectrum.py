# -*- coding: utf-8 -*-
"""
mc_and_spectrum.py
- 競馬スコアから勝率/連対率/複勝率を推定する高速モンテカルロ
- 区間ペースなど等間隔系列からスペクトル適合係数とZ値を計算

依存: numpy, pandas
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd


# ===============================
# 1) モンテカルロ（高速・安定版）
# ===============================
def run_mc_prob(
    df: pd.DataFrame,
    score_col: str = "final_score",
    n_sims: int = 200_000,
    *,
    dist: str = "gumbel",          # "gumbel"（推奨, PL系に近い）or "normal"
    noise_scale: float = 1.0,      # 1.0 を基準（スコアのスケールに合わせて微調整）
    seed: Optional[int] = 42,
    chunk_size: int = 50_000,      # メモリ対策。18頭×5万なら余裕
    return_counts: bool = False,   # Trueなら回数も返す
) -> pd.DataFrame:
    """
    df[score_col] を基に n_sims 回の擬似レースを行い、win/連対/複勝/Top4 を推定。
    - タイブレーク: 微小一様乱数で安定化
    - ベクトル化: チャンクごとに一括演算で高速
    - dist: "gumbel" はGumbel-Max trickに近く、順位選好と相性が良い

    返り値: df と同じ行順の DataFrame（%は 0-100 の実数）
    """
    assert score_col in df.columns, f"{score_col} がありません"
    scores = df[score_col].to_numpy(dtype=np.float64)
    n = len(scores)
    if n == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)

    win_ct  = np.zeros(n, dtype=np.int64)
    pl2_ct  = np.zeros(n, dtype=np.int64)  # 2着以内（連対）
    pl3_ct  = np.zeros(n, dtype=np.int64)  # 3着以内（複勝）
    top4_ct = np.zeros(n, dtype=np.int64)  # 4着以内（参考）

    sims_done = 0
    while sims_done < n_sims:
        m = min(chunk_size, n_sims - sims_done)

        # ノイズ生成（分布切替）
        if dist == "gumbel":
            noise = rng.gumbel(loc=0.0, scale=noise_scale, size=(m, n))
        elif dist == "normal":
            noise = rng.normal(loc=0.0, scale=noise_scale, size=(m, n))
        else:
            raise ValueError("dist は 'gumbel' か 'normal' を指定してください")

        # タイブレーク用の極小ノイズ（順位安定）
        eps = rng.uniform(0.0, 1e-9, size=(m, n))

        sample = scores[None, :] + noise + eps  # (m, n)

        # 高い順に並べる（stableで僅差の順序を安定化）
        order = np.argsort(-sample, axis=1, kind="stable")  # (m, n)

        # 各種カウント（np.add.at で一括加算）
        top1 = order[:, 0]
        np.add.at(win_ct, top1, 1)

        top2 = order[:, :2].reshape(-1)
        np.add.at(pl2_ct, top2, 1)

        top3 = order[:, :3].reshape(-1)
        np.add.at(pl3_ct, top3, 1)

        top4 = order[:, :4].reshape(-1)
        np.add.at(top4_ct, top4, 1)

        sims_done += m

    # 割合へ（%表記）
    to_pct = lambda x: (x.astype(np.float64) / n_sims) * 100.0
    out = pd.DataFrame({
        "勝率%_MC":   to_pct(win_ct),
        "連対率%_MC": to_pct(pl2_ct),
        "複勝率%_MC": to_pct(pl3_ct),
        "Top4%_MC":   to_pct(top4_ct),
    }, index=df.index)

    if return_counts:
        out[ "win_n_MC"]  = win_ct
        out["pl2_n_MC"]   = pl2_ct
        out["pl3_n_MC"]   = pl3_ct
        out["top4_n_MC"]  = top4_ct

    return out


# ===================================
# 2) スペクトル適合係数（“確実に動く”版）
# ===================================
def _hann(n: int) -> np.ndarray:
    # SciPy不要のHann窓
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (n - 1)))


def spectrum_fit(
    signal: np.ndarray,
    fs: float,
    *,
    band: Tuple[float, float] = (0.3, 4.0),    # 有効帯域（Hz相当）：ペース系列なら 0.3–2Hz 程度が無難
    peak_frac: float = 0.5,                    # FWHM相当閾値（ピークの何割で幅を測るか）
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    等間隔サンプリングの系列 `signal`（例: 200m区間平均速度を線形補間して等間隔にした列）から、
    rFFTでパワースペクトルを計算し、以下を返す:
      - peak_freq: バンド内の最大ピーク周波数
      - band_energy_ratio: バンド内エネルギー / 全エネルギー（0-1）
      - sharpness: ピークの鋭さ（~Q値）。大きいほど周波数集中。
      - specfit: 適合係数（0-1）。band比と鋭さの両方を反映した指標。

    備考:
    - “まずは確実に動く”ことを優先した素直な実装。
    - 入力は等間隔。等間隔でない区間ラップしかない場合は、先に等間隔へ補間してから渡すこと。
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 8:
        return dict(peak_freq=0.0, band_energy_ratio=0.0, sharpness=0.0, specfit=0.0)

    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    w = _hann(n)
    xw = x * w

    # 実数FFT
    X = np.fft.rfft(xw)
    P = (X.real**2 + X.imag**2).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    total_power = P.sum() + eps

    # バンド抽出
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return dict(peak_freq=0.0, band_energy_ratio=0.0, sharpness=0.0, specfit=0.0)

    band_power = P[mask].sum()
    band_energy_ratio = float(band_power / total_power)

    # バンド内ピーク
    idx_local = np.argmax(P[mask])
    peak_power = P[mask][idx_local] + eps
    peak_freq = float(freqs[mask][idx_local])

    # ピークの幅（FWHM類似）→ 鋭さ（Q相当） = f0 / bw
    thr = float(peak_frac * peak_power)
    # 左右に探索
    band_indices = np.where(mask)[0]
    center = band_indices[idx_local]

    left = center
    while left > band_indices[0] and P[left] >= thr:
        left -= 1
    right = center
    while right < band_indices[-1] and P[right] >= thr:
        right += 1

    bw = (freqs[right] - freqs[left]) if right > left else (freqs[1] - freqs[0])
    bw = max(bw, 1e-9)
    sharpness = float(peak_freq / bw)

    # 適合係数（0-1）: 帯域比と鋭さをロジスティックで合成
    #   - band_energy_ratio は 0-1
    #   - sharpness は 1以上が多いのでスケーリング
    s_norm = np.tanh(sharpness / 8.0)  # 0-1 相当に圧縮
    specfit = 0.6 * band_energy_ratio + 0.4 * s_norm
    specfit = float(np.clip(specfit, 0.0, 1.0))

    return dict(
        peak_freq=peak_freq,
        band_energy_ratio=band_energy_ratio,
        sharpness=sharpness,
        specfit=specfit,
    )


def specfit_to_z(specfit: float, mu: float = 0.35, sigma: float = 0.10) -> float:
    """
    適合係数（0-1）を Z に線形マッピング。
    - mu, sigma はベース母集団の平均と標準偏差の想定（必要に応じて更新）
    """
    z = (specfit - mu) / (sigma if sigma > 1e-9 else 1e-9)
    # 極端値を暴走させない
    return float(np.clip(z, -4.0, 4.0))


# ===============================
# 3) 使い方の最小例（コメント）
# ===============================
"""
# --- Monte Carlo の使い方 ---
df = pd.DataFrame({
    "馬名": ["A","B","C","D"],
    "final_score": [1.2, 0.9, 0.1, -0.3],
})
prob = run_mc_prob(df, score_col="final_score", n_sims=200_000,
                   dist="gumbel", noise_scale=1.0, seed=123)
df_prob = pd.concat([df, prob], axis=1)
# df_prob に 勝率%_MC / 連対率%_MC / 複勝率%_MC / Top4%_MC が追加

# --- Spectrum の使い方 ---
# 例: 等間隔にサンプリングされたペース系列（速度など）
fs = 5.0  # 5 Hz 等（等間隔なら何でもOK）
series = np.array([10.0, 10.1, 10.3, 10.2, 10.1, 10.0, 9.9, 9.8, 9.9, 10.0], dtype=float)
sp = spectrum_fit(series, fs=fs, band=(0.3, 3.0))  # dict を返す
specfit_z = specfit_to_z(sp["specfit"], mu=0.35, sigma=0.10)
"""

# ここまで
