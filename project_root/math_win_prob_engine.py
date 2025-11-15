# -*- coding: utf-8 -*-
"""
math_win_prob_engine.py

厩舎のエクセルから「各馬の特徴量 → 多項ロジットで勝率・連対率・複勝率」を出す
Codex / Code Interpreter 向けの単体スクリプト。

前提:
- 入力: 1レース分のデータを含む DataFrame or CSV/Excel
- 出力: 各馬の P(win), P(top2), P(top3) と logit スコアなど

学術的背景:
- McFadden (1974) の Conditional Logit / Multinomial Logit に基づく
- 線形効用 U_ih = βᵀ x_ih
- P_ih = exp(U_ih) / Σ_j exp(U_jh)

使い方(例):
- まず手元の特徴量で学習用データを作って β を推定しておき、
  ここでは「既に学習済みの係数 β を読み込んで推論だけ行う」想定。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# =========================
# データ構造とヘルパー
# =========================

@dataclass
class HorseFeatures:
    """1頭分の特徴ベクトル"""
    horse_id: str
    race_id: str
    features: Dict[str, float]  # 例: {"speed_index": 1.2, "pace_fit": -0.3, ...}


@dataclass
class LogitModel:
    """
    多項ロジット用のパラメータを保持するクラス。

    beta: 特徴量名 → 係数
    intercept: 切片 (任意)
    """
    beta: Dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0

    def utility(self, x: Dict[str, float]) -> float:
        """
        U = βᵀ x + intercept
        欠損している特徴は 0 扱い。
        """
        s = self.intercept
        for name, coef in self.beta.items():
            s += coef * float(x.get(name, 0.0))
        return s

    def utilities_for_race(self, horses: List[HorseFeatures]) -> np.ndarray:
        """
        あるレースに出走する全頭について効用 U を計算して np.array で返す。
        """
        return np.array([self.utility(h.features) for h in horses], dtype=float)

    def probs_for_race(self, horses: List[HorseFeatures]) -> np.ndarray:
        """
        あるレースに出走する全頭について P(win) を計算して np.array で返す。
        P_i = exp(U_i) / Σ_j exp(U_j)
        """
        u = self.utilities_for_race(horses)
        u_max = np.max(u)
        exps = np.exp(u - u_max)
        probs = exps / np.sum(exps)
        return probs


# =========================
# Top-k の確率近似
# =========================

def approx_topk_probs_from_win(probs: np.ndarray, k: int) -> np.ndarray:
    """
    P(win) だけから P(Top k) を近似する簡易関数。
    厳密には順位分布をモデリングすべきだが、ここでは実務向け近似。

    近似アイデア:
    - P(top2) ≈ P(win) + (1 - P(win)) * P(win) / (1 - P(win_max))
    など色々あるが、ここでは「Softmax を温度補正して再度かける」形にする。
    """
    if k <= 1:
        return probs.copy()

    # 温度パラメータで「分布をなだらかに」して足し合わせる簡易近似
    # k が大きいほど温度を上げる (分布を平坦にする)
    temperature = 1.0 + 0.6 * (k - 1)  # 適当な経験的スケーリング
    log_p = np.log(probs + 1e-12)
    log_p_scaled = log_p / temperature
    log_p_scaled -= np.max(log_p_scaled)
    p_soft = np.exp(log_p_scaled)
    p_soft /= p_soft.sum()

    # 「元の P(win) と平坦化した分布の合成」で top-k を近似
    # 重み w を調整してもよい
    w = min(0.6 + 0.1 * (k - 1), 0.95)
    approx = probs + w * p_soft
    approx /= approx.sum()  # 正規化
    return approx


# =========================
# メイン: DataFrame ベースの推論パイプライン
# =========================

@dataclass
class WinProbEngineConfig:
    """
    推論エンジンの設定
    """
    race_id_col: str = "race_id"
    horse_id_col: str = "horse_id"
    feature_cols: Optional[List[str]] = None  # None の場合は自動推定
    # 3つの異なる β を用意して、勝率/連対率/複勝率に対応させてもよい
    win_model: LogitModel = field(default_factory=LogitModel)


class WinProbEngine:
    """
    多項ロジット理論に基づき、レース内の相対効用 U_i から勝率などを算出するエンジン。
    """

    def __init__(self, config: WinProbEngineConfig):
        self.cfg = config

    def _detect_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """
        race_id, horse_id を除いた数値列を自動的に特徴量として拾う。
        """
        cols = []
        for col in df.columns:
            if col in (self.cfg.race_id_col, self.cfg.horse_id_col):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
        return cols

    def _df_to_horse_list(self, df_race: pd.DataFrame) -> List[HorseFeatures]:
        """
        1レース分の DataFrame から HorseFeatures のリストに変換。
        """
        feature_cols = (
            self.cfg.feature_cols
            if self.cfg.feature_cols is not None
            else self._detect_feature_cols(df_race)
        )
        horses = []
        for _, row in df_race.iterrows():
            feat = {c: float(row[c]) for c in feature_cols}
            horses.append(
                HorseFeatures(
                    horse_id=str(row[self.cfg.horse_id_col]),
                    race_id=str(row[self.cfg.race_id_col]),
                    features=feat,
                )
            )
        return horses

    def infer_for_race(self, df_race: pd.DataFrame) -> pd.DataFrame:
        """
        1レース分 df_race について、P(win), P(top2), P(top3) を付与した DataFrame を返す。
        """
        horses = self._df_to_horse_list(df_race)
        p_win = self.cfg.win_model.probs_for_race(horses)
        p_top2 = approx_topk_probs_from_win(p_win, k=2)
        p_top3 = approx_topk_probs_from_win(p_win, k=3)

        df_out = df_race.copy()
        df_out["prob_win"] = p_win
        df_out["prob_top2"] = p_top2
        df_out["prob_top3"] = p_top3

        # 対数オッズなどのスコアもつけておくと後で使いやすい
        df_out["logit_win"] = np.log(p_win + 1e-12) - np.log(
            1.0 - p_win + 1e-12
        )

        # 順位ソート用の rank
        df_out["rank_by_prob_win"] = df_out["prob_win"].rank(
            ascending=False, method="min"
        )

        return df_out

    def infer_for_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全レース分 df について groupby(race_id) で処理。
        """
        results = []
        for race_id, df_race in df.groupby(self.cfg.race_id_col):
            df_res = self.infer_for_race(df_race)
            results.append(df_res)
        return pd.concat(results, ignore_index=True)


# =========================
# Excel / CSV I/O ヘルパー
# =========================

def load_race_data_from_excel(
    path: str,
    sheet_name: Optional[str] = None,
    race_id_col: str = "race_id",
    horse_id_col: str = "horse_id",
) -> pd.DataFrame:
    """
    Excel からレースデータを読み込む簡易ヘルパ。
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    if race_id_col not in df.columns:
        df[race_id_col] = "R1"
    if horse_id_col not in df.columns:
        df[horse_id_col] = [f"H{i+1}" for i in range(len(df))]
    return df


def save_with_probs_to_excel(df: pd.DataFrame, path: str) -> None:
    """
    予測結果付き DataFrame を Excel に書き出す。
    """
    df.to_excel(path, index=False)


# =========================
# サンプル用: 学習済み β の読み込み
# =========================

def load_beta_from_json(path: str) -> Dict[str, float]:
    """
    特徴量名 → 係数 β を JSON から読むヘルパ。
    例:
    {
        "speed_index": 1.2,
        "pace_fit": 0.8,
        "jockey_score": 0.3,
        "trainer_score": 0.2
    }
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        beta = json.load(f)
    return {k: float(v) for k, v in beta.items()}


# =========================
# スクリプトとして使う場合の入口
# =========================

def main():
    """
    ここは Codex / CLI から実行するための例。
    パスなどは適宜書き換え。
    """
    # --- 1. データ読み込み ---
    # 例: エリザベス女王杯用の Excel を読む
    input_excel_path = "エリザベス女王.xlsx"  # ←手元のファイル名に合わせて変更
    df = load_race_data_from_excel(input_excel_path, sheet_name=0)

    # --- 2. β を JSON or ハードコードから読み込み ---
    # 例: 事前に学習して保存しておいた係数
    beta_json_path = "win_model_beta.json"  # ←任意のパス
    beta = load_beta_from_json(beta_json_path)
    win_model = LogitModel(beta=beta, intercept=0.0)

    # 特徴量列を明示したい場合はここで指定
    # feature_cols = ["FinalZ", "AR", "local_index", "jockey_score", ...]
    feature_cols = None

    cfg = WinProbEngineConfig(
        race_id_col="race_id",
        horse_id_col="horse_id",
        feature_cols=feature_cols,
        win_model=win_model,
    )

    engine = WinProbEngine(cfg)

    # --- 3. 推論実行 ---
    df_out = engine.infer_for_all(df)

    # --- 4. 結果を Excel に保存 ---
    output_excel_path = "エリザベス女王_with_probs.xlsx"
    save_with_probs_to_excel(df_out, output_excel_path)

    # --- 5. コンソールにも上位馬だけ出す ---
    print(df_out.sort_values("prob_win", ascending=False)[
        ["race_id", "horse_id", "prob_win", "prob_top2", "prob_top3", "rank_by_prob_win"]
    ])


if __name__ == "__main__":
    main()
