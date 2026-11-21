#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""干渉分光スペクトルのフーリエ変換解析ツール.

干渉分光法で得られた「波長 (nm) 対 反射率」の干渉スペクトルをフーリエ変換し、
干渉縞の周期成分を評価する。変換の独立変数は 2 通りから選べる。

    --domain wavelength : 波長 λ [nm] に対する FFT
                          共役軸は「100 nm 当たりの波数 (波数/100nm)」
    --domain wavenumber : 波数 1/λ [1/nm] に対する FFT
                          共役軸は「光路差 OPD [nm]」= 2nd
    --domain both       : 両方を実行

主な機能
    1. 波長帯域 (例: 200-400, 300-600) をユーザーが選択してフーリエ変換
    2. 窓関数をユーザーが選択 (矩形/ハン/ハミング/ブラックマン/カイザー等)
    3. B 列以降の全条件列について、振幅最大の位置とその振幅値を記録
    4. サンプリング間隔からナイキスト限界を導出
    5. 横軸=条件 (列名の数値)、縦軸=振幅最大の位置 の散布図を描画
    6. 条件 対 振幅最大位置 の線形回帰 (直線の式・決定係数を図中に記載)
    7. 散布図にナイキスト限界の破線を引き、回帰直線との交点 (ナイキスト限界に
       達する条件) を算出・記録
    8. FFT ピークの精密化 (--fit grid / lm)。FFT 結果の近傍をグリッドサーチし、
       その最良点を初期値に Levenberg-Marquardt 法で非線形回帰する
       (M. Quinten, SN Applied Sciences 1, 823 (2019) 第 5 章の手法)

使用ライブラリ: 標準ライブラリ, numpy, pandas, matplotlib, seaborn, scikit-learn

使用例
    python ir_ft_analysis.py --input dataset.csv --band 300-600 --window hann
    python ir_ft_analysis.py -i dataset_film.csv -b 700-1000 --domain both
    python ir_ft_analysis.py                      # 引数なしなら対話モードで起動
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

#: 波長軸 FFT の共役軸の単位。1 nm 当たりの波数に本係数を掛けて「波数/100nm」とする。
WAVENUMBER_UNIT_NM = 100.0

#: 波長軸が等間隔とみなせるかどうかの判定に使う相対許容誤差。
UNIFORM_GRID_RTOL = 1e-3

#: --input 省略時に読み込む既定の入力ファイル。
DEFAULT_INPUT = "dataset.csv"


# ---------------------------------------------------------------------------
# 変換ドメイン (何を独立変数として FFT するか)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainInfo:
    """FFT の独立変数と共役軸の単位・表記をまとめたもの."""

    key: str                  # "wavelength" / "wavenumber"
    suffix: str               # 出力ファイル名の識別子
    jp_domain: str            # 独立変数の日本語表記
    jp_step: str              # サンプリング間隔の日本語表記
    jp_freq: str              # 共役軸 (FFT 出力) の日本語表記
    jp_freq_unit: str         # 共役軸の単位 (日本語)
    sample_axis_label: str    # 生波形プロットの横軸ラベル (英語)
    freq_axis_label: str      # フーリエスペクトルの横軸ラベル (英語)
    freq_short_en: str        # 図タイトル等で使う短い英語名
    freq_unit_en: str         # 共役軸の単位 (英語)
    unit_scale: float         # rfftfreq の値に掛ける係数
    peak_col: str             # CSV の列名 (ピーク位置)
    nyquist_col: str          # CSV の列名 (ナイキスト限界)
    freq_col: str             # CSV の列名 (共役軸)


DOMAINS: dict[str, DomainInfo] = {
    "wavelength": DomainInfo(
        key="wavelength",
        suffix="lambda",
        jp_domain="波長 λ [nm]",
        jp_step="Δλ",
        jp_freq="波数",
        jp_freq_unit="波数/100nm",
        sample_axis_label="Wavelength (nm)",
        freq_axis_label="Wavenumber (cycles / 100 nm)",
        freq_short_en="wavenumber",
        freq_unit_en="cycles / 100 nm",
        unit_scale=WAVENUMBER_UNIT_NM,
        peak_col="peak_wavenumber_per_100nm",
        nyquist_col="nyquist_per_100nm",
        freq_col="wavenumber_per_100nm",
    ),
    "wavenumber": DomainInfo(
        key="wavenumber",
        suffix="sigma",
        jp_domain="波数 1/λ [1/nm]",
        jp_step="Δ(1/λ)",
        jp_freq="光路差 OPD",
        jp_freq_unit="nm",
        sample_axis_label="Wavenumber 1/$\\lambda$ (1/nm)",
        freq_axis_label="Optical path difference (nm)",
        freq_short_en="OPD",
        freq_unit_en="nm",
        unit_scale=1.0,
        peak_col="peak_opd_nm",
        nyquist_col="nyquist_opd_nm",
        freq_col="opd_nm",
    ),
}

#: ドメイン名のエイリアス
DOMAIN_ALIASES = {
    "lambda": "wavelength", "wl": "wavelength", "l": "wavelength",
    "sigma": "wavenumber", "wn": "wavenumber", "k": "wavenumber",
    "opd": "wavenumber", "1/l": "wavenumber",
}


def resolve_domain(name: str) -> DomainInfo:
    """ドメイン名 (エイリアス可) を DomainInfo に解決する."""
    key = str(name).strip().lower()
    key = DOMAIN_ALIASES.get(key, key)
    if key not in DOMAINS:
        raise ValueError(
            f"未知のドメイン: {name!r}. 選択肢: {', '.join(DOMAINS)}, both")
    return DOMAINS[key]


# ---------------------------------------------------------------------------
# 窓関数
# ---------------------------------------------------------------------------

def _cosine_sum_window(n: int, coeffs: Sequence[float]) -> np.ndarray:
    """余弦和 (cosine-sum) 型の窓関数を生成する (対称形)."""
    if n == 1:
        return np.ones(1)
    x = 2.0 * np.pi * np.arange(n) / (n - 1)
    w = np.zeros(n)
    for i, c in enumerate(coeffs):
        w += ((-1) ** i) * c * np.cos(i * x)
    return w


#: 選択可能な窓関数。key: CLI 名, value: (生成関数, 説明)
WINDOW_FUNCTIONS: dict[str, tuple[Callable[..., np.ndarray], str]] = {
    "rectangular": (lambda n, **kw: np.ones(n),
                    "矩形窓 (窓なし)。周波数分解能は最良だが漏れが大きい"),
    "hann": (lambda n, **kw: np.hanning(n),
             "ハン窓。汎用。干渉縞解析の標準的な選択"),
    "hamming": (lambda n, **kw: np.hamming(n),
                "ハミング窓。第一サイドローブが低い"),
    "blackman": (lambda n, **kw: np.blackman(n),
                 "ブラックマン窓。漏れが小さくメインローブは広い"),
    "bartlett": (lambda n, **kw: np.bartlett(n),
                 "バートレット窓 (三角窓)"),
    "blackmanharris": (lambda n, **kw: _cosine_sum_window(
                           n, (0.35875, 0.48829, 0.14128, 0.01168)),
                       "ブラックマン・ハリス窓。サイドローブ -92 dB"),
    "nuttall": (lambda n, **kw: _cosine_sum_window(
                    n, (0.355768, 0.487396, 0.144232, 0.012604)),
                "ナットール窓。漏れが非常に小さい"),
    "flattop": (lambda n, **kw: _cosine_sum_window(
                    n, (0.21557895, 0.41663158, 0.277263158,
                        0.083578947, 0.006947368)),
                "フラットトップ窓。振幅精度が最良 (分解能は劣る)"),
    "kaiser": (lambda n, beta=8.6, **kw: np.kaiser(n, beta),
               "カイザー窓。--kaiser-beta で漏れと分解能を調整"),
}

#: 窓関数名のエイリアス
WINDOW_ALIASES = {
    "none": "rectangular", "rect": "rectangular", "boxcar": "rectangular",
    "hanning": "hann", "bh": "blackmanharris", "blackman-harris": "blackmanharris",
    "flat-top": "flattop", "triangular": "bartlett",
}


def resolve_window_name(name: str) -> str:
    """窓関数名 (エイリアス可) を正式名に解決する."""
    key = name.strip().lower()
    key = WINDOW_ALIASES.get(key, key)
    if key not in WINDOW_FUNCTIONS:
        raise ValueError(
            f"未知の窓関数: {name!r}. 選択肢: {', '.join(WINDOW_FUNCTIONS)}")
    return key


def make_window(name: str, n: int, kaiser_beta: float = 8.6) -> np.ndarray:
    """指定した窓関数の係数配列 (長さ n) を返す."""
    key = resolve_window_name(name)
    return np.asarray(WINDOW_FUNCTIONS[key][0](n, beta=kaiser_beta), dtype=float)


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class SpectrumData:
    """CSV から読み込んだ干渉スペクトル一式."""

    wavelength: np.ndarray            # A 列: 波長 (nm), 昇順・等間隔
    intensity: np.ndarray             # shape = (n_points, n_columns)
    column_names: list[str]           # B 列以降の元の列名
    conditions: np.ndarray            # 列名を数値化した条件値
    delta_lambda: float               # 波長サンプリング間隔 (nm)
    resampled: bool = False           # 等間隔化のため再サンプリングしたか
    source: str = ""

    @property
    def nyquist_per_100nm(self) -> float:
        """A 列から導出した波長軸 FFT のナイキスト周波数 [波数/100nm]."""
        return WAVENUMBER_UNIT_NM / (2.0 * self.delta_lambda)


@dataclass
class FitResult:
    """1 列分の精密化 (グリッドサーチ + Levenberg-Marquardt 法) の結果.

    位置の値はすべて共役軸の単位 (波数/100nm または OPD [nm]) で持つ。
    """

    method: str                       # "grid" または "lm"
    fft_frequency: float              # 出発点にした FFT のピーク位置
    grid_frequency: float             # グリッドサーチで χ² 最小となった位置
    frequency: float                  # 最終推定値
    stderr: float | None              # 最終推定値の標準誤差 (LM のみ)
    amplitude: float                  # フィットした正弦波の振幅
    phase: float                      # 帯域中央での位相 [rad]
    rmse: float                       # 残差の二乗平均平方根
    r2: float                         # モデルの決定係数
    iterations: int                   # LM の反復回数
    converged: bool                   # LM が収束したか
    note: str                         # 補足 (フォールバックした理由など)
    grid_frequencies: np.ndarray      # グリッドサーチで調べた位置
    grid_chi2: np.ndarray             # 各位置での χ²
    x: np.ndarray                     # フィットに使った独立変数 (λ または 1/λ)
    y: np.ndarray                     # フィットに使った反射率
    model: np.ndarray                 # フィット結果のモデル値
    params: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_center: float = 0.0             # 座標正規化の中心
    half_width: float = 1.0           # 座標正規化の半幅
    degree: int = 1                   # ベースライン多項式の次数

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """任意の独立変数でフィット後のモデル値を計算する (作図用)."""
        u = (np.asarray(x, dtype=float) - self.x_center) / self.half_width
        return _sinusoid_design(u, self.params[0], self.degree) @ self.params[1:]


@dataclass
class ColumnFFTResult:
    """1 条件 (1 列) 分のフーリエ変換結果."""

    column: str
    condition: float
    frequency: np.ndarray             # 共役軸 (波数/100nm または OPD [nm])
    amplitude: np.ndarray             # 振幅 (入力と同じ単位)
    peak_frequency: float             # FFT で求めた振幅最大の位置
    peak_amplitude: float
    peak_index: int
    fit: FitResult | None = None      # 精密化した場合の結果

    @property
    def estimate(self) -> float:
        """回帰と出力に使う最終推定値 (精密化していれば精密化後の値)."""
        return self.fit.frequency if self.fit is not None else self.peak_frequency


@dataclass
class RegressionResult:
    """条件 対 振幅最大位置 の線形回帰結果."""

    slope: float
    intercept: float
    r2: float
    nyquist: float
    crossing_condition: float | None   # 回帰直線がナイキスト限界に達する条件
    extrapolated: bool                 # 交点が測定条件範囲の外か

    @property
    def equation(self) -> str:
        sign = "+" if self.intercept >= 0 else "-"
        return f"y = {self.slope:.4g} x {sign} {abs(self.intercept):.4g}"


@dataclass
class BandAnalysis:
    """1 つの波長帯域・1 つのドメインに対する解析結果一式."""

    domain: DomainInfo
    band: tuple[float, float]          # ユーザー指定の波長帯域 [nm]
    band_actual: tuple[float, float]   # 実データ上の波長帯域 [nm]
    grid: np.ndarray                   # FFT に用いた等間隔グリッド
    signals: np.ndarray                # グリッド上の信号 (n_points, n_columns)
    n_points: int
    step: float                        # グリッド間隔 (Δλ または Δ(1/λ))
    window: str
    detrend: str
    zero_pad: int
    nyquist: float                     # ナイキスト限界 (共役軸の単位)
    resolution: float                  # 共役軸の分解能
    interp: str = "cubic"              # グリッド再配置に使った補間方式
    fit_method: str = "none"           # 精密化の方式 (none / grid / lm)
    grid_points: int = 0               # グリッドサーチの分割数 M
    results: list[ColumnFFTResult] = field(default_factory=list)
    regression: RegressionResult | None = None

    @property
    def label(self) -> str:
        lo, hi = self.band
        return f"band_{lo:g}-{hi:g}nm_{self.domain.suffix}"


# ---------------------------------------------------------------------------
# CSV 読み込み
# ---------------------------------------------------------------------------

def load_spectra(csv_path: str | Path, encoding: str | None = None) -> SpectrumData:
    """干渉スペクトル CSV を読み込む.

    A 列: 波長 (nm)、B 列以降: 各条件での反射率。
    B 列以降の列名は条件を表す数値である必要がある。
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV が見つかりません: {path}")

    encodings = [encoding] if encoding else ["utf-8-sig", "cp932", "utf-8"]
    last_err: Exception | None = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as exc:      # pragma: no cover - 環境依存
            last_err = exc
    if df is None:
        raise RuntimeError(f"CSV を読み込めませんでした: {last_err}")

    if df.shape[1] < 2:
        raise ValueError("CSV には波長列と 1 つ以上の条件列が必要です。")

    # --- A 列: 波長 ---------------------------------------------------------
    wl_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    valid_rows = wl_raw.notna().to_numpy()
    if valid_rows.sum() < 4:
        raise ValueError("A 列 (波長) の有効な数値が不足しています。")

    # --- B 列以降: 値のある列のみ採用 --------------------------------------
    column_names: list[str] = []
    conditions: list[float] = []
    columns: list[np.ndarray] = []
    for name in df.columns[1:]:
        series = pd.to_numeric(df[name], errors="coerce")[valid_rows]
        if series.notna().sum() == 0:
            continue                                   # 空列はスキップ
        try:
            cond = float(str(name).strip())
        except ValueError:
            print(f"[警告] 列 '{name}' は列名が数値でないため除外しました。",
                  file=sys.stderr)
            continue
        column_names.append(str(name))
        conditions.append(cond)
        columns.append(series.to_numpy(dtype=float))

    if not columns:
        raise ValueError("解析対象となる条件列 (列名が数値の列) がありません。")

    wl = wl_raw[valid_rows].to_numpy(dtype=float)
    data = np.column_stack(columns)

    # --- 波長昇順に整列 -----------------------------------------------------
    order = np.argsort(wl)
    wl, data = wl[order], data[order]

    # --- 列内の欠測を線形補間 ----------------------------------------------
    for j in range(data.shape[1]):
        col = data[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            if (~nan_mask).sum() < 4:
                raise ValueError(f"列 '{column_names[j]}' の有効データが不足しています。")
            col[nan_mask] = np.interp(wl[nan_mask], wl[~nan_mask], col[~nan_mask])
            print(f"[情報] 列 '{column_names[j]}' の欠測 {nan_mask.sum()} 点を線形補間しました。",
                  file=sys.stderr)

    # --- 等間隔性の確認 (必要なら再サンプリング) ---------------------------
    diffs = np.diff(wl)
    if np.any(diffs <= 0):
        raise ValueError("A 列に重複した波長があります。")
    step = float(np.median(diffs))
    resampled = False
    if np.max(np.abs(diffs - step)) > UNIFORM_GRID_RTOL * step:
        n_new = int(round((wl[-1] - wl[0]) / step)) + 1
        wl_new = np.linspace(wl[0], wl[-1], n_new)
        data = np.column_stack([interpolate(wl, data[:, j], wl_new, "cubic")
                                for j in range(data.shape[1])])
        wl = wl_new
        step = float(wl[1] - wl[0])
        resampled = True
        print("[情報] 波長軸が等間隔でないため、中央値間隔 "
              f"{step:.6g} nm の等間隔グリッドへ再サンプリングしました。", file=sys.stderr)

    return SpectrumData(
        wavelength=wl,
        intensity=data,
        column_names=column_names,
        conditions=np.asarray(conditions, dtype=float),
        delta_lambda=step,
        resampled=resampled,
        source=str(path),
    )


# ---------------------------------------------------------------------------
# 帯域切り出しとグリッド生成
# ---------------------------------------------------------------------------

def select_band(spec: SpectrumData, lo: float, hi: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """波長帯域 [lo, hi] のデータを切り出す."""
    if hi <= lo:
        raise ValueError(f"波長帯域の指定が不正です: {lo}-{hi}")
    mask = (spec.wavelength >= lo) & (spec.wavelength <= hi)
    n = int(mask.sum())
    if n < 8:
        raise ValueError(
            f"帯域 {lo}-{hi} nm のデータ点が {n} 点しかありません "
            f"(データ範囲: {spec.wavelength[0]:g}-{spec.wavelength[-1]:g} nm)")
    return spec.wavelength[mask], spec.intensity[mask]


def interpolate(x_src: np.ndarray, y_src: np.ndarray, x_new: np.ndarray,
                kind: str = "cubic") -> np.ndarray:
    """再サンプリング用の補間。線形または 3 次エルミート補間を選べる.

    線形補間は 1 周期あたりのデータ点数が少ない (ナイキストに近い) 成分の
    振幅を大きく削ってしまう。3 次エルミート補間なら減衰をほぼ抑えられる。
    x_src は昇順であればよく、等間隔である必要はない。
    """
    if kind == "linear":
        return np.interp(x_new, x_src, y_src)
    if kind != "cubic":
        raise ValueError(f"未知の補間方式: {kind!r} (linear/cubic)")

    h = np.diff(x_src)
    slope = np.diff(y_src) / h
    # 各節点の傾きを 3 点差分 (不等間隔対応) で求める
    m = np.empty_like(y_src, dtype=float)
    m[1:-1] = ((slope[:-1] * h[1:] + slope[1:] * h[:-1]) / (h[1:] + h[:-1]))
    m[0], m[-1] = slope[0], slope[-1]

    idx = np.clip(np.searchsorted(x_src, x_new) - 1, 0, x_src.size - 2)
    t = (x_new - x_src[idx]) / h[idx]
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (h00 * y_src[idx] + h10 * h[idx] * m[idx]
            + h01 * y_src[idx + 1] + h11 * h[idx] * m[idx + 1])


def build_analysis_grid(spec: SpectrumData, lo: float, hi: float,
                        domain: DomainInfo, interp: str = "cubic"
                        ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """FFT に使う等間隔グリッドと、その上の信号を作る.

    波長ドメインでは読み込み済みの等間隔波長グリッドをそのまま使う。
    波数ドメインでは 1/λ が等間隔になるよう再サンプリングする
    (等間隔な λ は 1/λ では不等間隔になるため、この再配置が必須)。
    """
    wl_band, data_band = select_band(spec, lo, hi)
    band_actual = (float(wl_band[0]), float(wl_band[-1]))

    if domain.key == "wavelength":
        return wl_band, data_band, band_actual

    # 波数ドメイン: λ 昇順 -> 1/λ は降順なので反転してから等間隔化する
    sigma_src = 1.0 / wl_band[::-1]
    data_src = data_band[::-1]
    grid = np.linspace(sigma_src[0], sigma_src[-1], wl_band.size)
    signals = np.column_stack([interpolate(sigma_src, data_src[:, j], grid, interp)
                               for j in range(data_src.shape[1])])
    return grid, signals, band_actual


# ---------------------------------------------------------------------------
# フーリエ変換
# ---------------------------------------------------------------------------

def detrend_signal(y: np.ndarray, mode: str) -> np.ndarray:
    """信号のトレンド (直流/傾き) を除去する.

    干渉縞そのものではないベースラインを落とすことで、振幅最大ピークが
    直流成分に埋もれるのを防ぐ。
    """
    if mode == "none":
        return y.astype(float)
    if mode == "mean":
        return y - float(np.mean(y))
    if mode == "linear":
        x = np.arange(y.size, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        return y - (slope * x + intercept)
    raise ValueError(f"未知のデトレンド方式: {mode!r} (none/mean/linear)")


def parabolic_peak(x: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float]:
    """離散ピーク周辺を放物線近似し、サブビン精度のピーク位置と振幅を返す."""
    if idx <= 0 or idx >= len(y) - 1:
        return float(x[idx]), float(y[idx])
    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(x[idx]), float(y[idx])
    delta = 0.5 * (y0 - y2) / denom              # -0.5 .. +0.5 [bin]
    delta = float(np.clip(delta, -0.5, 0.5))
    dx = float(x[idx + 1] - x[idx])
    return float(x[idx]) + delta * dx, y1 - 0.25 * (y0 - y2) * delta


def fourier_transform(grid: np.ndarray, y: np.ndarray, window: str = "hann",
                      detrend: str = "linear", zero_pad: int = 1,
                      kaiser_beta: float = 8.6,
                      unit_scale: float = WAVENUMBER_UNIT_NM
                      ) -> tuple[np.ndarray, np.ndarray]:
    """等間隔グリッド上の信号をフーリエ変換し (共役軸, 振幅) を返す.

    通常の FFT は「単位時間当たりの波数 = 周波数 [Hz]」を返すが、独立変数が
    波長 λ [nm] なら共役軸は 1 nm 当たりの波数となる。これを 100 倍して
    波数/100nm とする (unit_scale = 100)。
    独立変数が波数 1/λ [1/nm] なら共役軸はそのまま長さ [nm]、すなわち
    光路差 OPD になる (unit_scale = 1)。

    振幅は窓関数のコヒーレントゲイン (窓係数の総和) で正規化し、実信号の
    正弦波成分の振幅と一致するよう片側スペクトルを 2 倍している。
    """
    n = y.size
    step = float(np.mean(np.diff(grid)))
    w = make_window(window, n, kaiser_beta=kaiser_beta)
    y_proc = detrend_signal(np.asarray(y, dtype=float), detrend) * w

    n_fft = int(n * max(1, int(zero_pad)))
    spectrum = np.fft.rfft(y_proc, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, d=step)

    coherent_gain = float(np.sum(w))
    amp = 2.0 * np.abs(spectrum) / coherent_gain
    amp[0] = np.abs(spectrum[0]) / coherent_gain       # 直流成分は 2 倍しない
    if n_fft % 2 == 0:
        amp[-1] = np.abs(spectrum[-1]) / coherent_gain  # ナイキスト成分も同様

    return freq * unit_scale, amp


# ---------------------------------------------------------------------------
# 精密化: グリッドサーチ + Levenberg-Marquardt 法
# ---------------------------------------------------------------------------
#
# M. Quinten, "On the use of fast Fourier transform for optical layer thickness
# determination", SN Applied Sciences 1, 823 (2019) 第 5 章の 2 段階法を実装する。
#
#   1. FFT のピーク f_FFT の周辺 [f_FFT - 2Δf, f_FFT + 2Δf] を M 等分し、
#      各点でモデルとデータの二乗誤差 χ² を計算して最小の点を選ぶ
#      (Δf は分解能。論文の d_min に当たる。論文の推奨は M = 16)
#   2. その点を初期値として Levenberg-Marquardt 法で非線形最小二乗フィットする
#
# モデルは各ドメインの座標 x に対する正弦波 + 多項式ベースライン:
#   y(x) = Σ b_k u^k + A_c cos(2π g u) + A_s sin(2π g u),   u = (x - x_c) / h
# x_c は帯域中央、h は帯域の半幅。g は半幅あたりの周期数で、共役軸の値 f とは
# f = g / h × unit_scale の関係にある。座標を [-1, 1] に正規化しておくと、
# 波長ドメインでも波数ドメインでも同じ数値スケールで計算でき、条件数が良くなる。

FIT_METHODS = ("none", "grid", "lm")

#: グリッドサーチの既定の分割数 M (論文の推奨値)
DEFAULT_GRID_POINTS = 16

#: グリッドサーチの探索範囲の半幅 (分解能の何倍か。論文では 2)
GRID_HALF_WIDTH = 2.0

#: デトレンド方式に対応する、フィットのベースライン多項式の次数
BASELINE_DEGREE = {"none": 0, "mean": 0, "linear": 1}


def fit_samples(spec: SpectrumData, lo: float, hi: float, domain: DomainInfo
                ) -> tuple[np.ndarray, np.ndarray]:
    """非線形フィットに使うデータ (昇順の独立変数, 信号) を返す.

    FFT と違い、最小二乗フィットはデータが等間隔である必要がない。
    そのため波数ドメインでも 1/λ グリッドへ補間する前の点をそのまま使い、
    補間による振幅の減衰や誤差を持ち込まないようにしている。
    """
    wl_band, data_band = select_band(spec, lo, hi)
    if domain.key == "wavelength":
        return wl_band, data_band
    return 1.0 / wl_band[::-1], data_band[::-1]


def _sinusoid_design(u: np.ndarray, g: float, degree: int) -> np.ndarray:
    """周期 g を固定したときの計画行列 [1, u, ..., u^deg, cos, sin]."""
    phase = 2.0 * np.pi * g * u
    cols = [u ** k for k in range(degree + 1)]
    cols += [np.cos(phase), np.sin(phase)]
    return np.column_stack(cols)


def _solve_linear(u: np.ndarray, y: np.ndarray, g: float, degree: int
                  ) -> tuple[np.ndarray, float]:
    """周期 g を固定し、残りの線形パラメータを最小二乗で解いて χ² を返す.

    周期以外のパラメータ (ベースライン、cos と sin の係数) はモデルに線形に
    入るので、周期さえ決めれば一発で最適値が求まる。グリッドサーチでは
    これを使い、実質的に周期 1 個だけを走査している。
    """
    design = _sinusoid_design(u, g, degree)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return coef, float(resid @ resid)


def grid_search(u: np.ndarray, y: np.ndarray, g0: float, dg: float,
                degree: int, n_grid: int = DEFAULT_GRID_POINTS
                ) -> tuple[np.ndarray, np.ndarray, int]:
    """[g0 - 2dg, g0 + 2dg] を n_grid 等分した各点の χ² を調べる (論文の手順 1).

    戻り値は (調べた周期, 各 χ², χ² 最小の添字)。周期は正でなければならない
    ので、FFT ピークが低周波側の端にあるときは負の側を捨てる。
    """
    offsets = np.linspace(-GRID_HALF_WIDTH, GRID_HALF_WIDTH, int(n_grid) + 1)
    gs = g0 + offsets * dg
    gs = gs[gs > 0.0]
    if gs.size == 0:
        raise ValueError("グリッドサーチの探索範囲が空です。")
    chi2 = np.array([_solve_linear(u, y, g, degree)[1] for g in gs])
    return gs, chi2, int(np.argmin(chi2))


def levenberg_marquardt(u: np.ndarray, y: np.ndarray, p0: np.ndarray,
                        degree: int, max_iter: int = 100,
                        ftol: float = 1e-12, xtol: float = 1e-12
                        ) -> tuple[np.ndarray, float, np.ndarray, int, bool]:
    """Levenberg-Marquardt 法で正弦波 + ベースラインを非線形フィットする (手順 2).

    パラメータ p = [g, b_0, ..., b_deg, A_c, A_s]。
    各反復で (JᵀJ + λ·diag(JᵀJ)) δ = Jᵀr を解いて p を更新する。
    λ が小さいとガウス・ニュートン法 (速いが初期値が悪いと発散しやすい)、
    大きいと最急降下法 (遅いが確実に下る) に近づく。χ² が下がれば λ を
    1/10 にして大胆に、下がらなければ 10 倍にして慎重に進む。

    戻り値は (最終パラメータ, χ², ヤコビアン, 反復回数, 収束したか)。
    """
    powers = np.column_stack([u ** k for k in range(degree + 1)])
    i_c, i_s = degree + 2, degree + 3

    def evaluate(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phase = 2.0 * np.pi * p[0] * u
        c, s = np.cos(phase), np.sin(phase)
        model = powers @ p[1:degree + 2] + p[i_c] * c + p[i_s] * s
        jac = np.empty((u.size, p.size))
        jac[:, 0] = 2.0 * np.pi * u * (-p[i_c] * s + p[i_s] * c)   # ∂y/∂g
        jac[:, 1:degree + 2] = powers                                # ∂y/∂b_k
        jac[:, i_c] = c                                              # ∂y/∂A_c
        jac[:, i_s] = s                                              # ∂y/∂A_s
        return model, jac

    p = np.asarray(p0, dtype=float).copy()
    model, jac = evaluate(p)
    resid = y - model
    chi2 = float(resid @ resid)
    lam = 1e-3
    tiny = np.finfo(float).tiny

    for it in range(1, int(max_iter) + 1):
        jtj = jac.T @ jac
        grad = jac.T @ resid
        scale = np.diag(jtj).copy()
        scale[scale <= 0.0] = 1.0
        while True:
            try:
                delta = np.linalg.solve(jtj + lam * np.diag(scale), grad)
            except np.linalg.LinAlgError:
                delta = None
            if delta is not None and np.all(np.isfinite(delta)):
                p_new = p + delta
                model_new, jac_new = evaluate(p_new)
                resid_new = y - model_new
                chi2_new = float(resid_new @ resid_new)
                if chi2_new <= chi2:
                    break
            lam *= 10.0
            if lam > 1e16:
                # どの方向に動いても χ² が下がらない = すでに極小にいる
                return p, chi2, jac, it - 1, True
        small_f = (chi2 - chi2_new) <= ftol * max(chi2, tiny)
        small_x = bool(np.all(np.abs(delta) <= xtol * (np.abs(p) + xtol)))
        p, model, jac, resid, chi2 = p_new, model_new, jac_new, resid_new, chi2_new
        lam = max(lam / 10.0, 1e-15)
        if small_f or small_x:
            return p, chi2, jac, it, True
    return p, chi2, jac, int(max_iter), False


def refine_peak_fit(x: np.ndarray, y: np.ndarray, fft_peak: float,
                    resolution: float, unit_scale: float, method: str = "lm",
                    degree: int = 1, n_grid: int = DEFAULT_GRID_POINTS,
                    max_iter: int = 100) -> FitResult:
    """FFT のピークを出発点に、グリッドサーチと LM 法でピーク位置を精密化する.

    x, y はフィットに使うデータ、fft_peak と resolution は共役軸の単位。
    method が "grid" ならグリッドサーチまで、"lm" なら LM 法まで実行する。
    """
    if method not in ("grid", "lm"):
        raise ValueError(f"未知の精密化方式: {method!r} (grid/lm)")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_c = 0.5 * (x.min() + x.max())
    half = 0.5 * (x.max() - x.min())
    u = (x - x_c) / half
    to_g = half / unit_scale                    # 共役軸の値 -> 半幅あたりの周期数

    g0 = fft_peak * to_g
    dg = resolution * to_g
    gs, chi2s, k = grid_search(u, y, g0, dg, degree, n_grid)
    g_grid = float(gs[k])
    coef, chi2 = _solve_linear(u, y, g_grid, degree)
    params = np.concatenate([[g_grid], coef])

    iterations, converged, stderr, note = 0, True, None, ""
    if method == "lm":
        p_lm, chi2_lm, jac, iterations, converged = levenberg_marquardt(
            u, y, params, degree, max_iter=max_iter)
        # 探索範囲から大きく外れたら、別の谷に落ちたとみなして採用しない
        if p_lm[0] <= 0.0 or abs(p_lm[0] - g0) > (GRID_HALF_WIDTH + 0.5) * dg:
            converged = False
            note = "LM 法が探索範囲の外へ移動したため、グリッドサーチの結果を採用"
        else:
            params, chi2 = p_lm, chi2_lm
            dof = max(y.size - params.size, 1)
            try:
                cov = np.linalg.inv(jac.T @ jac) * (chi2 / dof)
                stderr = float(np.sqrt(max(cov[0, 0], 0.0))) / to_g
            except np.linalg.LinAlgError:
                stderr = float("nan")
            if not converged:
                note = f"LM 法が {max_iter} 回で収束しなかった"

    model = _sinusoid_design(u, params[0], degree) @ params[1:]
    a_c, a_s = float(params[-2]), float(params[-1])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return FitResult(
        method=method,
        fft_frequency=float(fft_peak),
        grid_frequency=g_grid / to_g,
        frequency=float(params[0]) / to_g,
        stderr=stderr,
        amplitude=float(np.hypot(a_c, a_s)),
        phase=float(np.arctan2(-a_s, a_c)),
        rmse=float(np.sqrt(chi2 / y.size)),
        r2=1.0 - chi2 / ss_tot if ss_tot > 0 else float("nan"),
        iterations=int(iterations),
        converged=bool(converged),
        note=note,
        grid_frequencies=gs / to_g,
        grid_chi2=chi2s,
        x=x,
        y=y,
        model=model,
        params=params,
        x_center=float(x_c),
        half_width=float(half),
        degree=int(degree),
    )


def analyze_band(spec: SpectrumData, lo: float, hi: float, window: str = "hann",
                 detrend: str = "linear", zero_pad: int = 1,
                 kaiser_beta: float = 8.6, peak_min: float = 0.0,
                 refine_peak: bool = False,
                 domain: str | DomainInfo = "wavelength",
                 interp: str = "cubic", fit: str = "none",
                 grid_points: int = DEFAULT_GRID_POINTS,
                 max_iter: int = 100,
                 baseline_degree: int | None = None) -> BandAnalysis:
    """指定帯域・指定ドメインで全条件列の FFT とピーク抽出を行う.

    fit が "grid" または "lm" なら、FFT のピークを出発点に精密化も行う。
    baseline_degree はフィットのベースライン多項式の次数で、None なら
    デトレンド方式に合わせる (linear なら 1 次、それ以外は 0 次)。
    """
    if fit not in FIT_METHODS:
        raise ValueError(f"未知の精密化方式: {fit!r} ({'/'.join(FIT_METHODS)})")
    dom = domain if isinstance(domain, DomainInfo) else resolve_domain(domain)
    grid, signals, band_actual = build_analysis_grid(spec, lo, hi, dom, interp)
    n = grid.size
    step = float(np.mean(np.diff(grid)))

    analysis = BandAnalysis(
        domain=dom,
        band=(lo, hi),
        band_actual=band_actual,
        grid=grid,
        signals=signals,
        n_points=n,
        step=step,
        window=resolve_window_name(window),
        detrend=detrend,
        zero_pad=max(1, int(zero_pad)),
        nyquist=dom.unit_scale / (2.0 * step),
        resolution=dom.unit_scale / (n * step),
        interp=interp,
        fit_method=fit,
        grid_points=int(grid_points) if fit != "none" else 0,
    )

    if fit != "none":
        x_fit, y_fit = fit_samples(spec, lo, hi, dom)
        degree = (int(baseline_degree) if baseline_degree is not None
                  else BASELINE_DEGREE.get(detrend, 1))
        if degree < 0:
            raise ValueError("ベースラインの次数は 0 以上にしてください。")

    for j, name in enumerate(spec.column_names):
        freq, amp = fourier_transform(grid, signals[:, j], window=window,
                                      detrend=detrend, zero_pad=zero_pad,
                                      kaiser_beta=kaiser_beta,
                                      unit_scale=dom.unit_scale)
        # 直流成分 (および peak_min 未満) を除外して振幅最大のピークを探す
        search = freq > max(peak_min, 0.0)
        search[0] = False
        if not search.any():
            raise ValueError("ピーク探索範囲が空です。--peak-min を小さくしてください。")
        idx = int(np.argmax(np.where(search, amp, -np.inf)))
        pk_freq, pk_amp = (parabolic_peak(freq, amp, idx) if refine_peak
                           else (float(freq[idx]), float(amp[idx])))

        fit_result = None
        if fit != "none":
            fit_result = refine_peak_fit(
                x_fit, y_fit[:, j], pk_freq, analysis.resolution, dom.unit_scale,
                method=fit, degree=degree, n_grid=grid_points, max_iter=max_iter)

        analysis.results.append(ColumnFFTResult(
            column=name,
            condition=float(spec.conditions[j]),
            frequency=freq,
            amplitude=amp,
            peak_frequency=pk_freq,
            peak_amplitude=pk_amp,
            peak_index=idx,
            fit=fit_result,
        ))
    return analysis


# ---------------------------------------------------------------------------
# 線形回帰
# ---------------------------------------------------------------------------

def fit_regression(analysis: BandAnalysis) -> RegressionResult:
    """条件 (列名) 対 振幅最大位置 の線形回帰と、ナイキスト到達条件の算出."""
    x = np.array([r.condition for r in analysis.results], dtype=float).reshape(-1, 1)
    y = np.array([r.estimate for r in analysis.results], dtype=float)

    model = LinearRegression().fit(x, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(r2_score(y, model.predict(x)))

    nyq = analysis.nyquist
    crossing: float | None = None
    extrapolated = False
    if slope != 0.0:
        crossing = (nyq - intercept) / slope
        extrapolated = not (x.min() <= crossing <= x.max())

    result = RegressionResult(slope=slope, intercept=intercept, r2=r2,
                              nyquist=nyq, crossing_condition=crossing,
                              extrapolated=extrapolated)
    analysis.regression = result
    return result


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------

def _condition_palette(n: int) -> list:
    return sns.color_palette("viridis", n_colors=max(n, 2))[:n]


def plot_raw_spectra(spec: SpectrumData, analysis: BandAnalysis,
                     out_path: Path | None) -> plt.Figure:
    """FFT に投入した信号 (帯域切り出し後・グリッド再配置後) を描画する."""
    dom = analysis.domain
    colors = _condition_palette(len(spec.column_names))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, name in enumerate(spec.column_names):
        ax.plot(analysis.grid, analysis.signals[:, j], lw=1.0,
                color=colors[j], label=name)
    ax.set_xlabel(dom.sample_axis_label)
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Interference spectra  ({analysis.band[0]:g}-"
                 f"{analysis.band[1]:g} nm, domain: {dom.key})")
    ax.legend(title="Condition", ncol=2, fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def auto_xmax(analysis: BandAnalysis, rel_threshold: float = 0.01,
              margin: float = 1.3) -> float:
    """フーリエスペクトル図の横軸上限を自動決定する.

    干渉縞のピークはナイキスト限界よりはるかに低いことが多く、
    横軸をナイキストまで取るとピークが原点に潰れて読めない。
    最大振幅の rel_threshold 倍を超える成分が存在する最高位置を上限とする。
    """
    nyq = analysis.nyquist
    peak_max = max(r.peak_amplitude for r in analysis.results)
    threshold = peak_max * rel_threshold
    hi = 0.0
    for r in analysis.results:
        mask = r.amplitude >= threshold
        mask[0] = False                                   # 直流成分は無視
        if mask.any():
            hi = max(hi, float(r.frequency[int(np.max(np.nonzero(mask)))]))
    hi = max(hi * margin, 5.0 * analysis.resolution)
    return float(min(hi, nyq))


def plot_fourier_spectra(analysis: BandAnalysis, out_path: Path | None,
                         xmax: float | str | None = "auto") -> plt.Figure:
    """フーリエスペクトル (横軸: 共役軸, 縦軸: 振幅) の折れ線グラフ."""
    dom = analysis.domain
    colors = _condition_palette(len(analysis.results))
    nyq = analysis.nyquist

    if xmax is None or (isinstance(xmax, str) and xmax.lower() in ("nyquist", "full")):
        x_hi = nyq
    elif isinstance(xmax, str) and xmax.lower() == "auto":
        x_hi = auto_xmax(analysis)
    else:
        x_hi = float(xmax)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for res, color in zip(analysis.results, colors):
        ax.plot(res.frequency, res.amplitude, lw=1.1, color=color,
                label=f"{res.column}")
        ax.plot([res.peak_frequency], [res.peak_amplitude], "o", ms=5,
                color=color, mec="black", mew=0.6, zorder=5)
    # 精密化した位置と振幅 (FFT の格子点の間にある真のピーク)
    fitted = [r.fit for r in analysis.results if r.fit is not None]
    if fitted:
        ax.plot([f.frequency for f in fitted], [f.amplitude for f in fitted],
                ls="none", marker="+", ms=11, mew=1.6, color="black", zorder=6,
                label=f"{analysis.fit_method.upper()} fit")

    if x_hi >= nyq * 0.999:
        ax.axvline(nyq, color="crimson", ls="--", lw=1.2,
                   label=f"Nyquist = {nyq:.4g}")
    else:
        ax.text(0.99, 0.02,
                f"x-axis truncated (Nyquist = {nyq:.4g} {dom.freq_unit_en})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color="crimson")
    ax.set_xlim(0, x_hi)
    ax.set_xlabel(dom.freq_axis_label)
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Fourier spectra  ({analysis.band[0]:g}-{analysis.band[1]:g} nm, "
        f"domain: {dom.key}, window: {analysis.window}, N = {analysis.n_points})")
    ax.legend(title="Condition", ncol=2, fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_peak_scatter(analysis: BandAnalysis, out_path: Path | None) -> plt.Figure:
    """条件 対 振幅最大位置 の散布図・回帰直線・ナイキスト線・交点を描画する."""
    dom = analysis.domain
    reg = analysis.regression or fit_regression(analysis)
    x = np.array([r.condition for r in analysis.results], dtype=float)
    y = np.array([r.estimate for r in analysis.results], dtype=float)
    nyq = reg.nyquist
    refined = analysis.fit_method != "none"
    y_fft = np.array([r.peak_frequency for r in analysis.results], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    if refined:
        # 精密化前の FFT の値を白抜きで重ね、補正量が見えるようにする
        ax.scatter(x, y_fft, s=90, facecolor="none", edgecolor="0.45", lw=1.2,
                   zorder=4, label="FFT peak")
    ax.scatter(x, y, s=70, color="#1f77b4", edgecolor="black", zorder=5,
               label=(f"Peak {dom.freq_short_en} ({analysis.fit_method.upper()})"
                      if refined else f"Peak {dom.freq_short_en}"))

    # 回帰直線 (交点が現実的な範囲にあれば、そこまで延長して描く)
    span = float(x.max() - x.min()) or 1.0
    x_hi = float(x.max()) + 0.05 * span
    if reg.crossing_condition is not None and reg.extrapolated:
        if x.min() - 3 * span <= reg.crossing_condition <= x.max() + 3 * span:
            x_hi = max(x_hi, reg.crossing_condition + 0.05 * span)
    x_line = np.linspace(float(x.min()) - 0.05 * span, x_hi, 200)
    ax.plot(x_line, reg.slope * x_line + reg.intercept, color="#ff7f0e", lw=1.8,
            label="Linear regression")

    # ナイキスト限界の破線
    ax.axhline(nyq, color="crimson", ls="--", lw=1.5,
               label=f"Nyquist = {nyq:.4g} {dom.freq_unit_en}")

    # 回帰直線とナイキスト線の交点
    if reg.crossing_condition is not None and x_line[0] <= reg.crossing_condition <= x_line[-1]:
        ax.plot([reg.crossing_condition], [nyq], marker="X", ms=13,
                color="crimson", mec="black", mew=0.8, zorder=6,
                label=f"Crossing: condition = {reg.crossing_condition:.4g}")
        ax.vlines(reg.crossing_condition, ax.get_ylim()[0], nyq,
                  color="crimson", ls=":", lw=1.2)

    text = (f"{reg.equation}\n$R^2$ = {reg.r2:.4f}\n"
            f"Nyquist = {nyq:.4g} {dom.freq_unit_en}")
    if reg.crossing_condition is not None:
        text += (f"\ncondition @ Nyquist = {reg.crossing_condition:.4g}"
                 + ("  (extrapolated)" if reg.extrapolated else ""))
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    ax.set_xlabel("Condition (column name)")
    ax.set_ylabel(f"Peak {dom.freq_short_en} ({dom.freq_unit_en})")
    ax.set_title(f"Peak {dom.freq_short_en} vs condition  "
                 f"({analysis.band[0]:g}-{analysis.band[1]:g} nm, "
                 f"domain: {dom.key}, window: {analysis.window})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
              fontsize=9, frameon=False)

    # 実測点はナイキスト線よりはるかに低いことが多いため、拡大図を重ねる
    y_span = float(y.max() - y.min())
    if (nyq - y.min()) > 0 and y_span < 0.25 * (nyq - y.min()):
        inset = ax.inset_axes((0.50, 0.42, 0.46, 0.34))
        if refined:
            inset.scatter(x, y_fft, s=50, facecolor="none", edgecolor="0.45",
                          lw=1.0, zorder=4)
        inset.scatter(x, y, s=40, color="#1f77b4", edgecolor="black", zorder=5)
        xz = np.linspace(float(x.min()) - 0.05 * span, float(x.max()) + 0.05 * span, 50)
        inset.plot(xz, reg.slope * xz + reg.intercept, color="#ff7f0e", lw=1.5)
        inset.set_title("zoom on data points", fontsize=9)
        inset.tick_params(labelsize=8)
        inset.set_facecolor("white")
        inset.patch.set_alpha(0.95)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


#: フィット確認図で表示する縞の周期数
FIT_CHECK_FRINGES = 8


def plot_fit_check(analysis: BandAnalysis, out_path: Path | None) -> plt.Figure | None:
    """精密化したモデルとデータの重なり具合を条件ごとに描画する.

    周期の推定がずれていると、帯域中央から離れるほど位相のずれが大きくなる。
    そこで差が最も目立つ帯域の端 (独立変数の大きい側) の数周期を拡大して見せる。
    """
    fitted = [r for r in analysis.results if r.fit is not None]
    if not fitted:
        return None
    dom = analysis.domain
    ncols = min(3, len(fitted))
    nrows = int(np.ceil(len(fitted) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.9 * nrows),
                             squeeze=False)
    colors = _condition_palette(len(analysis.results))
    color_of = {r.column: c for r, c in zip(analysis.results, colors)}

    for ax, res in zip(axes.flat, fitted):
        fr = res.fit
        native = fr.frequency / dom.unit_scale            # 独立変数 1 単位あたりの周期
        span = FIT_CHECK_FRINGES / native if native > 0 else np.inf
        mask = fr.x >= fr.x.max() - span
        x_dense = np.linspace(fr.x[mask].min(), fr.x[mask].max(), 1500)
        ax.plot(x_dense, fr.evaluate(x_dense), "-", lw=1.1,
                color=color_of[res.column], label="fit", zorder=2)
        ax.plot(fr.x[mask], fr.y[mask], "o", ms=2.6, color="black",
                label="data", zorder=3)
        ax.set_title(f"condition {res.column}: {fr.frequency:.6g} {dom.freq_unit_en}",
                     fontsize=9)
        ax.text(0.02, 0.04, f"$R^2$ = {fr.r2:.5f}\nRMSE = {fr.rmse:.2e}",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.tick_params(labelsize=7)
        if dom.key == "wavenumber":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for ax in list(axes.flat)[len(fitted):]:
        ax.axis("off")

    fig.suptitle(f"Fit check  ({analysis.band[0]:g}-{analysis.band[1]:g} nm, "
                 f"domain: {dom.key}, method: {analysis.fit_method.upper()}, "
                 f"last ~{FIT_CHECK_FRINGES} fringes of the band)", fontsize=10)
    fig.supxlabel(dom.sample_axis_label, fontsize=9)
    fig.supylabel("Reflectance", fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
    return fig


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

#: ピークがナイキスト限界に近すぎると判定する幅 (分解能の何倍か)
NEAR_NYQUIST_MARGIN = 2.0


def is_near_nyquist(analysis: BandAnalysis, peak: float) -> bool:
    """ピークがナイキスト限界の直下にあるかを判定する.

    片側振幅スペクトルは正と負の周波数成分が重ならない前提で 2 倍している。
    ナイキスト限界のごく近傍ではこの前提が崩れ、振幅が最大 2 倍まで
    過大評価される。加えて、真の成分が限界を超えて折り返してきた結果
    そこに見えている可能性もある。
    """
    return peak >= analysis.nyquist - NEAR_NYQUIST_MARGIN * analysis.resolution


#: フィットの決定係数がこれを下回ったら、正弦波モデルが合っていないと警告する
FIT_R2_WARN = 0.9


def peaks_dataframe(analysis: BandAnalysis,
                    refractive_index: float | None = None) -> pd.DataFrame:
    """列ごとのピーク情報を DataFrame にまとめる.

    dom.peak_col の列には最終推定値 (精密化していれば精密化後の値) が入る。
    精密化した場合は、FFT とグリッドサーチの値、標準誤差、フィットの品質も付く。
    """
    dom = analysis.domain
    res = analysis.results
    df = pd.DataFrame({
        "column": [r.column for r in res],
        "condition": [r.condition for r in res],
        dom.peak_col: [r.estimate for r in res],
        "peak_amplitude": [r.peak_amplitude for r in res],
        dom.nyquist_col: analysis.nyquist,
        "peak_over_nyquist_ratio": [r.estimate / analysis.nyquist for r in res],
        "near_nyquist": [is_near_nyquist(analysis, r.estimate) for r in res],
    })

    if analysis.fit_method != "none":
        df["peak_method"] = analysis.fit_method
        df[f"fft_{dom.peak_col}"] = [r.peak_frequency for r in res]
        df[f"grid_{dom.peak_col}"] = [r.fit.grid_frequency for r in res]
        if analysis.fit_method == "lm":
            df[f"{dom.peak_col}_stderr"] = [r.fit.stderr for r in res]
        df["fit_amplitude"] = [r.fit.amplitude for r in res]
        df["fit_phase_rad"] = [r.fit.phase for r in res]
        df["fit_r2"] = [r.fit.r2 for r in res]
        df["fit_rmse"] = [r.fit.rmse for r in res]
        if analysis.fit_method == "lm":
            df["fit_iterations"] = [r.fit.iterations for r in res]
            df["fit_converged"] = [r.fit.converged for r in res]
        df["fit_note"] = [r.fit.note for r in res]

    # 波数ドメインでは光路差から膜厚 d = OPD / (2n) を推定できる
    if dom.key == "wavenumber" and refractive_index:
        n_ref = 2.0 * float(refractive_index)
        df["estimated_thickness_nm"] = df[dom.peak_col] / n_ref
        if f"{dom.peak_col}_stderr" in df.columns:
            df["estimated_thickness_stderr_nm"] = df[f"{dom.peak_col}_stderr"] / n_ref
    return df


def spectra_dataframe(analysis: BandAnalysis) -> pd.DataFrame:
    """全条件のフーリエスペクトルを 1 つの DataFrame にまとめる."""
    df = pd.DataFrame({analysis.domain.freq_col: analysis.results[0].frequency})
    for r in analysis.results:
        df[f"amplitude_{r.column}"] = r.amplitude
    return df


def summary_dict(spec: SpectrumData, analysis: BandAnalysis,
                 refractive_index: float | None = None) -> dict:
    """解析条件と結果のサマリを辞書化する (JSON 出力用)."""
    dom = analysis.domain
    reg = analysis.regression
    columns = []
    for r in analysis.results:
        entry = {
            "column": r.column,
            "condition": r.condition,
            dom.peak_col: r.estimate,
            "peak_amplitude": r.peak_amplitude,
        }
        if r.fit is not None:
            entry["fit"] = {
                "method": r.fit.method,
                "fft_peak": r.fit.fft_frequency,
                "grid_peak": r.fit.grid_frequency,
                "peak": r.fit.frequency,
                "stderr": r.fit.stderr,
                "amplitude": r.fit.amplitude,
                "phase_rad": r.fit.phase,
                "r2": r.fit.r2,
                "rmse": r.fit.rmse,
                "iterations": r.fit.iterations,
                "converged": r.fit.converged,
                "note": r.fit.note,
            }
        if dom.key == "wavenumber" and refractive_index:
            entry["estimated_thickness_nm"] = (
                r.estimate / (2.0 * float(refractive_index)))
        columns.append(entry)

    return {
        "source": spec.source,
        "domain": dom.key,
        "wavelength_range_nm": [float(spec.wavelength[0]), float(spec.wavelength[-1])],
        "delta_lambda_nm": spec.delta_lambda,
        "resampled_to_uniform_grid": spec.resampled,
        "analysis_band_nm": list(analysis.band),
        "analysis_band_actual_nm": list(analysis.band_actual),
        "n_points": analysis.n_points,
        "grid_step": analysis.step,
        "grid_step_unit": "nm" if dom.key == "wavelength" else "1/nm",
        "window": analysis.window,
        "detrend": analysis.detrend,
        "zero_pad_factor": analysis.zero_pad,
        "interpolation": analysis.interp,
        "peak_method": analysis.fit_method,
        "grid_points": analysis.grid_points or None,
        "grid_half_width_in_resolution": (GRID_HALF_WIDTH
                                          if analysis.fit_method != "none" else None),
        "refractive_index": refractive_index,
        dom.nyquist_col: analysis.nyquist,
        "resolution": analysis.resolution,
        "resolution_unit": dom.freq_unit_en,
        "columns": columns,
        "regression": None if reg is None else {
            "slope": reg.slope,
            "intercept": reg.intercept,
            "equation": reg.equation,
            "r2": reg.r2,
            "condition_at_nyquist": reg.crossing_condition,
            "condition_at_nyquist_is_extrapolated": reg.extrapolated,
        },
    }


def print_report(spec: SpectrumData, analysis: BandAnalysis,
                 refractive_index: float | None = None) -> None:
    """解析結果を標準出力に整形して表示する."""
    dom = analysis.domain
    reg = analysis.regression
    bar = "=" * 72
    print(bar)
    print(f" 解析帯域 : {analysis.band[0]:g} - {analysis.band[1]:g} nm "
          f"(実データ {analysis.band_actual[0]:g} - {analysis.band_actual[1]:g} nm, "
          f"{analysis.n_points} 点)")
    print(f" 変換の独立変数 : {dom.jp_domain}  ->  共役軸: {dom.jp_freq} "
          f"[{dom.jp_freq_unit}]")
    print(f" 窓関数   : {analysis.window}   デトレンド: {analysis.detrend}   "
          f"ゼロ詰め: x{analysis.zero_pad}"
          + (f"   補間: {analysis.interp}" if dom.key == "wavenumber" else ""))
    print(f" サンプリング間隔 {dom.jp_step} : {analysis.step:.6g}")
    print(f" ナイキスト限界      : {analysis.nyquist:.6g} {dom.jp_freq_unit}")
    print(f" 分解能              : {analysis.resolution:.6g} {dom.jp_freq_unit}")
    method_label = {
        "none": "なし (FFT のピーク位置をそのまま使用)",
        "grid": (f"FFT -> グリッドサーチ (±{GRID_HALF_WIDTH:g}×分解能 を "
                 f"{analysis.grid_points} 分割)"),
        "lm": (f"FFT -> グリッドサーチ (±{GRID_HALF_WIDTH:g}×分解能 を "
               f"{analysis.grid_points} 分割) -> Levenberg-Marquardt 法"),
    }[analysis.fit_method]
    print(f" 精密化              : {method_label}")
    print(bar)

    df = peaks_dataframe(analysis, refractive_index)
    if analysis.fit_method == "none":
        cols = ["column", "condition", dom.peak_col, "peak_amplitude"]
    else:
        cols = ["column", "condition", f"fft_{dom.peak_col}",
                f"grid_{dom.peak_col}"]
        if analysis.fit_method == "lm":
            cols += [dom.peak_col, f"{dom.peak_col}_stderr"]
        cols += ["fit_amplitude", "fit_r2"]
    if "estimated_thickness_nm" in df.columns:
        cols.append("estimated_thickness_nm")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.8g}"))

    if analysis.fit_method != "none":
        poor = df.loc[df["fit_r2"] < FIT_R2_WARN, "column"].tolist()
        notes = df.loc[df["fit_note"] != "", ["column", "fit_note"]]
        if poor or len(notes):
            print(bar)
        if poor:
            print(f" [警告] 列 {', '.join(poor)} はフィットの決定係数が "
                  f"{FIT_R2_WARN} 未満です。")
            print("        単一の正弦波モデルがデータに合っていません。縞の間隔が帯域内で")
            print("        変化している (波長ドメインで広帯域を見ている、分散が大きい)、")
            print("        複数の縞が重なっている、などが考えられます。精密化の値は")
            print("        信頼できないので、波数ドメインや帯域の分割を検討してください。")
        for _, row in notes.iterrows():
            print(f" [注意] 列 {row['column']}: {row['fit_note']}")

    near = df.loc[df["near_nyquist"], "column"].tolist()
    if near:
        print(bar)
        print(f" [警告] 列 {', '.join(near)} のピークがナイキスト限界の直下にあります。")
        print("        振幅が最大 2 倍まで過大評価される可能性があり、また真の成分が")
        print("        限界を超えて折り返している可能性もあります。サンプリング間隔を")
        print("        細かくするか、解析帯域を見直してください。")
    print(bar)
    if reg is not None:
        print(f" 回帰直線     : {reg.equation}")
        print(f" 決定係数 R^2 : {reg.r2:.6f}")
        if reg.crossing_condition is None:
            print(" ナイキスト到達条件 : 傾きが 0 のため算出不可")
        else:
            note = " (測定条件範囲外への外挿)" if reg.extrapolated else ""
            print(f" ナイキスト到達条件 : {reg.crossing_condition:.6g}{note}")
    print(bar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#: 帯域指定で区切り文字として受け付ける表記 (全角・BOM 等の揺れを吸収)
_BAND_SEPARATOR_ALIASES = ("～", "~", "－", "ー", "—", "–", "＿", "_")


def parse_band(text: str) -> tuple[float, float]:
    """'300-600' / '300:600' / '300~600' / '300,600' を (300.0, 600.0) に変換."""
    s = str(text).replace("﻿", "").strip()
    s = s.replace("nm", "").replace("ｎｍ", "")
    for alias in _BAND_SEPARATOR_ALIASES:
        s = s.replace(alias, "-")
    s = s.translate(str.maketrans("０１２３４５６７８９．", "0123456789."))
    for sep in ("-", ":", ",", " "):
        if sep in s:
            parts = [p for p in s.split(sep) if p]
            if len(parts) == 2:
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                except ValueError:
                    break
                if hi <= lo:
                    raise argparse.ArgumentTypeError(
                        f"帯域の上限は下限より大きくしてください: {text!r}")
                return lo, hi
    raise argparse.ArgumentTypeError(
        f"波長帯域は '300-600' の形式で指定してください: {text!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="干渉分光スペクトルのフーリエ変換解析ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例: python ir_ft_analysis.py -i dataset.csv -b 300-600 -w hann")
    parser.add_argument("-i", "--input", default=None,
                        help=f"入力 CSV (既定: {DEFAULT_INPUT})")
    parser.add_argument("-b", "--band", action="append", type=parse_band,
                        metavar="LO-HI",
                        help="フーリエ変換する波長帯域 (例: 300-600)。"
                             "複数回指定可。省略時は全波長域")
    parser.add_argument("-d", "--domain", default="wavelength",
                        help="FFT の独立変数: wavelength (波長 λ) / "
                             "wavenumber (波数 1/λ) / both (既定: wavelength)")
    parser.add_argument("-w", "--window", default="hann",
                        help=f"窓関数 (既定: hann)。選択肢: {', '.join(WINDOW_FUNCTIONS)}")
    parser.add_argument("--kaiser-beta", type=float, default=8.6,
                        help="カイザー窓の β (既定: 8.6)")
    parser.add_argument("--detrend", choices=["none", "mean", "linear"],
                        default="linear",
                        help="FFT 前のトレンド除去 (既定: linear)")
    parser.add_argument("--zero-pad", type=int, default=1,
                        help="ゼロ詰め倍率。>1 で共役軸を細かく補間 (既定: 1)")
    parser.add_argument("--interp", choices=["cubic", "linear"], default="cubic",
                        help="グリッド再配置の補間方式。波数ドメインで効く "
                             "(既定: cubic。linear は高周波成分の振幅が落ちる)")
    parser.add_argument("--peak-min", type=float, default=0.0,
                        help="ピーク探索の下限値 (共役軸の単位, 既定: 0 = 直流のみ除外)")
    parser.add_argument("--refine-peak", action="store_true",
                        help="放物線近似でピーク位置をサブビン精度に補正する")
    parser.add_argument("-f", "--fit", choices=list(FIT_METHODS), default="none",
                        help="FFT 結果の精密化: none (FFT のみ) / grid (グリッドサーチ) / "
                             "lm (グリッドサーチ + Levenberg-Marquardt 法)。"
                             "Quinten (2019) の手法 (既定: none)")
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS,
                        help=f"グリッドサーチの分割数 M。±{GRID_HALF_WIDTH:g}×分解能 を "
                             f"M 等分する (既定: {DEFAULT_GRID_POINTS})")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Levenberg-Marquardt 法の最大反復回数 (既定: 100)")
    parser.add_argument("--baseline-degree", type=int, default=None,
                        help="精密化モデルのベースライン多項式の次数 "
                             "(既定: --detrend linear なら 1、それ以外は 0)")
    parser.add_argument("-n", "--refractive-index", type=float, default=None,
                        help="屈折率 n。波数ドメインで膜厚 d = OPD/(2n) を推定する")
    parser.add_argument("-o", "--outdir", default="output",
                        help="出力ディレクトリ (既定: output)")
    parser.add_argument("--xmax", default="auto",
                        help="フーリエスペクトル図の横軸上限。"
                             "数値, 'auto' (ピークが見える範囲に自動調整), "
                             "'nyquist' (ナイキストまで表示) から選択 (既定: auto)")
    parser.add_argument("--no-raw-plot", action="store_true",
                        help="帯域内の干渉スペクトル図を出力しない")
    parser.add_argument("--show", action="store_true", help="グラフを画面表示する")
    parser.add_argument("--style", default="whitegrid",
                        help="seaborn のスタイル (既定: whitegrid)")
    parser.add_argument("--encoding", default=None, help="CSV のエンコーディング")
    parser.add_argument("--list-windows", action="store_true",
                        help="利用可能な窓関数を一覧表示して終了")
    parser.add_argument("--interactive", action="store_true",
                        help="帯域・窓関数・ドメインを対話的に入力する")
    return parser


def resolve_input_path(given: str | None) -> Path:
    """--input を解決する。省略時は既定の入力ファイルを使う."""
    if given:
        return Path(given)
    path = Path(DEFAULT_INPUT)
    if not path.exists():
        raise FileNotFoundError(
            f"入力 CSV {DEFAULT_INPUT} が見つかりません。--input で指定してください。")
    return path


def list_windows() -> None:
    print("利用可能な窓関数:")
    width = max(len(k) for k in WINDOW_FUNCTIONS)
    for name, (_, desc) in WINDOW_FUNCTIONS.items():
        print(f"  {name:<{width}}  {desc}")
    aliases = ", ".join(f"{k} -> {v}" for k, v in WINDOW_ALIASES.items())
    print(f"\nエイリアス: {aliases}")


def resolve_domain_arg(name: str) -> list[DomainInfo]:
    """--domain の値を DomainInfo のリストに変換する ('both' は 2 要素)."""
    key = str(name).strip().lower()
    if key in ("both", "all", "2", "両方"):
        return [DOMAINS["wavelength"], DOMAINS["wavenumber"]]
    return [resolve_domain(key)]


def prompt_settings(args: argparse.Namespace, spec: SpectrumData) -> argparse.Namespace:
    """対話モード: 波長帯域・窓関数・ドメインを入力させる."""
    wl_lo, wl_hi = float(spec.wavelength[0]), float(spec.wavelength[-1])
    print(f"\nデータ波長範囲: {wl_lo:g} - {wl_hi:g} nm "
          f"(Δλ = {spec.delta_lambda:g} nm, {spec.wavelength.size} 点)")
    print(f"条件列: {', '.join(spec.column_names)}")

    while True:
        raw = input(f"\n波長帯域を入力 (例 300-600 / 複数は '300-600, 600-900' "
                    f"/ Enter で {wl_lo:g}-{wl_hi:g}): ").strip()
        if not raw:
            break
        try:
            bands = [parse_band(t) for t in raw.replace("、", ",").split(",")
                     if t.strip()]
        except (argparse.ArgumentTypeError, ValueError) as exc:
            print(f"  入力エラー: {exc}")
            continue
        args.band = bands or None
        break

    while True:
        dom = input(f"\nFFT の独立変数 [wavelength / wavenumber / both] "
                    f"(Enter で {args.domain}): ").strip()
        if not dom:
            break
        try:
            resolve_domain_arg(dom)
        except ValueError as exc:
            print(f"  入力エラー: {exc}")
            continue
        args.domain = dom
        break

    print(f"\n窓関数: {', '.join(WINDOW_FUNCTIONS)}")
    while True:
        win = input(f"窓関数を選択 (Enter で {args.window}): ").strip()
        if not win:
            break
        try:
            args.window = resolve_window_name(win)
        except ValueError as exc:
            print(f"  入力エラー: {exc}")
            continue
        if args.window == "kaiser":
            beta = input(f"カイザー窓の β (Enter で {args.kaiser_beta}): ").strip()
            if beta:
                args.kaiser_beta = float(beta)
        break

    while True:
        fit = input(f"\n精密化 [none / grid / lm] (Enter で {args.fit}): ").strip().lower()
        if not fit:
            break
        if fit not in FIT_METHODS:
            print(f"  入力エラー: {fit!r} ({' / '.join(FIT_METHODS)} から選択)")
            continue
        args.fit = fit
        break
    print()
    return args


def run_band(spec: SpectrumData, band: tuple[float, float], domain: DomainInfo,
             args: argparse.Namespace, outdir: Path) -> BandAnalysis:
    """1 帯域・1 ドメイン分の解析・保存・描画をまとめて実行する."""
    analysis = analyze_band(spec, band[0], band[1], window=args.window,
                            detrend=args.detrend, zero_pad=args.zero_pad,
                            kaiser_beta=args.kaiser_beta,
                            peak_min=args.peak_min, refine_peak=args.refine_peak,
                            domain=domain, interp=args.interp, fit=args.fit,
                            grid_points=args.grid_points, max_iter=args.max_iter,
                            baseline_degree=args.baseline_degree)
    fit_regression(analysis)
    n_ref = args.refractive_index
    print_report(spec, analysis, n_ref)

    prefix = outdir / analysis.label
    peaks_dataframe(analysis, n_ref).to_csv(f"{prefix}_peaks.csv", index=False,
                                            encoding="utf-8-sig")
    spectra_dataframe(analysis).to_csv(f"{prefix}_fourier_spectrum.csv",
                                       index=False, encoding="utf-8-sig")
    with open(f"{prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_dict(spec, analysis, n_ref), fh,
                  ensure_ascii=False, indent=2)

    plot_fourier_spectra(analysis, Path(f"{prefix}_fourier_spectrum.png"),
                         xmax=args.xmax)
    plot_peak_scatter(analysis, Path(f"{prefix}_peak_regression.png"))
    if analysis.fit_method != "none":
        plot_fit_check(analysis, Path(f"{prefix}_fit_check.png"))
    if not args.no_raw_plot:
        plot_raw_spectra(spec, analysis, Path(f"{prefix}_raw_spectra.png"))

    print(f" 出力先: {prefix}_*.csv / *.json / *.png\n")
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    if args.list_windows:
        list_windows()
        return 0

    sns.set_theme(style=args.style, context="notebook")

    spec = load_spectra(resolve_input_path(args.input), encoding=args.encoding)
    print(f"読み込み: {spec.source}")
    print(f"  波長 {spec.wavelength[0]:g} - {spec.wavelength[-1]:g} nm / "
          f"{spec.wavelength.size} 点 / Δλ = {spec.delta_lambda:g} nm")
    print(f"  条件列 {len(spec.column_names)} 本: {', '.join(spec.column_names)}")
    print(f"  A 列から導出した波長軸 FFT のナイキスト周波数: "
          f"{spec.nyquist_per_100nm:.6g} 波数/100nm\n")

    # 引数なしで起動された場合も対話モードに入る
    if args.interactive or not argv:
        args = prompt_settings(args, spec)

    args.window = resolve_window_name(args.window)
    domains = resolve_domain_arg(args.domain)
    bands = args.band or [(float(spec.wavelength[0]), float(spec.wavelength[-1]))]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for band in bands:
        for domain in domains:
            run_band(spec, band, domain, args, outdir)

    if args.show:
        plt.show()
    else:
        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
