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
    9. 同論文のその他の手法
       - PSD に番号の 2 乗を掛けた重み付きピーク探索 (--peak-weight m2, 第 4 章)
       - 屈折率の分散を考慮した有効屈折率と最小膜厚 (式 6, 7)
       - 隣接する極大の位置から膜厚を求める方法 (--fringe, 式 8)
       - 多重反射の反射率モデルによるフィット (--fit-model reflectance, 式 1, 15)

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
# 屈折率モデル
# ---------------------------------------------------------------------------
#
# 複素屈折率は N = n - i k の符号で扱う (k > 0 が吸収)。反射率モデルで
# 位相因子を exp(-i 4π N d / λ) と書くと、この符号で振幅が exp(-4π k d / λ)
# に減衰し、論文の式 (16) の E と一致する。

@dataclass
class RefractiveIndex:
    """波長 λ [nm] の関数としての複素屈折率 N(λ) = n(λ) - i k(λ)."""

    kind: str                                   # "constant" / "cauchy" / "table"
    description: str                            # 表示用の説明
    func: Callable[[np.ndarray], np.ndarray]    # λ [nm] -> 複素屈折率
    wl_range: tuple[float, float] | None = None # テーブルの有効範囲 [nm]

    def __call__(self, wl: np.ndarray | float) -> np.ndarray:
        wl = np.asarray(wl, dtype=float)
        if self.wl_range is not None:
            lo, hi = self.wl_range
            if wl.min() < lo - 1e-9 or wl.max() > hi + 1e-9:
                raise ValueError(
                    f"屈折率テーブルの範囲 {lo:g}-{hi:g} nm の外の波長 "
                    f"({wl.min():g}-{wl.max():g} nm) が必要です。解析帯域を狭めてください。")
        return np.asarray(self.func(wl), dtype=complex)

    def n(self, wl: np.ndarray | float) -> np.ndarray:
        """屈折率の実部 n(λ)."""
        return np.real(self(wl))

    def k(self, wl: np.ndarray | float) -> np.ndarray:
        """消衰係数 k(λ) (吸収がなければ 0)."""
        return -np.imag(self(wl))


def constant_index(n: float, k: float = 0.0) -> RefractiveIndex:
    """波長によらない一定の屈折率."""
    value = complex(float(n), -float(k))
    desc = f"一定 n = {n:g}" + (f", k = {k:g}" if k else "")
    return RefractiveIndex("constant", desc,
                           lambda wl: np.full(np.shape(wl), value, dtype=complex))


def cauchy_index(coeffs: Sequence[float]) -> RefractiveIndex:
    """Cauchy の分散式 n(λ) = A + B/λ² + C/λ⁴ (λ は µm)."""
    c = [float(v) for v in coeffs]
    if not 1 <= len(c) <= 3:
        raise ValueError("Cauchy の係数は A[,B[,C]] の 1〜3 個で指定してください。")
    c += [0.0] * (3 - len(c))
    a, b, cc = c

    def func(wl: np.ndarray) -> np.ndarray:
        um2 = (np.asarray(wl, dtype=float) / 1000.0) ** 2
        return (a + b / um2 + cc / um2 ** 2).astype(complex)

    desc = f"Cauchy n = {a:g} + {b:g}/λ² + {cc:g}/λ⁴ (λ: µm)"
    return RefractiveIndex("cauchy", desc, func)


def table_index(path: str | Path) -> RefractiveIndex:
    """CSV の屈折率テーブル (1 列目: 波長 nm, 2 列目: n, 3 列目: k [任意]).

    refractiveindex.info などから取得したデータを想定し、テーブルの点の間は
    線形補間する。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"屈折率テーブルが見つかりません: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError("屈折率テーブルには波長 [nm] と n の 2 列以上が必要です。")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=df.columns[:2])
    df = df.sort_values(df.columns[0])
    wl = df.iloc[:, 0].to_numpy(dtype=float)
    n = df.iloc[:, 1].to_numpy(dtype=float)
    k = (df.iloc[:, 2].fillna(0.0).to_numpy(dtype=float)
         if df.shape[1] >= 3 else np.zeros_like(n))
    if wl.size < 2:
        raise ValueError("屈折率テーブルの有効な行が 2 行未満です。")

    def func(x: np.ndarray) -> np.ndarray:
        return np.interp(x, wl, n) - 1j * np.interp(x, wl, k)

    return RefractiveIndex("table", f"テーブル {path.name} ({wl[0]:g}-{wl[-1]:g} nm)",
                           func, wl_range=(float(wl[0]), float(wl[-1])))


def as_index(value: "RefractiveIndex | float | None") -> RefractiveIndex | None:
    """数値なら一定の屈折率に変換し、モデルならそのまま返す."""
    if value is None or isinstance(value, RefractiveIndex):
        return value
    return constant_index(float(value))


def effective_index(index: RefractiveIndex, wl_min: float, wl_max: float) -> float:
    """分散を考慮した有効屈折率 (論文の式 7).

        n_eff = (n(λmin)/λmin - n(λmax)/λmax) / (1/λmin - 1/λmax)

    干渉の位相は 4π n(λ) d / λ なので、帯域の両端の位相差は
    4π d (n(λmin)/λmin - n(λmax)/λmax) になる。1/λ に対する FFT はこれを
    一定の屈折率 n_eff の膜とみなして光路差 2 n_eff d を返す。
    一定の屈折率なら n_eff = n。
    """
    n1, n2 = index.n(np.array([wl_min, wl_max]))
    return float((n1 / wl_min - n2 / wl_max) / (1.0 / wl_min - 1.0 / wl_max))


def group_index_range(index: RefractiveIndex, wl_min: float, wl_max: float,
                      samples: int = 512) -> tuple[float, float]:
    """帯域内の群屈折率 n_g = d(n/λ)/d(1/λ) = n - λ dn/dλ の最小値と最大値.

    1/λ に対する縞の局所的な間隔は 1/(2 n_g d) で決まる。帯域内で n_g が
    変わると縞の間隔が変わり、FFT のピークが広がる (論文の第 4 章)。
    """
    wl = np.linspace(wl_min, wl_max, samples)
    sigma = 1.0 / wl
    ng = np.gradient(index.n(wl) * sigma, sigma)
    return float(ng.min()), float(ng.max())


def parse_cauchy(text: str) -> list[float]:
    """'1.56,0.005' や '1.56 0.005 0' を Cauchy 係数のリストに変換する."""
    parts = [p for p in str(text).replace(",", " ").split() if p]
    try:
        coeffs = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Cauchy の係数は 'A,B[,C]' の数値で指定してください: {text!r}") from exc
    if not 1 <= len(coeffs) <= 3:
        raise argparse.ArgumentTypeError(
            f"Cauchy の係数は 1〜3 個で指定してください: {text!r}")
    return coeffs


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
    model_name: str = "sinusoid"      # "sinusoid" または "reflectance"
    thickness: float | None = None    # 膜厚 [nm] (屈折率が与えられたとき)
    thickness_stderr: float | None = None
    scale: float | None = None        # 反射率モデルの倍率 (理想は 1)
    grid_points_used: int = 0         # 実際に調べたグリッド点の数
    predict: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """任意の独立変数でフィット後のモデル値を計算する (作図用)."""
        if self.predict is not None:
            return self.predict(np.asarray(x, dtype=float))
        u = (np.asarray(x, dtype=float) - self.x_center) / self.half_width
        return _sinusoid_design(u, self.params[0], self.degree) @ self.params[1:]


@dataclass
class FringeResult:
    """隣接する極大の位置から求めた膜厚 (論文の式 8)."""

    thickness: float                  # 各ペアの膜厚の平均 [nm]
    sem: float                        # 平均の標準誤差 [nm]
    std: float                        # 各ペアのばらつき (標準偏差) [nm]
    pairs: int                        # 使った隣接極大のペア数
    points_per_fringe: float          # 縞 1 周期あたりのデータ点数
    reliable: bool                    # 点数が十分で信頼できるか
    maxima_wavelengths: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    per_pair: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


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
    fringe: FringeResult | None = None  # 式 8 による膜厚 (--fringe のとき)

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
    peak_weight: str = "none"          # ピーク探索の重み (none / m2)
    fit_model: str = "sinusoid"        # 精密化のモデル (sinusoid / reflectance)
    index: RefractiveIndex | None = None      # 膜の屈折率モデル
    substrate: RefractiveIndex | None = None  # 基板の屈折率モデル
    n_band: tuple[float, float] | None = None # 帯域両端の屈折率 n(λmin), n(λmax)
    n_eff: float | None = None                # 有効屈折率 (式 7)
    group_index: tuple[float, float] | None = None  # 帯域内の群屈折率の範囲
    results: list[ColumnFFTResult] = field(default_factory=list)
    regression: RegressionResult | None = None

    @property
    def label(self) -> str:
        lo, hi = self.band
        return f"band_{lo:g}-{hi:g}nm_{self.domain.suffix}"

    @property
    def thickness_resolution(self) -> float | None:
        """膜厚の分解能 d_min = 光路差の分解能 / (2 n_eff) (論文の式 6)."""
        return None if self.n_eff is None else self.resolution / (2.0 * self.n_eff)

    @property
    def thickness_max(self) -> float | None:
        """FFT で求められる最大膜厚 = ナイキスト限界 / (2 n_eff)."""
        return None if self.n_eff is None else self.nyquist / (2.0 * self.n_eff)


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

#: 精密化に使うモデル
FIT_MODELS = ("sinusoid", "reflectance")

#: ピーク探索の重み (論文の第 4 章)
PEAK_WEIGHTS = ("none", "m2")

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


def _sinusoid_evaluator(u: np.ndarray, degree: int
                        ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """正弦波 + ベースラインのモデル値とヤコビアンを返す関数を作る.

    パラメータ p = [g, b_0, ..., b_deg, A_c, A_s]。
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

    return evaluate


def levenberg_marquardt(y: np.ndarray,
                        evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
                        p0: np.ndarray, max_iter: int = 100,
                        ftol: float = 1e-12, xtol: float = 1e-12
                        ) -> tuple[np.ndarray, float, np.ndarray, int, bool]:
    """Levenberg-Marquardt 法による非線形最小二乗フィット (論文の手順 2).

    evaluate(p) はモデル値とヤコビアン (各パラメータでの偏微分) を返す関数。
    正弦波モデルと反射率モデルの両方でこの関数を共用している。

    各反復で (JᵀJ + λ·diag(JᵀJ)) δ = Jᵀr を解いて p を更新する。
    λ が小さいとガウス・ニュートン法 (速いが初期値が悪いと発散しやすい)、
    大きいと最急降下法 (遅いが確実に下る) に近づく。χ² が下がれば λ を
    1/10 にして大胆に、下がらなければ 10 倍にして慎重に進む。

    戻り値は (最終パラメータ, χ², ヤコビアン, 反復回数, 収束したか)。
    """
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
            y, _sinusoid_evaluator(u, degree), params, max_iter=max_iter)
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
        grid_points_used=int(gs.size),
    )


# ---------------------------------------------------------------------------
# 多重反射の反射率モデル (論文の式 1, 15)
# ---------------------------------------------------------------------------

def layer_reflectance(wl: np.ndarray, d: float, n_film: np.ndarray,
                      n_sub: np.ndarray, n_amb: complex = 1.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """基板上の単層膜の反射率と、その膜厚 d による微分を返す.

    多重反射をすべて足し合わせた振幅反射率 (Airy の式):

        r = (r01 + r12 E) / (1 + r01 r12 E),   E = exp(-i 4π N d / λ)
        r01 = (N0 - N) / (N0 + N),   r12 = (N - Ns) / (N + Ns)

    反射率は R = |r|²。屈折率が実数なら論文の式 (1) と一致し、
    吸収 (N = n - ik) があれば E の大きさが exp(-4π k d / λ) となって
    式 (15), (16) と一致する。式 (1) は √(R01 R12) を使うため r01 r12 の
    符号 (反射時の位相反転) を区別できないが、ここでは符号付きで扱う。
    """
    wl = np.asarray(wl, dtype=float)
    r01 = (n_amb - n_film) / (n_amb + n_film)
    r12 = (n_film - n_sub) / (n_film + n_sub)
    rate = 4.0 * np.pi * n_film / wl                # 位相の膜厚微分 (複素)
    e = np.exp(-1j * rate * d)
    den = 1.0 + r01 * r12 * e
    r = (r01 + r12 * e) / den
    dr_dd = r12 * (1.0 - r01 ** 2) / den ** 2 * (-1j * rate * e)
    reflectance = np.abs(r) ** 2
    d_reflectance = 2.0 * np.real(np.conj(r) * dr_dd)
    return reflectance, d_reflectance


def refine_reflectance_fit(x: np.ndarray, y: np.ndarray, d_start: float,
                           d_min: float, index: RefractiveIndex,
                           substrate: RefractiveIndex, n_eff: float,
                           method: str = "lm", degree: int = 0,
                           n_grid: int = DEFAULT_GRID_POINTS, max_iter: int = 100,
                           d_margin: float = 0.0) -> FitResult:
    """反射率モデルでグリッドサーチと LM 法を行い、膜厚を直接求める.

    論文の第 5 章の手順そのもの。x は 1/λ [1/nm]、d_start は FFT から求めた
    膜厚 (光路差 / 2 n_eff)、d_min は膜厚の分解能 (式 6)。

    測定値 ≈ s × R(λ; d) + ベースライン多項式 とし、倍率 s とベースラインは
    線形パラメータとして最小二乗で消去する。s は参照試料の校正のずれや、
    パターン付き試料で膜が占める面積の割合 (論文の式 9) を吸収する。

    モデルは位相 4π n d / λ の絶対値まで使うため、χ² は膜厚に対して
    λ/(2n) 程度の間隔で細かい谷が並ぶ。論文の M = 16 では刻みが粗すぎて
    隣の谷に落ちることがあるので、刻みが λmin / (8 n) 以下になるよう
    グリッドを自動で細かくする。屈折率の分散で FFT のピークがずれる分は
    d_margin として探索範囲に足す。
    """
    if method not in ("grid", "lm"):
        raise ValueError(f"未知の精密化方式: {method!r} (grid/lm)")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wl = 1.0 / x
    n_film = index(wl)
    n_sub = substrate(wl)
    x_c = 0.5 * (x.min() + x.max())
    half = 0.5 * (x.max() - x.min())
    u = (x - x_c) / half
    powers = np.column_stack([u ** k for k in range(degree + 1)])

    def solve(d: float) -> tuple[np.ndarray, float, np.ndarray]:
        refl, _ = layer_reflectance(wl, d, n_film, n_sub)
        design = np.column_stack([refl, powers])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        return coef, float(resid @ resid), refl

    # --- 手順 1: グリッドサーチ --------------------------------------------
    half_range = GRID_HALF_WIDTH * d_min + d_margin
    step_target = float(wl.min() / (8.0 * np.real(n_film).max()))
    n_pts = max(int(n_grid), int(np.ceil(2.0 * half_range / step_target)))
    ds = d_start + np.linspace(-half_range, half_range, n_pts + 1)
    ds = ds[ds > 0.0]
    if ds.size == 0:
        raise ValueError("反射率モデルの探索範囲が空です。")
    chi2s = np.array([solve(d)[1] for d in ds])
    k = int(np.argmin(chi2s))
    d_grid = float(ds[k])
    coef, chi2, _ = solve(d_grid)
    d_best = d_grid
    params = np.concatenate([[1.0], coef])      # 膜厚は d_grid に対する比で持つ

    iterations, converged, stderr_d, note = 0, True, None, ""
    if method == "lm":
        def evaluate(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            refl, drefl = layer_reflectance(wl, p[0] * d_grid, n_film, n_sub)
            model = p[1] * refl + powers @ p[2:]
            jac = np.empty((y.size, p.size))
            jac[:, 0] = p[1] * drefl * d_grid          # ∂y/∂(d/d_grid)
            jac[:, 1] = refl                           # ∂y/∂s
            jac[:, 2:] = powers                        # ∂y/∂b_k
            return model, jac

        p_lm, chi2_lm, jac, iterations, converged = levenberg_marquardt(
            y, evaluate, params, max_iter=max_iter)
        d_lm = float(p_lm[0] * d_grid)
        if d_lm <= 0.0 or abs(d_lm - d_start) > half_range + 2.0 * step_target:
            converged = False
            note = "LM 法が探索範囲の外へ移動したため、グリッドサーチの結果を採用"
        else:
            params, chi2, d_best = p_lm, chi2_lm, d_lm
            dof = max(y.size - params.size, 1)
            try:
                cov = np.linalg.inv(jac.T @ jac) * (chi2 / dof)
                stderr_d = float(np.sqrt(max(cov[0, 0], 0.0))) * d_grid
            except np.linalg.LinAlgError:
                stderr_d = float("nan")
            if not converged:
                note = f"LM 法が {max_iter} 回で収束しなかった"

    scale = float(params[1])
    base = params[2:].copy()
    refl_best, _ = layer_reflectance(wl, d_best, n_film, n_sub)
    model = scale * refl_best + powers @ base
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if scale < 0.0:
        note = (note + "; " if note else "") + (
            "倍率が負: 基板や膜の屈折率の大小関係が実際と逆の可能性")

    def predict(xq: np.ndarray, _d: float = d_best) -> np.ndarray:
        wq = 1.0 / np.asarray(xq, dtype=float)
        uq = (np.asarray(xq, dtype=float) - x_c) / half
        rq, _ = layer_reflectance(wq, _d, index(wq), substrate(wq))
        return scale * rq + np.column_stack([uq ** j for j in range(degree + 1)]) @ base

    opd = 2.0 * n_eff
    return FitResult(
        method=method,
        fft_frequency=opd * float(d_start),
        grid_frequency=opd * d_grid,
        frequency=opd * d_best,
        stderr=None if stderr_d is None else opd * stderr_d,
        amplitude=0.5 * float(np.ptp(scale * refl_best)),
        phase=float("nan"),
        rmse=float(np.sqrt(chi2 / y.size)),
        r2=1.0 - chi2 / ss_tot if ss_tot > 0 else float("nan"),
        iterations=int(iterations),
        converged=bool(converged),
        note=note,
        grid_frequencies=opd * ds,
        grid_chi2=chi2s,
        x=x,
        y=y,
        model=model,
        params=np.asarray(params, dtype=float),
        x_center=float(x_c),
        half_width=float(half),
        degree=int(degree),
        model_name="reflectance",
        thickness=d_best,
        thickness_stderr=stderr_d,
        scale=scale,
        grid_points_used=int(ds.size),
        predict=predict,
    )


# ---------------------------------------------------------------------------
# 隣接する極大から膜厚を求める (論文の式 8)
# ---------------------------------------------------------------------------

#: 式 8 の結果を信頼できるとみなす、縞 1 周期あたりの最小データ点数
FRINGE_MIN_POINTS = 4.0


def _parabola_vertex(x3: np.ndarray, y3: np.ndarray) -> float:
    """不等間隔の 3 点を通る放物線の頂点の x 座標."""
    (x0, x1, x2), (y0, y1, y2) = x3, y3
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
    if a >= 0.0:                                   # 上に凸でなければ中央の点
        return float(x1)
    return float(np.clip(-b / (2.0 * a), x0, x2))


def fringe_thickness(wl: np.ndarray, y: np.ndarray, index: RefractiveIndex,
                     opd_hint: float) -> FringeResult | None:
    """隣接する反射率の極大の位置から膜厚を求める (論文の式 8).

        d = 1 / (2 (n(λ1)/λ1 - n(λ2)/λ2))     λ1 < λ2 は隣り合う極大の波長

    隣り合う極大のあいだで位相 4π n d / λ がちょうど 2π 変わることを使う。
    FFT と違って各点で屈折率 n(λ) をそのまま使うので、分散が強くて FFT が
    使えないときの代わりになる。1 組だけでは誤差が大きいので、帯域内の
    すべての隣接ペアについて求めて平均する。opd_hint (FFT の光路差) は
    極大を探すときの目安の間隔にだけ使う。
    """
    if opd_hint <= 0.0:
        return None
    sigma = 1.0 / np.asarray(wl, dtype=float)
    order = np.argsort(sigma)
    sigma, yy = sigma[order], np.asarray(y, dtype=float)[order]

    # 緩やかなベースラインを 2 次式で除いてから極大を探す
    u = (sigma - sigma.mean()) / (0.5 * np.ptp(sigma))
    yd = yy - np.polyval(np.polyfit(u, yy, 2), u)

    period = 1.0 / opd_hint                        # 1/λ での縞の間隔の目安
    points = period / float(np.median(np.diff(sigma)))
    cand = np.where((yd[1:-1] > yd[:-2]) & (yd[1:-1] >= yd[2:]) & (yd[1:-1] > 0.0))[0] + 1

    # 目安の間隔の 6 割より近い極大は、高い方だけを残す (ノイズによる偽の極大を除く)
    kept: list[int] = []
    for i in cand[np.argsort(-yd[cand])]:
        if all(abs(sigma[i] - sigma[j]) >= 0.6 * period for j in kept):
            kept.append(int(i))
    kept.sort()
    if len(kept) < 2:
        return None
    peaks = np.array([_parabola_vertex(sigma[i - 1:i + 2], yd[i - 1:i + 2]) for i in kept])

    d_pairs = []
    for s_lo, s_hi in zip(peaks[:-1], peaks[1:]):
        gap = s_hi - s_lo
        if not 0.6 * period <= gap <= 1.6 * period:  # 極大の取りこぼしがある区間は使わない
            continue
        n_lo, n_hi = index.n(np.array([1.0 / s_lo, 1.0 / s_hi]))
        d_pairs.append(1.0 / (2.0 * (n_hi * s_hi - n_lo * s_lo)))
    if not d_pairs:
        return None
    d_pairs = np.asarray(d_pairs)
    std = float(np.std(d_pairs, ddof=1)) if d_pairs.size > 1 else float("nan")
    return FringeResult(
        thickness=float(d_pairs.mean()),
        sem=std / np.sqrt(d_pairs.size) if d_pairs.size > 1 else float("nan"),
        std=std,
        pairs=int(d_pairs.size),
        points_per_fringe=float(points),
        reliable=bool(points >= FRINGE_MIN_POINTS and d_pairs.size >= 3),
        maxima_wavelengths=1.0 / peaks[::-1],
        per_pair=d_pairs,
    )


def _climb_to_local_max(amp: np.ndarray, idx: int, allowed: np.ndarray) -> int:
    """振幅が増える方向へ隣の点をたどり、最寄りの極大に移る."""
    while True:
        left = idx - 1 if idx - 1 >= 0 and allowed[idx - 1] else None
        right = idx + 1 if idx + 1 < amp.size and allowed[idx + 1] else None
        best = idx
        for j in (left, right):
            if j is not None and amp[j] > amp[best]:
                best = j
        if best == idx:
            return idx
        idx = best


def analyze_band(spec: SpectrumData, lo: float, hi: float, window: str = "hann",
                 detrend: str = "linear", zero_pad: int = 1,
                 kaiser_beta: float = 8.6, peak_min: float = 0.0,
                 refine_peak: bool = False,
                 domain: str | DomainInfo = "wavelength",
                 interp: str = "cubic", fit: str = "none",
                 grid_points: int = DEFAULT_GRID_POINTS,
                 max_iter: int = 100,
                 baseline_degree: int | None = None,
                 peak_weight: str = "none",
                 index: RefractiveIndex | float | None = None,
                 substrate: RefractiveIndex | float | None = None,
                 fit_model: str = "sinusoid",
                 fringe: bool = False) -> BandAnalysis:
    """指定帯域・指定ドメインで全条件列の FFT とピーク抽出を行う.

    fit が "grid" または "lm" なら、FFT のピークを出発点に精密化も行う。
    baseline_degree はフィットのベースライン多項式の次数で、None なら
    デトレンド方式に合わせる (linear なら 1 次、それ以外は 0 次)。

    peak_weight="m2" なら、パワースペクトルに番号の 2 乗を掛けて
    ピークを探す (論文の第 4 章)。

    以下は波数ドメインで膜の屈折率 index を与えたときだけ有効:
      - 有効屈折率 n_eff (式 7) と膜厚の分解能 d_min (式 6) を求める
      - fit_model="reflectance" なら多重反射の反射率モデルで膜厚をフィットする
        (式 1, 15)。substrate は基板の屈折率 (既定 1.0 = 自立膜)
      - fringe=True なら隣接する極大の位置から膜厚を求める (式 8)
    """
    if fit not in FIT_METHODS:
        raise ValueError(f"未知の精密化方式: {fit!r} ({'/'.join(FIT_METHODS)})")
    if peak_weight not in PEAK_WEIGHTS:
        raise ValueError(f"未知の重み: {peak_weight!r} ({'/'.join(PEAK_WEIGHTS)})")
    if fit_model not in FIT_MODELS:
        raise ValueError(f"未知のモデル: {fit_model!r} ({'/'.join(FIT_MODELS)})")
    dom = domain if isinstance(domain, DomainInfo) else resolve_domain(domain)
    index = as_index(index)
    substrate = as_index(substrate)
    if fit_model == "reflectance":
        if dom.key != "wavenumber":
            raise ValueError("反射率モデルは波数ドメイン (--domain wavenumber) でのみ使えます。")
        if index is None:
            raise ValueError("反射率モデルには膜の屈折率 (-n / --cauchy / --n-table) が必要です。")
        if fit == "none":
            raise ValueError("反射率モデルを使うには --fit grid または lm を指定してください。")
        if substrate is None:
            substrate = constant_index(1.0)
    if fringe and (dom.key != "wavenumber" or index is None):
        raise ValueError("式 8 の膜厚算出は、波数ドメインで膜の屈折率を与えたときだけ使えます。")

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
        peak_weight=peak_weight,
        fit_model=fit_model,
        index=index if dom.key == "wavenumber" else None,
        substrate=substrate if fit_model == "reflectance" else None,
    )

    # 波数ドメインで屈折率があれば、分散を考慮した有効屈折率を求める (式 6, 7)
    if analysis.index is not None:
        wl_lo, wl_hi = band_actual
        analysis.n_band = tuple(float(v) for v in index.n(np.array([wl_lo, wl_hi])))
        analysis.n_eff = effective_index(index, wl_lo, wl_hi)
        analysis.group_index = group_index_range(index, wl_lo, wl_hi)

    if fit != "none" or fringe:
        x_fit, y_fit = fit_samples(spec, lo, hi, dom)
    if fit != "none":
        default_degree = BASELINE_DEGREE.get(detrend, 1)
        if fit_model == "reflectance":
            default_degree = 0                    # 物理モデルなので一定のずれだけ許す
        degree = int(baseline_degree) if baseline_degree is not None else default_degree
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
        score = amp ** 2
        if peak_weight == "m2":
            # 論文の第 4 章: パワースペクトルに番号 m の 2 乗を掛けて探す。
            # 高い位置にある膜のピークを強め、低い位置の偽のピークを弱める。
            score = score * (freq / analysis.resolution) ** 2
        idx = int(np.argmax(np.where(search, score, -np.inf)))
        if peak_weight == "m2":
            # 重みで山の肩が選ばれることがあるので、元の振幅の極大まで移る
            idx = _climb_to_local_max(amp, idx, search)
        pk_freq, pk_amp = (parabolic_peak(freq, amp, idx) if refine_peak
                           else (float(freq[idx]), float(amp[idx])))

        fit_result = None
        if fit != "none" and fit_model == "reflectance":
            d_start = pk_freq / (2.0 * analysis.n_eff)
            ng_lo, ng_hi = analysis.group_index
            # 分散で縞の間隔が帯域内で変わると、FFT のピークは真の膜厚から
            # 最大 d × (n_g の幅) / n_eff ずれる。その分だけ探索範囲を広げる
            margin = d_start * (ng_hi - ng_lo) / analysis.n_eff
            fit_result = refine_reflectance_fit(
                x_fit, y_fit[:, j], d_start, analysis.thickness_resolution,
                index, substrate, analysis.n_eff, method=fit, degree=degree,
                n_grid=grid_points, max_iter=max_iter, d_margin=margin)
        elif fit != "none":
            fit_result = refine_peak_fit(
                x_fit, y_fit[:, j], pk_freq, analysis.resolution, dom.unit_scale,
                method=fit, degree=degree, n_grid=grid_points, max_iter=max_iter)
            if analysis.n_eff is not None:
                fit_result.thickness = fit_result.frequency / (2.0 * analysis.n_eff)
                if fit_result.stderr is not None:
                    fit_result.thickness_stderr = fit_result.stderr / (2.0 * analysis.n_eff)

        fringe_result = None
        if fringe:
            opd_hint = fit_result.frequency if fit_result is not None else pk_freq
            fringe_result = fringe_thickness(1.0 / x_fit, y_fit[:, j], index, opd_hint)

        analysis.results.append(ColumnFFTResult(
            column=name,
            condition=float(spec.conditions[j]),
            frequency=freq,
            amplitude=amp,
            peak_frequency=pk_freq,
            peak_amplitude=pk_amp,
            peak_index=idx,
            fit=fit_result,
            fringe=fringe_result,
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
        f"domain: {dom.key}, window: {analysis.window}, N = {analysis.n_points}"
        + (", peak search: PSD x m$^2$" if analysis.peak_weight == "m2" else "") + ")")
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
        title = f"condition {res.column}: {fr.frequency:.6g} {dom.freq_unit_en}"
        if fr.thickness is not None:
            title += f"  (d = {fr.thickness:.6g} nm)"
        ax.set_title(title, fontsize=9)
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
                 f"model: {analysis.fit_model}, "
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
        df["fit_model"] = analysis.fit_model
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
        if analysis.fit_model == "reflectance":
            df["fit_scale"] = [r.fit.scale for r in res]
        df["fit_note"] = [r.fit.note for r in res]

    # 波数ドメインでは光路差から膜厚 d = OPD / (2 n_eff) を推定できる (式 7)
    n_eff = _effective_index_for_output(analysis, refractive_index)
    if n_eff is not None:
        df["estimated_thickness_nm"] = df[dom.peak_col] / (2.0 * n_eff)
        if f"{dom.peak_col}_stderr" in df.columns:
            df["estimated_thickness_stderr_nm"] = (
                df[f"{dom.peak_col}_stderr"] / (2.0 * n_eff))

    # 隣接する極大から求めた膜厚 (式 8)
    if any(r.fringe is not None for r in res):
        def pick(attr):
            return [getattr(r.fringe, attr) if r.fringe is not None else np.nan
                    for r in res]
        df["fringe_thickness_nm"] = pick("thickness")
        df["fringe_thickness_sem_nm"] = pick("sem")
        df["fringe_pairs"] = [r.fringe.pairs if r.fringe is not None else 0 for r in res]
        df["fringe_reliable"] = [bool(r.fringe.reliable) if r.fringe is not None else False
                                 for r in res]
    return df


def _effective_index_for_output(analysis: BandAnalysis,
                                refractive_index: float | None = None) -> float | None:
    """出力に使う有効屈折率. 解析時に屈折率モデルがあればそれを優先する."""
    if analysis.domain.key != "wavenumber":
        return None
    if analysis.n_eff is not None:
        return analysis.n_eff
    if refractive_index:
        return float(refractive_index)
    return None


def absorption_thickness_limit(analysis: BandAnalysis) -> tuple[float, float] | None:
    """吸収のある膜で測定できる最大膜厚の目安 d_m ≈ (0.4〜0.75) λ / k (論文の式 17).

    吸収が強いと裏面からの反射が減衰し、縞の振幅が雑音に埋もれる。
    論文は SNR 1000〜5000 の検出器で係数 0.4〜0.75 としている。
    帯域内で λ/k が最も小さい波長が制約になる。吸収がなければ None。
    """
    if analysis.index is None:
        return None
    wl = np.linspace(*analysis.band_actual, 256)
    k = analysis.index.k(wl)
    if not np.any(k > 0.0):
        return None
    ratio = float(np.min(wl[k > 0.0] / k[k > 0.0]))
    return 0.4 * ratio, 0.75 * ratio


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
                "model": r.fit.model_name,
                "method": r.fit.method,
                "thickness_nm": r.fit.thickness,
                "thickness_stderr_nm": r.fit.thickness_stderr,
                "scale": r.fit.scale,
                "grid_points_used": r.fit.grid_points_used,
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
        n_eff = _effective_index_for_output(analysis, refractive_index)
        if n_eff is not None:
            entry["estimated_thickness_nm"] = r.estimate / (2.0 * n_eff)
        if r.fringe is not None:
            entry["fringe"] = {
                "thickness_nm": r.fringe.thickness,
                "sem_nm": r.fringe.sem,
                "std_nm": r.fringe.std,
                "pairs": r.fringe.pairs,
                "points_per_fringe": r.fringe.points_per_fringe,
                "reliable": r.fringe.reliable,
            }
        columns.append(entry)

    n_eff_out = _effective_index_for_output(analysis, refractive_index)
    limit = absorption_thickness_limit(analysis)
    thickness_info = None if n_eff_out is None else {
        "film_index": (analysis.index.description if analysis.index is not None
                       else f"一定 n = {refractive_index:g}"),
        "n_at_band_edges": list(analysis.n_band) if analysis.n_band else None,
        "n_eff_eq7": n_eff_out,
        "group_index_range": list(analysis.group_index) if analysis.group_index else None,
        "thickness_resolution_nm_eq6": analysis.resolution / (2.0 * n_eff_out),
        "thickness_max_nm": analysis.nyquist / (2.0 * n_eff_out),
        "substrate_index": (analysis.substrate.description
                            if analysis.substrate is not None else None),
        "absorption_thickness_limit_nm_eq17": list(limit) if limit else None,
    }

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
        "peak_weight": analysis.peak_weight,
        "peak_method": analysis.fit_method,
        "fit_model": analysis.fit_model if analysis.fit_method != "none" else None,
        "thickness": thickness_info,
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
    if analysis.fit_method != "none":
        model_label = {"sinusoid": "正弦波 + ベースライン",
                       "reflectance": "多重反射の反射率 (式 1, 15)"}[analysis.fit_model]
        method_label += f"  [モデル: {model_label}]"
    print(f" 精密化              : {method_label}")
    if analysis.peak_weight == "m2":
        print(" ピーク探索          : パワースペクトル × m² (論文の第 4 章)")

    # 膜厚に関する情報 (波数ドメインで屈折率があるとき)
    n_eff = _effective_index_for_output(analysis, refractive_index)
    if n_eff is not None:
        desc = (analysis.index.description if analysis.index is not None
                else f"一定 n = {refractive_index:g}")
        print(f" 膜の屈折率          : {desc}")
        if analysis.n_band is not None and analysis.index.kind != "constant":
            print(f" n(λmin), n(λmax)    : {analysis.n_band[0]:.6g}, {analysis.n_band[1]:.6g}"
                  f"   -> 有効屈折率 n_eff (式 7) = {n_eff:.6g}")
        print(f" 膜厚の分解能 (式 6) : {analysis.resolution / (2 * n_eff):.6g} nm"
              f"   最大膜厚: {analysis.nyquist / (2 * n_eff):.6g} nm")
        if analysis.substrate is not None:
            print(f" 基板の屈折率        : {analysis.substrate.description}")
        limit = absorption_thickness_limit(analysis)
        if limit is not None:
            print(f" 吸収による最大膜厚の目安 (式 17) : {limit[0]:.6g} - {limit[1]:.6g} nm")
    print(bar)

    df = peaks_dataframe(analysis, refractive_index)
    if analysis.fit_method == "none":
        cols = ["column", "condition", dom.peak_col, "peak_amplitude"]
    else:
        cols = ["column", "condition", f"fft_{dom.peak_col}",
                f"grid_{dom.peak_col}"]
        if analysis.fit_method == "lm":
            cols += [dom.peak_col, f"{dom.peak_col}_stderr"]
        cols += (["fit_scale"] if analysis.fit_model == "reflectance"
                 else ["fit_amplitude"]) + ["fit_r2"]
    for extra in ("estimated_thickness_nm", "estimated_thickness_stderr_nm",
                  "fringe_thickness_nm", "fringe_thickness_sem_nm", "fringe_pairs"):
        if extra in df.columns:
            cols.append(extra)
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.8g}"))

    # 分散で FFT のピークが広がっていないか (論文の第 4 章)
    uses_dispersion_method = (
        (analysis.fit_method != "none" and analysis.fit_model == "reflectance")
        or any(r.fringe is not None for r in analysis.results))
    if (analysis.group_index is not None and n_eff is not None
            and not uses_dispersion_method):
        spread = max(abs(r.estimate) for r in analysis.results) * (
            analysis.group_index[1] - analysis.group_index[0]) / n_eff
        if spread > 2.0 * analysis.resolution:
            print(bar)
            print(f" [注意] 屈折率の分散で縞の間隔が帯域内で変わり、FFT のピークが最大で")
            print(f"        分解能の約 {spread / analysis.resolution:.0f} 倍に広がっています。"
                  "正弦波モデルは合わなくなるので、")
            print("        --fit-model reflectance (式 1) か --fringe (式 8) を推奨します。")
    if "fringe_reliable" in df.columns and not df["fringe_reliable"].all():
        bad = df.loc[~df["fringe_reliable"], "column"].tolist()
        print(bar)
        print(f" [注意] 列 {', '.join(bad)} は式 8 の結果の信頼性が低いです "
              f"(縞 1 周期あたり {FRINGE_MIN_POINTS:g} 点未満、または隣接ペアが 3 組未満)。")

    if analysis.fit_method != "none":
        poor = df.loc[df["fit_r2"] < FIT_R2_WARN, "column"].tolist()
        notes = df.loc[df["fit_note"] != "", ["column", "fit_note"]]
        if poor or len(notes):
            print(bar)
        if poor:
            print(f" [警告] 列 {', '.join(poor)} はフィットの決定係数が "
                  f"{FIT_R2_WARN} 未満です。")
            if analysis.fit_model == "reflectance":
                print("        反射率モデルがデータに合っていません。膜や基板の屈折率、")
                print("        多層膜であることなどを見直してください。")
            else:
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
    parser.add_argument("--peak-weight", choices=list(PEAK_WEIGHTS), default="none",
                        help="ピーク探索の重み。m2 ならパワースペクトルに番号の 2 乗を掛けて "
                             "探し、低い位置の偽のピークを抑える (論文の第 4 章, 既定: none)")
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument("-n", "--refractive-index", type=float, default=None,
                             help="膜の屈折率 (一定)。波数ドメインで膜厚 d = OPD/(2n) を推定する")
    index_group.add_argument("--cauchy", type=parse_cauchy, default=None, metavar="A,B[,C]",
                             help="膜の屈折率を Cauchy の式 n = A + B/λ² + C/λ⁴ "
                                  "(λ: µm) で与える。分散を考慮した有効屈折率 (式 7) を使う")
    index_group.add_argument("--n-table", default=None, metavar="CSV",
                             help="膜の屈折率テーブル (1 列目 波長 nm, 2 列目 n, 3 列目 k [任意])")
    sub_group = parser.add_mutually_exclusive_group()
    sub_group.add_argument("--substrate-n", type=float, default=None,
                           help="基板の屈折率 (反射率モデル用, 既定: 1.0 = 自立膜)")
    sub_group.add_argument("--substrate-table", default=None, metavar="CSV",
                           help="基板の屈折率テーブル (形式は --n-table と同じ)")
    parser.add_argument("--substrate-k", type=float, default=0.0,
                        help="基板の消衰係数 k (--substrate-n と併用, 既定: 0)")
    parser.add_argument("--fit-model", choices=list(FIT_MODELS), default="sinusoid",
                        help="精密化のモデル。reflectance なら多重反射の反射率の式で "
                             "膜厚を直接フィットする (論文の式 1, 15。波数ドメインと屈折率が必要, "
                             "既定: sinusoid)")
    parser.add_argument("--fringe", action="store_true",
                        help="隣接する極大の位置から膜厚を求める (論文の式 8。"
                             "波数ドメインと屈折率が必要)")
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


def configure_index_options(args: argparse.Namespace, domains: list[DomainInfo],
                            parser: argparse.ArgumentParser) -> None:
    """屈折率関連のオプションからモデルを組み立て、組み合わせを検証する.

    結果は args.index_model / args.substrate_model に入れる。
    """
    if args.cauchy is not None:
        args.index_model = cauchy_index(args.cauchy)
    elif args.n_table is not None:
        args.index_model = table_index(args.n_table)
    elif args.refractive_index is not None:
        args.index_model = constant_index(args.refractive_index)
    else:
        args.index_model = None

    if args.substrate_table is not None:
        args.substrate_model = table_index(args.substrate_table)
    elif args.substrate_n is not None:
        args.substrate_model = constant_index(args.substrate_n, args.substrate_k)
    else:
        args.substrate_model = None

    # --- まず組み合わせの誤りを検出する ----------------------------------
    has_sigma = any(d.key == "wavenumber" for d in domains)
    needs_sigma = [name for name, used in (("--fit-model reflectance",
                                            args.fit_model == "reflectance"),
                                           ("--fringe", args.fringe)) if used]
    if needs_sigma and args.index_model is None:
        parser.error(f"{', '.join(needs_sigma)} には膜の屈折率 "
                     "(-n / --cauchy / --n-table) が必要です。")
    if needs_sigma and not has_sigma:
        parser.error(f"{', '.join(needs_sigma)} は波数ドメインでのみ使えます。"
                     "--domain wavenumber または both を指定してください。")

    # --- 問題なければ、必要な既定値を補う ----------------------------------
    if args.fit_model == "reflectance":
        if args.fit == "none":
            args.fit = "lm"
            print("[情報] --fit-model reflectance のため --fit lm を有効にしました。")
        if args.substrate_model is None:
            args.substrate_model = constant_index(1.0)
            print("[情報] 基板の屈折率が未指定のため 1.0 (自立膜) とします。"
                  "基板上の膜なら --substrate-n を指定してください。")
    if needs_sigma and len(domains) > 1:
        print(f"[情報] {', '.join(needs_sigma)} は波数ドメインの解析にだけ適用します。")


def run_band(spec: SpectrumData, band: tuple[float, float], domain: DomainInfo,
             args: argparse.Namespace, outdir: Path) -> BandAnalysis:
    """1 帯域・1 ドメイン分の解析・保存・描画をまとめて実行する."""
    sigma = domain.key == "wavenumber"
    analysis = analyze_band(spec, band[0], band[1], window=args.window,
                            detrend=args.detrend, zero_pad=args.zero_pad,
                            kaiser_beta=args.kaiser_beta,
                            peak_min=args.peak_min, refine_peak=args.refine_peak,
                            domain=domain, interp=args.interp, fit=args.fit,
                            grid_points=args.grid_points, max_iter=args.max_iter,
                            baseline_degree=args.baseline_degree,
                            peak_weight=args.peak_weight,
                            index=args.index_model if sigma else None,
                            substrate=args.substrate_model if sigma else None,
                            fit_model=args.fit_model if sigma else "sinusoid",
                            fringe=args.fringe and sigma)
    fit_regression(analysis)
    n_ref = None                       # 屈折率は analysis.index に保持している
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
    configure_index_options(args, domains, parser)

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
