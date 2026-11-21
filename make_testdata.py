#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""干渉スペクトル解析ツールの検証用テストデータ生成スクリプト.

ir_ft_analysis.py の動作確認、特に高周波側 (ナイキスト限界に近い領域) の
挙動を検証するための仮想データを生成する。

生成するデータ
    highfreq : 波長 λ に対する正弦波。干渉縞の波数がナイキスト限界の
               10 % から 90 % まで並ぶ。波長ドメイン FFT の検証用。
    alias    : ナイキスト限界を超える波数を含む。折り返し (エイリアシング)
               が起きる様子を確認するためのデータ。
    film     : 屈折率 n の平行平板による薄膜干渉モデル。位相が 1/λ に比例
               するため、波数ドメイン FFT の検証用。条件列は膜厚 [μm]。
    dispersion : 屈折率に分散がある膜 (Cauchy の式) を Si 基板に載せた、
               多重反射を含む厳密な反射率。有効屈折率 (式 7)、隣接極大からの
               膜厚 (式 8)、反射率モデルによるフィットの検証用。条件列は膜厚 [μm]。
    background : 強い山形のベースラインに弱い縞が乗ったデータ。
               重み付きピーク探索 (PSD × m²) の検証用。条件列は膜厚 [μm]。

各データには真値を記した *_truth.csv を併せて出力する。

使用例
    python make_testdata.py                 # 3 種類すべてをカレントに生成
    python make_testdata.py --kind film --outdir .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: 位相オフセット [rad]。ナイキスト限界ちょうどの成分が偶然消えるのを防ぐ。
PHASE = 0.7


def _add_noise(y: np.ndarray, amp: float, snr: float,
               rng: np.random.Generator) -> np.ndarray:
    """振幅に対する信号対雑音比 snr でガウスノイズを加える."""
    if not snr or snr <= 0:
        return y
    return y + rng.normal(0.0, amp / snr, size=y.shape)


# ---------------------------------------------------------------------------
# 1. 高周波データ (波長ドメイン検証用)
# ---------------------------------------------------------------------------

def make_highfreq(wl_min: float = 200.0, wl_max: float = 1000.0,
                  d_lambda: float = 0.5, wavenumbers: tuple[float, ...] =
                  (10, 20, 30, 40, 50, 60, 70, 80, 90),
                  amp: float = 0.2, offset: float = 0.5,
                  baseline_slope: float = 3e-5, snr: float = 200.0,
                  seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """波数 [波数/100nm] が条件値そのものになる正弦波データを作る.

    条件 c の列は 100 nm 当たり c 周期の干渉縞を持つ。
    Δλ から決まるナイキスト限界は 100/(2Δλ) [波数/100nm] なので、
    回帰直線 y = x はちょうどナイキスト限界の値でその限界に到達する。
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(wl_min, wl_max + d_lambda / 2, d_lambda)
    nyquist = 100.0 / (2.0 * d_lambda)

    data = {"Wavelength (nm)": wl}
    truth = []
    for c in wavenumbers:
        f_per_nm = c / 100.0                       # 1 nm 当たりの波数
        y = offset + amp * np.sin(2 * np.pi * f_per_nm * (wl - wl_min) + PHASE)
        y = y + baseline_slope * (wl - wl_min)     # 緩やかなベースライン
        data[str(c)] = _add_noise(y, amp, snr, rng)
        truth.append({
            "condition": c,
            "true_wavenumber_per_100nm": c,
            "true_amplitude": amp,
            "samples_per_fringe": 1.0 / (f_per_nm * d_lambda),
            "fraction_of_nyquist": c / nyquist,
        })

    meta = {
        "kind": "highfreq",
        "delta_lambda_nm": d_lambda,
        "nyquist_per_100nm": nyquist,
        "expected_regression": {"slope": 1.0, "intercept": 0.0},
        "expected_condition_at_nyquist": nyquist,
        "note": "条件値がそのまま波数 [波数/100nm] になる。回帰直線は y = x。",
    }
    return pd.DataFrame(data), pd.DataFrame(truth), meta


# ---------------------------------------------------------------------------
# 2. エイリアシングデータ
# ---------------------------------------------------------------------------

def make_alias(wl_min: float = 200.0, wl_max: float = 1000.0,
               d_lambda: float = 0.5, wavenumbers: tuple[float, ...] =
               (60, 80, 100, 120, 140, 160),
               amp: float = 0.2, offset: float = 0.5,
               snr: float = 200.0, seed: int = 2
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ナイキスト限界を超える波数を含むデータを作る.

    真の波数 w がナイキスト限界 f_n を超えると、観測される波数は
    2*f_n - w に折り返される。条件と観測波数の直線関係が崩れるため、
    ナイキスト到達条件を求める意味がここで確認できる。
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(wl_min, wl_max + d_lambda / 2, d_lambda)
    nyquist = 100.0 / (2.0 * d_lambda)

    data = {"Wavelength (nm)": wl}
    truth = []
    for c in wavenumbers:
        f_per_nm = c / 100.0
        y = offset + amp * np.sin(2 * np.pi * f_per_nm * (wl - wl_min) + PHASE)
        data[str(c)] = _add_noise(y, amp, snr, rng)
        folded = c if c <= nyquist else 2.0 * nyquist - c
        truth.append({
            "condition": c,
            "true_wavenumber_per_100nm": c,
            "expected_observed_wavenumber_per_100nm": folded,
            "aliased": bool(c > nyquist),
        })

    meta = {
        "kind": "alias",
        "delta_lambda_nm": d_lambda,
        "nyquist_per_100nm": nyquist,
        "note": "真の波数がナイキスト限界を超える列は 2*f_n - w に折り返される。",
    }
    return pd.DataFrame(data), pd.DataFrame(truth), meta


# ---------------------------------------------------------------------------
# 3. 薄膜干渉データ (波数ドメイン検証用)
# ---------------------------------------------------------------------------

def make_film(wl_min: float = 700.0, wl_max: float = 1000.0,
              d_lambda: float = 0.25, thickness_um: tuple[float, ...] =
              (40, 80, 120, 160, 200, 240, 280),
              refractive_index: float = 1.5, amp: float = 0.15,
              offset: float = 0.5, baseline_slope: float = 5e-5,
              snr: float = 300.0, seed: int = 3
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """平行平板の二光束干渉モデルを作る.

    反射率 R(λ) = offset + amp * cos(2π * OPD / λ + φ),  OPD = 2 n d
    位相が 1/λ に比例するため、干渉縞は波数 1/λ に対して等間隔になる。
    波数ドメイン FFT のピークは光路差 OPD [nm] にそのまま現れる。
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(wl_min, wl_max + d_lambda / 2, d_lambda)
    n_points = wl.size

    # 波数ドメインのナイキスト限界 (ツール側の再サンプリングと同じ条件で計算)
    sigma_span = 1.0 / wl_min - 1.0 / wl_max
    d_sigma = sigma_span / (n_points - 1)
    nyquist_opd = 1.0 / (2.0 * d_sigma)
    lambda_nyquist = 100.0 / (2.0 * d_lambda)     # 波長ドメインのナイキスト

    data = {"Wavelength (nm)": wl}
    truth = []
    for d_um in thickness_um:
        d_nm = d_um * 1000.0
        opd = 2.0 * refractive_index * d_nm                 # 光路差 [nm]
        y = offset + amp * np.cos(2 * np.pi * opd / wl + PHASE)
        y = y + baseline_slope * (wl - wl_min)
        data[str(d_um)] = _add_noise(y, amp, snr, rng)
        # 波長ドメインでの局所的な縞の波数 (帯域の両端で異なる = ピークが広がる)
        truth.append({
            "condition_thickness_um": d_um,
            "true_opd_nm": opd,
            "true_amplitude": amp,
            "fraction_of_opd_nyquist": opd / nyquist_opd,
            "local_wavenumber_at_wl_min_per_100nm": 100.0 * opd / wl_min ** 2,
            "local_wavenumber_at_wl_max_per_100nm": 100.0 * opd / wl_max ** 2,
            "fraction_of_lambda_nyquist": (100.0 * opd / wl_min ** 2) / lambda_nyquist,
        })

    meta = {
        "kind": "film",
        "refractive_index": refractive_index,
        "delta_lambda_nm": d_lambda,
        "n_points": int(n_points),
        "nyquist_opd_nm": nyquist_opd,
        "nyquist_per_100nm_lambda_domain": lambda_nyquist,
        "expected_regression_wavenumber_domain": {
            "slope_nm_per_um": 2.0 * refractive_index * 1000.0, "intercept": 0.0},
        "expected_condition_at_nyquist_um": nyquist_opd / (2.0 * refractive_index * 1000.0),
        "note": ("条件列は膜厚 [μm]。波数ドメイン FFT のピークは OPD = 2nd [nm] に立つ。"
                 "波長ドメインでは帯域内で縞間隔が変化するためピークが広がる。"),
    }
    return pd.DataFrame(data), pd.DataFrame(truth), meta


# ---------------------------------------------------------------------------
# 4. 分散のある膜 (有効屈折率・式 8・反射率モデルの検証用)
# ---------------------------------------------------------------------------

def single_layer_reflectance(wl: np.ndarray, d: float, n_film: np.ndarray,
                             n_sub: complex) -> np.ndarray:
    """特性行列法で単層膜の反射率を計算する.

    ir_ft_analysis.layer_reflectance (Airy の式) とは別の導出で計算し、
    解析側のモデル式に誤りがあれば検出できるようにしている。
    屈折率は N = n - ik の符号で与える。
    """
    delta = 2.0 * np.pi * n_film * d / wl
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    # [B, C]ᵀ = [[cos δ, i sin δ / η1], [i η1 sin δ, cos δ]] [1, η2]ᵀ
    b = cos_d + 1j * sin_d * n_sub / n_film
    c = 1j * n_film * sin_d + cos_d * n_sub
    r = (b - c) / (b + c)                          # 入射側は空気 (η0 = 1)
    return np.abs(r) ** 2


def make_dispersion(wl_min: float = 400.0, wl_max: float = 1000.0,
                    d_lambda: float = 0.25,
                    thickness_um: tuple[float, ...] = (2, 5, 10, 20, 30, 40, 50),
                    cauchy: tuple[float, float] = (1.56, 0.005),
                    substrate_n: float = 3.6, noise: float = 3e-4,
                    seed: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """屈折率に分散がある高分子膜を Si 基板に載せたときの反射率.

    膜の屈折率は Cauchy の式 n(λ) = A + B/λ² (λ: µm) で与える。多重反射を
    含む厳密な反射率で、倍率やオフセットは加えない。帯域が広く分散が強いので、
    一定の屈折率で膜厚を求めると数 % ずれ、厚い膜では FFT のピークも広がる。
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(wl_min, wl_max + d_lambda / 2, d_lambda)
    a, b = cauchy
    n_film = a + b / (wl / 1000.0) ** 2
    s1, s2 = 1.0 / wl_min, 1.0 / wl_max
    n1, n2 = n_film[0], n_film[-1]
    n_eff = (n1 * s1 - n2 * s2) / (s1 - s2)       # 論文の式 7
    n_center = a + b / (0.5 * (wl_min + wl_max) / 1000.0) ** 2

    data = {"Wavelength (nm)": wl}
    truth = []
    for d_um in thickness_um:
        d_nm = d_um * 1000.0
        refl = single_layer_reflectance(wl, d_nm, n_film.astype(complex),
                                        complex(substrate_n))
        data[str(d_um)] = refl + (rng.normal(0.0, noise, wl.size) if noise else 0.0)
        truth.append({
            "condition_thickness_um": d_um,
            "true_thickness_nm": d_nm,
            "effective_opd_nm": 2.0 * n_eff * d_nm,
            "thickness_if_constant_n_center_nm": 2.0 * n_eff * d_nm / (2.0 * n_center),
        })

    meta = {
        "kind": "dispersion",
        "delta_lambda_nm": d_lambda,
        "film_cauchy_A_B_um2": [a, b],
        "substrate_n": substrate_n,
        "n_at_band_edges": [float(n1), float(n2)],
        "n_at_band_center": float(n_center),
        "n_eff_full_band": float(n_eff),
        "constant_n_bias_percent": float((n_eff / n_center - 1.0) * 100.0),
        "noise_sigma": noise,
        "note": ("条件列は膜厚 [μm]。一定の屈折率 (帯域中央の値) で求めると膜厚が "
                 "n_eff / n_center だけずれる。--cauchy 1.56,0.005 --substrate-n 3.6 で解析する。"),
    }
    return pd.DataFrame(data), pd.DataFrame(truth), meta


# ---------------------------------------------------------------------------
# 5. 強いベースラインの上の弱い縞 (重み付きピーク探索の検証用)
# ---------------------------------------------------------------------------

def make_background(wl_min: float = 500.0, wl_max: float = 1000.0,
                    d_lambda: float = 0.25,
                    thickness_um: tuple[float, ...] = (20, 40, 60, 80, 100),
                    refractive_index: float = 1.5, amp: float = 0.02,
                    noise: float = 5e-4, seed: int = 5
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """光源スペクトルや吸収端のような強い山形のベースラインに、弱い縞が乗ったデータ.

    1 次のデトレンドでは山形を取りきれず、その残りが低い位置に大きなピークを
    作る。振幅最大のピークを探すと縞ではなくこちらを拾ってしまう。
    パワースペクトルに番号の 2 乗を掛けると (論文の第 4 章)、高い位置にある
    縞のピークが強調されて正しく検出できる。
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(wl_min, wl_max + d_lambda / 2, d_lambda)
    background = (0.30 + 0.12 * np.exp(-((wl - 650.0) / 60.0) ** 2)
                  + 0.05 * ((wl - 750.0) / 250.0) ** 2)
    data = {"Wavelength (nm)": wl}
    truth = []
    for d_um in thickness_um:
        opd = 2.0 * refractive_index * d_um * 1000.0
        y = background + amp * np.cos(2.0 * np.pi * opd / wl + PHASE)
        data[str(d_um)] = _add_noise(y, amp, amp / noise, rng)
        truth.append({"condition_thickness_um": d_um, "true_opd_nm": opd,
                      "true_amplitude": amp})
    meta = {
        "kind": "background",
        "delta_lambda_nm": d_lambda,
        "refractive_index": refractive_index,
        "fringe_amplitude": amp,
        "note": ("山形のベースライン (振幅 0.12) の上に振幅 0.02 の縞が乗る。"
                 "波数ドメインで --peak-weight m2 を付けると縞のピークを検出できる。"),
    }
    return pd.DataFrame(data), pd.DataFrame(truth), meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GENERATORS = {
    "highfreq": (make_highfreq, "dataset_highfreq"),
    "alias": (make_alias, "dataset_alias"),
    "film": (make_film, "dataset_film"),
    "dispersion": (make_dispersion, "dataset_dispersion"),
    "background": (make_background, "dataset_background"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="干渉スペクトル解析ツールの検証用テストデータを生成する")
    parser.add_argument("-k", "--kind", choices=[*GENERATORS, "all"], default="all",
                        help="生成するデータの種類 (既定: all)")
    parser.add_argument("-o", "--outdir", default=".",
                        help="出力ディレクトリ (既定: カレントディレクトリ)")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kinds = list(GENERATORS) if args.kind == "all" else [args.kind]

    for kind in kinds:
        func, stem = GENERATORS[kind]
        data, truth, meta = func()
        data.to_csv(outdir / f"{stem}.csv", index=False, encoding="utf-8-sig")
        truth.to_csv(outdir / f"{stem}_truth.csv", index=False, encoding="utf-8-sig")
        with open(outdir / f"{stem}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        print(f"[{kind}] {outdir / (stem + '.csv')}  "
              f"({data.shape[0]} 行 x {data.shape[1] - 1} 条件列)")
        for key, value in meta.items():
            if key != "note":
                print(f"    {key}: {value}")
        print(f"    真値: {outdir / (stem + '_truth.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
