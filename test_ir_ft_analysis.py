#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ir_ft_analysis.py の自己テスト.

既知の真値を持つ合成データを作り、解析結果が真値と一致するかを検証する。
pytest は不要で、そのまま実行できる。

    python test_ir_ft_analysis.py

ソースを更新したら、このテストと CODE_GUIDE.md の更新も忘れないこと。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import make_testdata as mtd
from ir_ft_analysis import (DEFAULT_GRID_POINTS, FIT_R2_WARN, GRID_HALF_WIDTH,
                            WINDOW_FUNCTIONS, analyze_band, fit_regression,
                            interpolate, is_near_nyquist, load_spectra,
                            make_window, parse_band, peaks_dataframe,
                            refine_peak_fit, resolve_domain,
                            resolve_window_name)

TMP = Path(tempfile.mkdtemp(prefix="ir_ft_test_"))
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """1 件の検証結果を記録して表示する."""
    mark = "OK  " if ok else "NG  "
    print(f"  {mark}{name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = TMP / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------------------
# 1. 波長ドメイン: 全窓関数で振幅と波数を復元できるか
# ---------------------------------------------------------------------------

def test_wavelength_domain_windows() -> None:
    print("\n[1] 波長ドメイン: 窓関数ごとの復元精度 (真値 波数=条件, 振幅=0.2)")
    data, truth, meta = mtd.make_highfreq(snr=0.0, baseline_slope=3e-5)
    spec = load_spectra(write_csv(data, "highfreq.csv"))

    check("ナイキスト周波数の導出", abs(spec.nyquist_per_100nm - 100.0) < 1e-9,
          f"{spec.nyquist_per_100nm} 波数/100nm")

    # ピーク位置の許容誤差は分解能の 30 %。窓ごとに主ローブ幅が違うため、
    # 絶対値ではなく分解能に対する比で判定する。
    for win in WINDOW_FUNCTIONS:
        a = analyze_band(spec, 200, 1000, window=win, zero_pad=8, refine_peak=True)
        d_wn = max(abs(r.peak_frequency - r.condition) for r in a.results)
        d_amp = max(abs(r.peak_amplitude - 0.2) / 0.2 for r in a.results) * 100
        check(f"window={win:<15}", d_wn < 0.3 * a.resolution and d_amp < 1.0,
              f"Δ波数 <= {d_wn:.4f} (分解能の {d_wn / a.resolution * 100:.0f} %), "
              f"振幅誤差 <= {d_amp:.3f} %")


# ---------------------------------------------------------------------------
# 2. 波長ドメイン: 回帰とナイキスト到達条件
# ---------------------------------------------------------------------------

def test_wavelength_regression() -> None:
    print("\n[2] 波長ドメイン: 回帰直線 (真値 y = 1.0 x, ナイキスト到達 = 100)")
    data, truth, meta = mtd.make_highfreq()
    spec = load_spectra(write_csv(data, "highfreq_noise.csv"))
    a = analyze_band(spec, 200, 1000, window="hann", zero_pad=8, refine_peak=True)
    reg = fit_regression(a)

    check("傾き = 1.0", abs(reg.slope - 1.0) < 1e-3, f"{reg.slope:.6f}")
    check("切片 = 0.0", abs(reg.intercept) < 1e-2, f"{reg.intercept:.3e}")
    check("決定係数 = 1.0", reg.r2 > 0.99999, f"{reg.r2:.8f}")
    check("ナイキスト到達条件 = 100",
          abs(reg.crossing_condition - meta["expected_condition_at_nyquist"]) < 0.05,
          f"{reg.crossing_condition:.5f}")


# ---------------------------------------------------------------------------
# 3. エイリアシング: ナイキスト超えの折り返し
# ---------------------------------------------------------------------------

def test_aliasing() -> None:
    print("\n[3] エイリアシング: 真値が限界を超えると 2*f_n - w に折り返される")
    data, truth, meta = mtd.make_alias()
    spec = load_spectra(write_csv(data, "alias.csv"))
    a = analyze_band(spec, 200, 1000, window="hann", zero_pad=8, refine_peak=True)

    expected = dict(zip(truth["condition"],
                        truth["expected_observed_wavenumber_per_100nm"]))
    for r in a.results:
        exp = expected[r.condition]
        check(f"条件 {r.condition:>5g} -> 観測波数 {exp:g}",
              abs(r.peak_frequency - exp) < 0.05, f"{r.peak_frequency:.4f}")

    reg = fit_regression(a)
    check("折り返しにより決定係数が大きく低下する", reg.r2 < 0.5, f"R^2 = {reg.r2:.4f}")
    near = [r for r in a.results if is_near_nyquist(a, r.peak_frequency)]
    check("ナイキスト直下のピークを検出できる", len(near) == 1,
          f"{[r.column for r in near]}")


# ---------------------------------------------------------------------------
# 4. 波数ドメイン: 光路差と膜厚の復元
# ---------------------------------------------------------------------------

def test_wavenumber_domain() -> None:
    print("\n[4] 波数ドメイン: ピーク = 光路差 OPD = 2nd")
    data, truth, meta = mtd.make_film()
    spec = load_spectra(write_csv(data, "film.csv"))
    n_ref = meta["refractive_index"]
    a = analyze_band(spec, 700, 1000, window="hann", zero_pad=8,
                     refine_peak=True, domain="wavenumber")
    reg = fit_regression(a)

    check("ナイキスト限界 (光路差)",
          abs(a.nyquist - meta["nyquist_opd_nm"]) / meta["nyquist_opd_nm"] < 1e-6,
          f"{a.nyquist:.1f} nm")

    expected = dict(zip(truth["condition_thickness_um"], truth["true_opd_nm"]))
    worst_opd = 0.0
    worst_thick = 0.0
    for r in a.results:
        exp = expected[r.condition]
        worst_opd = max(worst_opd, abs(r.peak_frequency - exp) / exp)
        worst_thick = max(worst_thick,
                          abs(r.peak_frequency / (2 * n_ref) - r.condition * 1000.0))
    check("光路差の相対誤差 < 0.01 %", worst_opd < 1e-4, f"{worst_opd * 100:.5f} %")
    check("膜厚の絶対誤差 < 10 nm", worst_thick < 10.0, f"{worst_thick:.2f} nm")

    slope_expected = meta["expected_regression_wavenumber_domain"]["slope_nm_per_um"]
    check(f"傾き = {slope_expected:g} nm/um",
          abs(reg.slope - slope_expected) / slope_expected < 1e-4, f"{reg.slope:.3f}")
    check("決定係数 = 1.0", reg.r2 > 0.999999, f"{reg.r2:.8f}")
    check("ナイキスト到達条件",
          abs(reg.crossing_condition - meta["expected_condition_at_nyquist_um"]) < 0.5,
          f"{reg.crossing_condition:.3f} um")


# ---------------------------------------------------------------------------
# 5. 補間方式による振幅の違い
# ---------------------------------------------------------------------------

def test_interpolation() -> None:
    print("\n[5] 波数ドメインの再サンプリング: cubic は linear より振幅を保つ")
    data, truth, meta = mtd.make_film()
    spec = load_spectra(write_csv(data, "film2.csv"))
    amps = {}
    for kind in ("linear", "cubic"):
        a = analyze_band(spec, 700, 1000, window="hann", zero_pad=8,
                         refine_peak=True, domain="wavenumber", interp=kind)
        amps[kind] = a.results[-1].peak_amplitude          # 最も厚い条件
        pos_ok = abs(a.results[-1].peak_frequency
                     - truth["true_opd_nm"].iloc[-1]) < 10.0
        check(f"interp={kind:<7} でもピーク位置は正確", pos_ok,
              f"振幅 {amps[kind]:.5f} (真値 {meta.get('amp', 0.15)})")
    check("cubic の方が振幅減衰が小さい", amps["cubic"] > amps["linear"],
          f"cubic {amps['cubic']:.5f} > linear {amps['linear']:.5f}")

    # 補間関数そのものの精度 (等間隔の正弦波を 1/3 点ずらして評価)
    x = np.linspace(0, 10, 200)
    y = np.sin(2 * np.pi * 0.7 * x)
    xq = x[:-1] + (x[1] - x[0]) / 3.0
    ref = np.sin(2 * np.pi * 0.7 * xq)
    err_lin = np.max(np.abs(interpolate(x, y, xq, "linear") - ref))
    err_cub = np.max(np.abs(interpolate(x, y, xq, "cubic") - ref))
    check("cubic 補間の誤差が linear より小さい", err_cub < err_lin,
          f"cubic {err_cub:.2e} < linear {err_lin:.2e}")


# ---------------------------------------------------------------------------
# 6. 振幅正規化 (窓のコヒーレントゲイン)
# ---------------------------------------------------------------------------

def test_amplitude_normalization() -> None:
    print("\n[6] 振幅正規化: 窓のコヒーレントゲインが理論値と一致する")
    for name, expected in (("rectangular", 1.0), ("hann", 0.5), ("blackman", 0.42)):
        gain = float(make_window(name, 4001).mean())
        check(f"{name:<12} コヒーレントゲイン = {expected}",
              abs(gain - expected) < 2e-3, f"{gain:.5f}")


# ---------------------------------------------------------------------------
# 7. 入力データの頑健性
# ---------------------------------------------------------------------------

def test_robustness() -> None:
    print("\n[7] 入力の頑健性: 不等間隔・欠測・非数値列名")
    wl = np.sort(np.concatenate([np.arange(200, 600, 0.5),
                                 np.arange(600, 1000.5, 0.7)]))
    frame = pd.DataFrame({"Wavelength (nm)": wl})
    for c in (10, 20, 30):
        frame[str(c)] = 0.5 + 0.2 * np.sin(2 * np.pi * (c / 100.0) * (wl - 200))
    frame["memo"] = ["x"] * len(wl)                 # 非数値列名 -> 除外
    frame["99"] = [np.nan] * len(wl)                # 全欠測 -> 除外
    frame.loc[10:14, "20"] = np.nan                 # 部分欠測 -> 補間
    spec = load_spectra(write_csv(frame, "robust.csv"))

    check("非数値列名と全欠測列を除外する", spec.column_names == ["10", "20", "30"],
          f"{spec.column_names}")
    check("不等間隔を等間隔化する", spec.resampled, f"Δλ = {spec.delta_lambda:.6g} nm")

    a = analyze_band(spec, 300, 600, window="hann", zero_pad=8, refine_peak=True)
    worst = max(abs(r.peak_frequency - r.condition) for r in a.results)
    check("再サンプリング後も波数を復元できる", worst < 0.05, f"Δ波数 <= {worst:.4f}")


# ---------------------------------------------------------------------------
# 8. 入力パーサと出力表
# ---------------------------------------------------------------------------

def test_parsers_and_output() -> None:
    print("\n[8] 引数パーサと出力表")
    for text in ("300-600", "300:600", "300~600", "300～600", "300-600nm",
                 "３００-６００", " 300 , 600 "):
        try:
            got = parse_band(text)
        except Exception as exc:                    # noqa: BLE001 - テスト用
            got = f"error: {exc}"
        check(f"parse_band({text!r})", got == (300.0, 600.0), f"{got}")

    check("窓関数エイリアス hanning -> hann", resolve_window_name("hanning") == "hann")
    check("ドメインエイリアス sigma -> wavenumber",
          resolve_domain("sigma").key == "wavenumber")

    data, _, _ = mtd.make_film()
    spec = load_spectra(write_csv(data, "film3.csv"))
    a = analyze_band(spec, 700, 1000, domain="wavenumber", refine_peak=True)
    fit_regression(a)
    df = peaks_dataframe(a, refractive_index=1.5)
    check("波数ドメインの出力列名", "peak_opd_nm" in df.columns, f"{list(df.columns)}")
    check("屈折率を渡すと膜厚推定列が付く", "estimated_thickness_nm" in df.columns)

    a_wl = analyze_band(spec, 700, 1000, domain="wavelength")
    df_wl = peaks_dataframe(a_wl)
    check("波長ドメインの出力列名", "peak_wavenumber_per_100nm" in df_wl.columns)
    check("ドメインで出力ファイル名が分かれる",
          a.label.endswith("_sigma") and a_wl.label.endswith("_lambda"),
          f"{a_wl.label} / {a.label}")


# ---------------------------------------------------------------------------
# 9. 精密化: グリッドサーチ + Levenberg-Marquardt 法
# ---------------------------------------------------------------------------

def _sinusoid_csv(name: str, conditions, freq_of, amp: float = 0.2,
                  noise: float = 0.0, seed: int = 0) -> Path:
    """条件ごとに波数 freq_of(c) [/100nm] の正弦波を持つ CSV を作る."""
    rng = np.random.default_rng(seed)
    wl = np.arange(300.0, 600.0 + 0.25, 0.5)
    data = {"Wavelength (nm)": wl}
    for c in conditions:
        y = (0.5 + 1e-4 * (wl - 450.0)
             + amp * np.sin(2 * np.pi * freq_of(c) / 100.0 * wl + 0.4))
        data[str(c)] = y + (rng.normal(0.0, noise, wl.size) if noise else 0.0)
    return write_csv(pd.DataFrame(data), name)


def test_peak_refinement() -> None:
    print("\n[9] 精密化 (グリッドサーチ + Levenberg-Marquardt 法)")

    # --- 格子点の間にある波数を、ノイズなしなら機械精度近くまで復元できるか ---
    conds = [2, 4, 6, 8, 10, 12, 14]
    spec = load_spectra(_sinusoid_csv("offbin.csv", conds, lambda c: c / (2 * np.pi)))
    a_fft = analyze_band(spec, 300, 600)
    a_grid = analyze_band(spec, 300, 600, fit="grid")
    a_lm = analyze_band(spec, 300, 600, fit="lm")
    truth = {c: c / (2 * np.pi) for c in conds}
    err = {name: max(abs(r.estimate - truth[r.condition]) for r in a.results)
           for name, a in (("fft", a_fft), ("grid", a_grid), ("lm", a_lm))}
    res = a_fft.resolution
    grid_step = 2 * GRID_HALF_WIDTH * res / DEFAULT_GRID_POINTS
    check("FFT のみでは格子間隔ぶんの誤差が残る", err["fft"] > 0.1 * res,
          f"最大誤差 {err['fft']:.4f} (分解能 {res:.4f})")
    check("グリッドサーチの誤差 <= 刻みの半分", err["grid"] <= 0.5 * grid_step + 1e-12,
          f"最大誤差 {err['grid']:.4f} (刻み {grid_step:.4f})")
    check("LM 法の誤差 < 1e-8 /100nm", err["lm"] < 1e-8, f"最大誤差 {err['lm']:.2e}")
    check("LM 法は FFT より 1000 倍以上正確", err["fft"] / max(err["lm"], 1e-300) > 1e3,
          f"{err['fft'] / max(err['lm'], 1e-300):.1e} 倍")
    check("LM 法は少ない反復で収束する (論文: 約 20 回)",
          all(r.fit.converged and r.fit.iterations <= 20 for r in a_lm.results),
          f"反復回数 {[r.fit.iterations for r in a_lm.results]}")
    check("LM 法で振幅も正確 (真値 0.2)",
          max(abs(r.fit.amplitude - 0.2) for r in a_lm.results) < 1e-8)
    reg = fit_regression(a_lm)
    check("回帰には精密化後の値が使われる (傾き = 1/2π)",
          abs(reg.slope - 1 / (2 * np.pi)) < 1e-9, f"{reg.slope:.10f}")
    check("最低次のビン付近 (探索範囲が負にはみ出す) でも動く",
          abs(a_lm.results[0].estimate - 1 / np.pi) < 1e-8,
          f"{a_lm.results[0].estimate:.10f} (真値 {1 / np.pi:.10f})")

    # --- 標準誤差は実際のばらつきと一致するか (モンテカルロ) ---
    rng = np.random.default_rng(7)
    wl = np.arange(300.0, 600.0 + 0.25, 0.5)
    f_true, sigma = 37.3456, 0.01
    ests, ses = [], []
    for _ in range(150):
        y = (0.5 + 2e-4 * (wl - 450) + 0.2 * np.cos(2 * np.pi * f_true / 100 * wl + 0.3)
             + rng.normal(0.0, sigma, wl.size))
        fr = refine_peak_fit(wl, y, round(f_true / res) * res, res, 100.0, "lm")
        ests.append(fr.frequency)
        ses.append(fr.stderr)
    ratio = np.std(ests, ddof=1) / np.mean(ses)
    check("標準誤差が実際のばらつきと一致 (比 0.8-1.25)", 0.8 < ratio < 1.25,
          f"実測 std / 推定 stderr = {ratio:.3f}")
    check("ノイズ下でも偏りがない (|偏り| < stderr)",
          abs(np.mean(ests) - f_true) < np.mean(ses), f"{np.mean(ests) - f_true:+.2e}")

    # --- 波数ドメイン: 補間前の生データにフィットして光路差を復元できるか ---
    data, truth_f, _ = mtd.make_film(snr=0.0, baseline_slope=0.0)
    spec_f = load_spectra(write_csv(data, "film_fit.csv"))
    a_f = analyze_band(spec_f, 700, 1000, domain="wavenumber", fit="lm")
    worst = max(abs(r.estimate - t) for r, t in zip(a_f.results, truth_f["true_opd_nm"]))
    check("波数ドメインで光路差の誤差 < 0.01 nm", worst < 0.01, f"最大誤差 {worst:.2e} nm")
    n_band = int(((spec_f.wavelength >= 700) & (spec_f.wavelength <= 1000)).sum())
    check("波数ドメインのフィットは補間前の点を使う",
          a_f.results[0].fit.x.size == n_band and not np.allclose(
              np.diff(a_f.results[0].fit.x), np.diff(a_f.results[0].fit.x)[0]),
          "不等間隔の 1/λ をそのまま使用")

    # --- モデルが合わないときに検出できるか ---
    a_bad = analyze_band(spec_f, 700, 1000, domain="wavelength", fit="lm")
    check("縞間隔が変化するデータではフィットの R² が低くなる",
          all(r.fit.r2 < FIT_R2_WARN for r in a_bad.results),
          f"R² 最大 {max(r.fit.r2 for r in a_bad.results):.3f}")

    # --- 出力表 ---
    df = peaks_dataframe(a_f, refractive_index=1.5)
    need = ["fft_peak_opd_nm", "grid_peak_opd_nm", "peak_opd_nm_stderr",
            "fit_r2", "estimated_thickness_stderr_nm"]
    check("精密化の列が出力される", all(c in df.columns for c in need),
          f"不足: {[c for c in need if c not in df.columns]}")
    df_none = peaks_dataframe(analyze_band(spec_f, 700, 1000, domain="wavenumber"))
    check("精密化しなければ列は増えない", "fit_r2" not in df_none.columns)


# ---------------------------------------------------------------------------

def main() -> int:
    print(f"作業ディレクトリ: {TMP}")
    test_wavelength_domain_windows()
    test_wavelength_regression()
    test_aliasing()
    test_wavenumber_domain()
    test_interpolation()
    test_amplitude_normalization()
    test_robustness()
    test_parsers_and_output()
    test_peak_refinement()

    print("\n" + "=" * 60)
    if _failures:
        print(f"失敗 {len(_failures)} 件:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("すべて成功しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
