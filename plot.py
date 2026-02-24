"""Compare Python vs Fortran spectra: overlay, side-by-side, and fractional error.

Usage examples:
  python plot.py --atmosphere at12_aaaaa_t04250g2.50.spec
  python plot.py --wl-start 350 --wl-end 800 --no-show
  python plot.py --python-spec path/to/python.spec --fortran-spec path/to/fortran.spec

Input spectra are expected to contain at least 3 numeric columns:
  wavelength_nm, flux, continuum
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

DEFAULT_ATMOSPHERE = "at12_aaaaa_t04250g2.50.spec"
DEFAULT_WL_START = 300.0
DEFAULT_WL_END = 1800.0
BASE_DIR = Path("results/validation_100")

FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Formatter that keeps newlines and includes default values."""


def load_spectrum(path: Path) -> np.ndarray:
    """Parse 3-column spectrum rows, including Fortran glued fields."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            nums = FLOAT_RE.findall(line)
            if len(nums) >= 3:
                rows.append([float(nums[0]), float(nums[1]), float(nums[2])])
    if not rows:
        raise ValueError(f"No valid 3-column rows found in {path}")
    return np.array(rows, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Python/Fortran spectra: row 1 overlay, row 2 side-by-side, row 3 fractional error."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plot.py --atmosphere at12_aaaaa_t04250g2.50.spec\n"
            "  python plot.py --wl-start 400 --wl-end 700 --save results/plots/zoom.png --no-show\n"
            "  python plot.py --python-spec ./py.spec --fortran-spec ./ft.spec\n\n"
            "Notes:\n"
            "  - If wavelength grids differ, the Fortran spectrum is linearly interpolated onto\n"
            "    the Python wavelength grid for fractional-error calculation.\n"
            "  - Fractional error is computed as (Python - Fortran) / Python."
        ),
    )
    parser.add_argument(
        "--atmosphere",
        type=str,
        default=DEFAULT_ATMOSPHERE,
        help=(
            "Spectrum filename under results/validation_100/{python_specs,fortran_specs}. "
            "Used unless explicit paths are provided."
        ),
    )
    parser.add_argument(
        "--python-spec",
        type=str,
        default=None,
        help="Explicit Python spectrum path. Provide together with --fortran-spec.",
    )
    parser.add_argument(
        "--fortran-spec",
        type=str,
        default=None,
        help="Explicit Fortran spectrum path. Provide together with --python-spec.",
    )
    parser.add_argument(
        "--wl-start",
        type=float,
        default=DEFAULT_WL_START,
        help="Lower wavelength bound (nm) for displayed top-panel spectra.",
    )
    parser.add_argument(
        "--wl-end",
        type=float,
        default=DEFAULT_WL_END,
        help="Upper wavelength bound (nm) for displayed top-panel spectra.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "Output PNG path. If omitted, writes to "
            "results/validation_100/plots/<atmosphere>.png."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plot without displaying (for batch mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.python_spec and args.fortran_spec:
        path1 = Path(args.python_spec)
        path2 = Path(args.fortran_spec)
    else:
        path1 = BASE_DIR / "python_specs" / args.atmosphere
        path2 = BASE_DIR / "fortran_specs" / args.atmosphere

    data1 = load_spectrum(path1)
    data2 = load_spectrum(path2)

    wavelength1_all, flux1_all, continuum1_all = data1[:, 0], data1[:, 1], data1[:, 2]
    wavelength2_all, flux2_all, continuum2_all = data2[:, 0], data2[:, 1], data2[:, 2]

    normalized_flux1_all = flux1_all / continuum1_all
    normalized_flux2_all = flux2_all / continuum2_all

    # Interpolate spectrum 2 onto spectrum 1 grid for error calculation.
    if not np.array_equal(wavelength1_all, wavelength2_all):
        interp_func = interp1d(
            wavelength2_all,
            normalized_flux2_all,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        normalized_flux2_interp_all = interp_func(wavelength1_all)
        wavelength_for_error = wavelength1_all
        normalized_flux_ref_all = normalized_flux1_all
        normalized_flux_spec_all = normalized_flux2_interp_all
    else:
        wavelength_for_error = wavelength1_all
        normalized_flux_ref_all = normalized_flux1_all
        normalized_flux_spec_all = normalized_flux2_all

    with np.errstate(divide="ignore", invalid="ignore"):
        fractional_error_all = np.divide(
            normalized_flux_ref_all - normalized_flux_spec_all,
            normalized_flux_ref_all,
            out=np.full_like(normalized_flux_ref_all, np.nan),
            where=np.abs(normalized_flux_ref_all) > 1e-30,
        )
        # Robust fractional error: suppress near-zero denominator blow-ups in
        # deep line cores where tiny sign flips produce visually "exploding"
        # relative error but negligible absolute difference.
        finite_ref = np.isfinite(normalized_flux_ref_all)
        ref_scale = (
            float(np.nanmedian(np.abs(normalized_flux_ref_all[finite_ref])))
            if np.any(finite_ref)
            else 0.0
        )
        robust_denom_threshold = max(1e-30, 1e-3 * ref_scale)
        robust_denom_mask = np.abs(normalized_flux_ref_all) > robust_denom_threshold
        fractional_error_robust_all = np.divide(
            normalized_flux_ref_all - normalized_flux_spec_all,
            normalized_flux_ref_all,
            out=np.full_like(normalized_flux_ref_all, np.nan),
            where=robust_denom_mask,
        )

    # #region agent log
    _ref = normalized_flux_ref_all
    _spec = normalized_flux_spec_all
    _err = fractional_error_all
    _wl = wavelength_for_error
    _small_denom = np.abs(_ref) < 0.01
    _outlier = np.abs(_err) > 0.5
    _infnan = ~np.isfinite(_err)
    _continuum = np.abs(_ref - 1.0) < 0.05
    _out_in_cont = _outlier & _continuum
    _out_in_small = _outlier & _small_denom
    _finite_err = np.isfinite(_err)
    _top5 = np.argsort(np.where(_finite_err, np.abs(_err), -1))[-5:][::-1] if np.any(_finite_err) else []
    def _safe(v):
        v = float(v)
        return None if (v != v or abs(v) == float("inf")) else v
    _samples = [{"wl": float(_wl[i]), "ref": _safe(_ref[i]), "spec": _safe(_spec[i]), "err": _safe(_err[i])} for i in _top5 if i < len(_wl)]
    _pct = lambda q: _safe(np.nanpercentile(_err, q))
    if os.getenv("PY_ENABLE_PLOT_DEBUG", "0") == "1":
        log_path = Path(
            os.getenv(
                "PY_PLOT_DEBUG_LOG_PATH",
                "results/validation_100/plot_fractional_error_debug.ndjson",
            )
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as _log:
            _log.write(
                __import__("json").dumps(
                    {
                        "sessionId": "fb1c65",
                        "hypothesisId": "A-E",
                        "location": "plot.py:fractional_error",
                        "message": "fractional_error_validation",
                        "data": {
                            "atmosphere": args.atmosphere,
                            "n_total": len(_ref),
                            "n_small_denom": int(np.sum(_small_denom)),
                            "n_outlier": int(np.sum(_outlier)),
                            "n_infnan": int(np.sum(_infnan)),
                            "n_out_in_continuum": int(np.sum(_out_in_cont)),
                            "n_out_in_small_denom": int(np.sum(_out_in_small)),
                            "robust_denom_threshold": _safe(robust_denom_threshold),
                            "n_robust_denom": int(np.sum(robust_denom_mask)),
                            "ref_at_small_denom": {
                                "min": (
                                    _safe(np.min(_ref[_small_denom]))
                                    if np.any(_small_denom)
                                    else None
                                ),
                                "max": (
                                    _safe(np.max(_ref[_small_denom]))
                                    if np.any(_small_denom)
                                    else None
                                ),
                            },
                            "top5_outliers": _samples,
                            "err_percentiles": {
                                "p50": _pct(50),
                                "p95": _pct(95),
                                "p99": _pct(99),
                                "min": _safe(np.nanmin(_err)),
                                "max": _safe(np.nanmax(_err)),
                            },
                        },
                        "timestamp": __import__("time").time() * 1000,
                    }
                )
                + "\n"
            )
    # #endregion

    wl_mask1 = (wavelength1_all >= args.wl_start) & (wavelength1_all <= args.wl_end)
    wl_mask2 = (wavelength2_all >= args.wl_start) & (wavelength2_all <= args.wl_end)

    wavelength1, flux1, continuum1 = (
        wavelength1_all[wl_mask1],
        flux1_all[wl_mask1],
        continuum1_all[wl_mask1],
    )
    wavelength2, flux2, continuum2 = (
        wavelength2_all[wl_mask2],
        flux2_all[wl_mask2],
        continuum2_all[wl_mask2],
    )

    normalized_flux1 = normalized_flux1_all[wl_mask1]
    normalized_flux2 = normalized_flux2_all[wl_mask2]

    mask1 = flux1 > continuum1
    mask2 = flux2 > continuum2
    print(
        f"First 10 wavelengths in {path1} where flux > continuum "
        f"({args.wl_start:g}-{args.wl_end:g} nm):"
    )
    print(wavelength1[mask1][:10])
    print(
        f"First 10 wavelengths in {path2} where flux > continuum "
        f"({args.wl_start:g}-{args.wl_end:g} nm):"
    )
    print(wavelength2[mask2][:10])

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    ax_overlay = fig.add_subplot(gs[0, :])
    ax_python = fig.add_subplot(gs[1, 0], sharex=ax_overlay)
    ax_fortran = fig.add_subplot(gs[1, 1], sharex=ax_overlay)
    ax_error = fig.add_subplot(gs[2, :], sharex=ax_overlay)

    finite_mask1 = np.isfinite(normalized_flux1)
    finite_mask2 = np.isfinite(normalized_flux2)
    all_finite_flux = []
    if np.sum(finite_mask1) > 0:
        all_finite_flux.extend(normalized_flux1[finite_mask1])
    if np.sum(finite_mask2) > 0:
        all_finite_flux.extend(normalized_flux2[finite_mask2])
    if len(all_finite_flux) > 0:
        all_finite_flux = np.array(all_finite_flux)
        margin = 0.05 * (np.max(all_finite_flux) - np.min(all_finite_flux))
        y_min = np.min(all_finite_flux) - margin
        y_max = np.max(all_finite_flux) + margin
        ax_overlay.set_ylim(y_min, y_max)
        ax_python.set_ylim(y_min, y_max)
        ax_fortran.set_ylim(y_min, y_max)

    # Row 1: overlaid normalized spectra (Fortran + Python)
    ax_overlay.plot(
        wavelength2,
        normalized_flux2,
        color="orange",
        alpha=0.7,
        linewidth=1,
        label="Fortran Spectrum",
    )
    ax_overlay.plot(
        wavelength1,
        normalized_flux1,
        color="blue",
        alpha=0.7,
        linewidth=1,
        label="Python Spectrum",
    )
    ax_overlay.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Continuum")
    ax_overlay.set_xlabel("Wavelength (nm)")
    ax_overlay.set_ylabel("Normalized Flux")
    ax_overlay.set_title("1. Overlaid Spectra (Fortran vs Python)")
    ax_overlay.set_xlim(args.wl_start, args.wl_end)
    ax_overlay.legend(loc="best")
    ax_overlay.grid(True, alpha=0.3)

    # Row 2: side by side (Python left, Fortran right)
    ax_python.plot(
        wavelength1,
        normalized_flux1,
        color="blue",
        linewidth=1,
    )
    ax_python.axhline(y=1, color="r", linestyle="--", alpha=0.5)
    ax_python.set_xlabel("Wavelength (nm)")
    ax_python.set_ylabel("Normalized Flux")
    ax_python.set_title("2a. Python Spectrum")
    ax_python.set_xlim(args.wl_start, args.wl_end)
    ax_python.grid(True, alpha=0.3)

    ax_fortran.plot(
        wavelength2,
        normalized_flux2,
        color="orange",
        linewidth=1,
    )
    ax_fortran.axhline(y=1, color="r", linestyle="--", alpha=0.5)
    ax_fortran.set_xlabel("Wavelength (nm)")
    ax_fortran.set_ylabel("Normalized Flux")
    ax_fortran.set_title("2b. Fortran Spectrum")
    ax_fortran.set_xlim(args.wl_start, args.wl_end)
    ax_fortran.grid(True, alpha=0.3)

    fig.suptitle(
        f"Normalized Spectra Comparison ({args.wl_start:g}-{args.wl_end:g} nm)"
    )

    # Row 3: fractional error
    error_range_mask = (wavelength_for_error >= args.wl_start) & (
        wavelength_for_error <= args.wl_end
    )
    wavelength_error = wavelength_for_error[error_range_mask]
    fractional_error = fractional_error_all[error_range_mask]
    fractional_error_robust = fractional_error_robust_all[error_range_mask]

    finite_error_mask = np.isfinite(fractional_error_robust)
    if np.sum(finite_error_mask) == 0:
        finite_error_mask = np.isfinite(fractional_error)
    if np.sum(finite_error_mask) > 0:
        if np.any(np.isfinite(fractional_error_robust)):
            finite_error = fractional_error_robust[finite_error_mask]
        else:
            finite_error = fractional_error[finite_error_mask]
        margin_error = 0.05 * (np.max(finite_error) - np.min(finite_error))
        ax_error.set_ylim(
            np.min(finite_error) - margin_error, np.max(finite_error) + margin_error
        )

    ax_error.plot(
        wavelength_error,
        fractional_error,
        color="gray",
        linewidth=0.7,
        alpha=0.25,
        label="Raw Fractional Error",
    )
    ax_error.plot(
        wavelength_error,
        fractional_error_robust,
        color="green",
        linewidth=1.0,
        label="Robust Fractional Error",
    )
    ax_error.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Zero Error")
    ax_error.set_xlabel("Wavelength (nm)")
    ax_error.set_ylabel("Fractional Error")
    ax_error.set_xlim(args.wl_start, args.wl_end)
    ax_error.set_title(
        "3. Fractional Error (raw + robust): (Python - Fortran) / Python"
    )
    ax_error.legend(loc="best")
    ax_error.grid(True, alpha=0.3)

    if args.save is not None:
        out_png = Path(args.save)
    else:
        out_png = BASE_DIR / "plots" / args.atmosphere.replace("/", "_")
        out_png = out_png.with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {out_png}")
    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
