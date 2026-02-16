import argparse
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
        description="Plot overlaid Python/Fortran spectra with fractional error."
    )
    parser.add_argument(
        "--atmosphere",
        type=str,
        default=DEFAULT_ATMOSPHERE,
        help="Spectrum filename under results/validation_100/{python_specs,fortran_specs}.",
    )
    parser.add_argument(
        "--python-spec",
        type=str,
        default=None,
        help="Explicit Python spectrum path (overrides --atmosphere).",
    )
    parser.add_argument(
        "--fortran-spec",
        type=str,
        default=None,
        help="Explicit Fortran spectrum path (overrides --atmosphere).",
    )
    parser.add_argument("--wl-start", type=float, default=DEFAULT_WL_START)
    parser.add_argument("--wl-end", type=float, default=DEFAULT_WL_END)
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output PNG path (default: results/validation_100/plots/<atmosphere>.png).",
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

    fractional_error_all = (
        normalized_flux_ref_all - normalized_flux_spec_all
    ) / normalized_flux_ref_all

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

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    ax = axes[0]
    ax_error = axes[1]

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
        ax.set_ylim(np.min(all_finite_flux) - margin, np.max(all_finite_flux) + margin)

    ax.plot(
        wavelength1,
        normalized_flux1,
        color="blue",
        label=f"Spectrum 1: {path1}",
        alpha=0.7,
        linewidth=1,
    )
    ax.plot(
        wavelength2,
        normalized_flux2,
        color="orange",
        label=f"Spectrum 2: {path2}",
        alpha=0.7,
        linewidth=1,
    )
    ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Continuum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(
        f"Overlaid Normalized Spectra Comparison ({args.wl_start:g}-{args.wl_end:g} nm)"
    )
    ax.set_xlim(args.wl_start, args.wl_end)
    ax.legend(loc="best")

    count1 = np.sum(mask1)
    count2 = np.sum(mask2)
    ax.text(
        0.98,
        0.95,
        (
            f"Spectrum 1: Flux > continuum: {count1} / {len(flux1)} ({100*count1/len(flux1):.2f}%)\n"
            f"Spectrum 2: Flux > continuum: {count2} / {len(flux2)} ({100*count2/len(flux2):.2f}%)"
        ),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    finite_error_mask = np.isfinite(fractional_error_all)
    if np.sum(finite_error_mask) > 0:
        finite_error = fractional_error_all[finite_error_mask]
        margin_error = 0.05 * (np.max(finite_error) - np.min(finite_error))
        ax_error.set_ylim(
            np.min(finite_error) - margin_error, np.max(finite_error) + margin_error
        )

    ax_error.plot(
        wavelength_for_error, fractional_error_all, color="green", linewidth=1
    )
    ax_error.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Zero Error")
    ax_error.set_xlabel("Wavelength (nm)")
    ax_error.set_ylabel("Fractional Error")
    ax_error.set_ylim(-1, 1)
    ax_error.set_title(
        "Fractional Error: (Spectrum 1 - Spectrum 2) / Spectrum 1 (All Wavelengths)"
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
    plt.show()


if __name__ == "__main__":
    main()
