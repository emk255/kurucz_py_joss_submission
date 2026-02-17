#!/usr/bin/env python3
"""Parity harness: compare Python-compiled metadata against `tfort.*` ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synthe_py.io.lines import compiler, fort19 as fort19_io, tfort
import math

_DELLIM = np.array([100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1], dtype=np.float64)


def _float_delta(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size == 0:
        return {"max_abs": 0.0, "rms_abs": 0.0}
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms_abs": float(np.sqrt(np.mean(delta * delta))),
    }


def _int_exact(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size == 0:
        return {"exact_match_rate": 1.0, "mismatch_count": 0}
    eq = np.asarray(a) == np.asarray(b)
    mismatches = int(np.count_nonzero(~eq))
    return {"exact_match_rate": float(np.mean(eq)), "mismatch_count": mismatches}


def _align_tfort12(
    py_data: dict[str, np.ndarray],
    gt_data: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    def _keys(data: dict[str, np.ndarray]) -> list[tuple[int, int, float, float]]:
        return [
            (
                int(nbuff),
                int(nelion),
                round(float(elo), 6),
                round(float(cgf), 14),
            )
            for nbuff, nelion, elo, cgf in zip(
                data["nbuff"], data["nelion"], data["elo_cm"], data["cgf"]
            )
        ]

    py_keys = _keys(py_data)
    gt_keys = _keys(gt_data)
    py_index: dict[object, list[int]] = {}
    gt_index: dict[object, list[int]] = {}
    for i, key in enumerate(py_keys):
        py_index.setdefault(key, []).append(i)
    for i, key in enumerate(gt_keys):
        gt_index.setdefault(key, []).append(i)

    py_match_idx: list[int] = []
    gt_match_idx: list[int] = []
    for key in sorted(set(py_index.keys()) & set(gt_index.keys()), key=str):
        py_list = py_index[key]
        gt_list = gt_index[key]
        n = min(len(py_list), len(gt_list))
        py_match_idx.extend(py_list[:n])
        gt_match_idx.extend(gt_list[:n])

    py_arr = np.asarray(py_match_idx, dtype=int)
    gt_arr = np.asarray(gt_match_idx, dtype=int)
    py_aligned = {k: v[py_arr] for k, v in py_data.items()}
    gt_aligned = {k: v[gt_arr] for k, v in gt_data.items()}
    return py_aligned, gt_aligned


def _grid_params(
    wlbeg: float, wlend: float, resolution: float
) -> tuple[float, float, float, float, float]:
    ratio = 1.0 + 1.0 / resolution
    rlog = math.log(ratio)
    ixwlbeg = math.floor(math.log(wlbeg) / rlog)
    if math.exp(ixwlbeg * rlog) < wlbeg:
        ixwlbeg += 1
    wbegin = math.exp(ixwlbeg * rlog)
    return wlbeg, wlend, ratio, rlog, wbegin


def _t12_dict_from_records(records: list[tfort.Tape12Record]) -> dict[str, np.ndarray]:
    return {
        "nbuff": np.array([r.nbuff for r in records], dtype=np.int32),
        "cgf": np.array([r.cgf for r in records], dtype=np.float64),
        "nelion": np.array([r.nelion for r in records], dtype=np.int16),
        "elo_cm": np.array([r.elo_cm for r in records], dtype=np.float64),
        "gamma_rad": np.array([r.gamma_rad for r in records], dtype=np.float64),
        "gamma_stark": np.array([r.gamma_stark for r in records], dtype=np.float64),
        "gamma_vdw": np.array([r.gamma_vdw for r in records], dtype=np.float64),
    }


def _subset_dict(
    data: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in data.items()}


def _t12_window_mask(
    nbuff: np.ndarray,
    nelion: np.ndarray,
    cmp_wlbeg: float,
    cmp_wlend: float,
    wbegin: float,
    ratio: float,
) -> np.ndarray:
    if nbuff.size == 0:
        return np.zeros(0, dtype=bool)
    wl = wbegin * np.power(ratio, nbuff.astype(np.float64) - 1.0)
    delfactor = 1.0 if cmp_wlbeg <= 500.0 else cmp_wlbeg / 500.0
    # For fort.12 plain lines, LIM is usually 7 (0.1 nm), with H I forced to LIM=1.
    margin = np.where(nelion == 1, 100.0 * delfactor, 0.1 * delfactor)
    return (wl >= (cmp_wlbeg - margin)) & (wl <= (cmp_wlend + margin))


def _fort19_window_subset(
    data: fort19_io.Fort19Data,
    cmp_wlbeg: float,
    cmp_wlend: float,
) -> fort19_io.Fort19Data:
    if data.wavelength_vacuum.size == 0:
        return data
    delfactor = 1.0 if cmp_wlbeg <= 500.0 else cmp_wlbeg / 500.0
    limb_idx = np.clip(data.limb.astype(np.int32), 1, 7) - 1
    margin = _DELLIM[limb_idx] * delfactor
    mask = (data.wavelength_vacuum >= (cmp_wlbeg - margin)) & (
        data.wavelength_vacuum <= (cmp_wlend + margin)
    )
    return data.subset(np.nonzero(mask)[0])


def _align_fort19(
    py19: fort19_io.Fort19Data,
    gt19: fort19_io.Fort19Data,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    py_data = {
        "line_type": py19.line_type.astype(np.int16),
        "nbuff": py19.nbuff.astype(np.int32),
        "limb": py19.limb.astype(np.int32),
        "n_lower": py19.n_lower.astype(np.int16),
        "n_upper": py19.n_upper.astype(np.int16),
        "ion_index": py19.ion_index.astype(np.int16),
        "gf": py19.oscillator_strength.astype(np.float64),
        "wl": py19.wavelength_vacuum.astype(np.float64),
    }
    gt_data = {
        "line_type": gt19.line_type.astype(np.int16),
        "nbuff": gt19.nbuff.astype(np.int32),
        "limb": gt19.limb.astype(np.int32),
        "n_lower": gt19.n_lower.astype(np.int16),
        "n_upper": gt19.n_upper.astype(np.int16),
        "ion_index": gt19.ion_index.astype(np.int16),
        "gf": gt19.oscillator_strength.astype(np.float64),
        "wl": gt19.wavelength_vacuum.astype(np.float64),
    }

    def _keys(data: dict[str, np.ndarray]) -> list[tuple[float, int, int, int]]:
        # Stable identity key for fort.19 records independent of TYPE/NBUFF/LIMB,
        # so we can diagnose classification/indexing mismatches separately.
        # Exclude GF to avoid key-splitting when only oscillator strength differs.
        return [
            (
                round(float(wl), 8),
                int(ion),
                int(nlo),
                int(nup),
            )
            for wl, ion, nlo, nup in zip(
                data["wl"],
                data["ion_index"],
                data["n_lower"],
                data["n_upper"],
            )
        ]

    py_keys = _keys(py_data)
    gt_keys = _keys(gt_data)
    py_index: dict[object, list[int]] = {}
    gt_index: dict[object, list[int]] = {}
    for i, key in enumerate(py_keys):
        py_index.setdefault(key, []).append(i)
    for i, key in enumerate(gt_keys):
        gt_index.setdefault(key, []).append(i)

    py_match_idx: list[int] = []
    gt_match_idx: list[int] = []
    for key in sorted(set(py_index.keys()) & set(gt_index.keys()), key=str):
        py_list = py_index[key]
        gt_list = gt_index[key]
        n = min(len(py_list), len(gt_list))
        py_match_idx.extend(py_list[:n])
        gt_match_idx.extend(gt_list[:n])

    py_arr = np.asarray(py_match_idx, dtype=int)
    gt_arr = np.asarray(gt_match_idx, dtype=int)
    py_aligned = {k: v[py_arr] for k, v in py_data.items()}
    gt_aligned = {k: v[gt_arr] for k, v in gt_data.items()}
    stats = {
        "python_total": int(len(py_keys)),
        "ground_truth_total": int(len(gt_keys)),
        "matched_records": int(py_arr.size),
        "python_unmatched": int(len(py_keys) - py_arr.size),
        "ground_truth_unmatched": int(len(gt_keys) - gt_arr.size),
    }
    return py_aligned, gt_aligned, stats


def _fort19_key(
    value_wl: float, value_ion: int, value_nlo: int, value_nup: int, value_gf: float
) -> tuple[float, int, int, int, float]:
    return (
        round(float(value_wl), 8),
        int(value_ion),
        int(value_nlo),
        int(value_nup),
        round(float(value_gf), 10),
    )


def _fort19_set_stats(
    py19: fort19_io.Fort19Data, gt19: fort19_io.Fort19Data
) -> dict[str, object]:
    py_keys = {
        _fort19_key(wl, ion, nlo, nup, gf)
        for wl, ion, nlo, nup, gf in zip(
            py19.wavelength_vacuum,
            py19.ion_index,
            py19.n_lower,
            py19.n_upper,
            py19.oscillator_strength,
        )
    }
    gt_keys = {
        _fort19_key(wl, ion, nlo, nup, gf)
        for wl, ion, nlo, nup, gf in zip(
            gt19.wavelength_vacuum,
            gt19.ion_index,
            gt19.n_lower,
            gt19.n_upper,
            gt19.oscillator_strength,
        )
    }
    extra = sorted(py_keys - gt_keys)[:20]
    missing = sorted(gt_keys - py_keys)[:20]
    return {
        "python_unique_keys": int(len(py_keys)),
        "ground_truth_unique_keys": int(len(gt_keys)),
        "intersection": int(len(py_keys & gt_keys)),
        "python_extra_key_count": int(len(py_keys - gt_keys)),
        "ground_truth_missing_key_count": int(len(gt_keys - py_keys)),
        "python_extra_key_samples": [list(x) for x in extra],
        "ground_truth_missing_key_samples": [list(x) for x in missing],
    }


def _fort19_structural_set_stats(
    py19: fort19_io.Fort19Data,
    gt19: fort19_io.Fort19Data,
) -> dict[str, int]:
    py_keys = {
        (
            round(float(wl), 8),
            int(ion),
            int(nlo),
            int(nup),
        )
        for wl, ion, nlo, nup in zip(
            py19.wavelength_vacuum,
            py19.ion_index,
            py19.n_lower,
            py19.n_upper,
        )
    }
    gt_keys = {
        (
            round(float(wl), 8),
            int(ion),
            int(nlo),
            int(nup),
        )
        for wl, ion, nlo, nup in zip(
            gt19.wavelength_vacuum,
            gt19.ion_index,
            gt19.n_lower,
            gt19.n_upper,
        )
    }
    return {
        "python_unique_keys": int(len(py_keys)),
        "ground_truth_unique_keys": int(len(gt_keys)),
        "intersection": int(len(py_keys & gt_keys)),
        "python_extra_key_count": int(len(py_keys - gt_keys)),
        "ground_truth_missing_key_count": int(len(gt_keys - py_keys)),
    }


def run_parity(
    gfall: Path,
    tfort12_path: Path,
    tfort19_path: Path,
    wlbeg: float,
    wlend: float,
    resolution: float,
) -> dict:
    t93_path = tfort12_path.with_suffix(".93")
    if t93_path.exists():
        t93 = tfort.parse_tfort93(t93_path)
        grid_wlbeg = float(t93.wlbeg)
        grid_wlend = float(t93.wlend)
        grid_res = float(t93.resolution) if t93.resolution > 0.0 else resolution
    else:
        grid_wlbeg = wlbeg
        grid_wlend = wlend
        grid_res = resolution

    compiled = compiler.compile_atomic_catalog(
        catalog_path=gfall,
        wlbeg=grid_wlbeg,
        wlend=grid_wlend,
        resolution=grid_res,
        line_filter=True,
    )

    t12 = list(tfort.parse_tfort12(tfort12_path))
    _, _, ratio, _, wbegin = _grid_params(grid_wlbeg, grid_wlend, grid_res)
    py12_all = {
        "nbuff": compiled.nbuff.astype(np.int32),
        "cgf": compiled.cgf.astype(np.float64),
        "nelion": compiled.nelion.astype(np.int16),
        "elo_cm": compiled.elo_cm.astype(np.float64),
        "gamma_rad": compiled.gamma_rad.astype(np.float64),
        "gamma_stark": compiled.gamma_stark.astype(np.float64),
        "gamma_vdw": compiled.gamma_vdw.astype(np.float64),
    }
    gt12_all = _t12_dict_from_records(t12)
    py12_mask = _t12_window_mask(
        py12_all["nbuff"], py12_all["nelion"], wlbeg, wlend, wbegin, ratio
    )
    gt12_mask = _t12_window_mask(
        gt12_all["nbuff"], gt12_all["nelion"], wlbeg, wlend, wbegin, ratio
    )
    py12, gt12 = _align_tfort12(
        _subset_dict(py12_all, py12_mask),
        _subset_dict(gt12_all, gt12_mask),
    )
    n = py12["nbuff"].size
    t19 = _fort19_window_subset(fort19_io.load(tfort19_path), wlbeg, wlend)
    py19_src = _fort19_window_subset(compiled.fort19_data, wlbeg, wlend)
    py19, gt19, align19_stats = _align_fort19(py19_src, t19)
    n19 = py19["line_type"].size

    report = {
        "window_nm": [wlbeg, wlend],
        "resolution": resolution,
        "python_records": int(np.count_nonzero(py12_mask)),
        "tfort12_records": int(np.count_nonzero(gt12_mask)),
        "compared_tfort12_records": int(n),
        "fields": {
            "nbuff": _int_exact(py12["nbuff"], gt12["nbuff"]),
            "nelion": _int_exact(py12["nelion"], gt12["nelion"]),
            "cgf": _float_delta(py12["cgf"], gt12["cgf"]),
            "elo_cm": _float_delta(py12["elo_cm"], gt12["elo_cm"]),
            "gamma_rad": _float_delta(py12["gamma_rad"], gt12["gamma_rad"]),
            "gamma_stark": _float_delta(py12["gamma_stark"], gt12["gamma_stark"]),
            "gamma_vdw": _float_delta(py12["gamma_vdw"], gt12["gamma_vdw"]),
        },
        "fort19": {
            "python_records": int(py19_src.wavelength_vacuum.size),
            "tfort19_records": int(t19.wavelength_vacuum.size),
            "compared_records": int(n19),
            "alignment": align19_stats,
            "structural_set_comparison": _fort19_structural_set_stats(
                py19_src,
                t19,
            ),
            "set_comparison": _fort19_set_stats(py19_src, t19),
            "line_type_exact": _int_exact(
                py19["line_type"],
                gt19["line_type"],
            ),
            "nbuff_exact": _int_exact(
                py19["nbuff"],
                gt19["nbuff"],
            ),
            "limb_exact": _int_exact(
                py19["limb"],
                gt19["limb"],
            ),
            "gf_delta": _float_delta(
                py19["gf"],
                gt19["gf"],
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Python metadata to tfort ground truth"
    )
    parser.add_argument("--gfall", type=Path, required=True)
    parser.add_argument("--tfort12", type=Path, required=True)
    parser.add_argument("--tfort19", type=Path, required=True)
    parser.add_argument("--wlbeg", type=float, required=True)
    parser.add_argument("--wlend", type=float, required=True)
    parser.add_argument("--resolution", type=float, default=300000.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run_parity(
        gfall=args.gfall,
        tfort12_path=args.tfort12,
        tfort19_path=args.tfort19,
        wlbeg=args.wlbeg,
        wlend=args.wlend,
        resolution=args.resolution,
    )
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
