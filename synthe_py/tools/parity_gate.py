#!/usr/bin/env python3
"""Run parity gates for metadata and staged pipeline comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from synthe_py.tools import line_metadata_parity, stage_compare


def _parse_timeout_value(value: str) -> float | None:
    token = value.strip().lower()
    if token in {"none", "no", "off", "0", "0.0"}:
        return None
    numeric = float(token)
    return None if numeric <= 0.0 else numeric


def _serialize_stage_results(
    results: dict[str, stage_compare.HarnessResult],
) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for name, res in results.items():
        serialized[name] = {
            "atmosphere": res.atmosphere,
            "wl_start": res.wl_start,
            "wl_end": res.wl_end,
            "first_failure": res.first_failure,
            "wall_time_python": res.wall_time_python,
            "wall_time_fortran": res.wall_time_fortran,
            "timings": res.timings,
            "stages": [
                {
                    "stage": s.stage,
                    "max_rel_err": s.max_rel_err,
                    "mean_rel_err": s.mean_rel_err,
                    "rms_rel_err": s.rms_rel_err,
                    "n_compared": s.n_compared,
                    "passed": s.passed,
                }
                for s in res.stages
            ],
        }
    return serialized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run metadata + stage parity gates")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--wlbeg", type=float, default=368.0)
    parser.add_argument("--wlend", type=float, default=372.0)
    parser.add_argument("--resolution", type=float, default=300000.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for JSON reports (default: <root>/stage_diag/parity_gate)",
    )
    parser.add_argument(
        "--skip-stage-compare",
        action="store_true",
        help="Only run metadata parity gate",
    )
    parser.add_argument(
        "--atmospheres",
        nargs="+",
        default=["t05770g4.44", "t03750g3.50", "t02500g-1.0"],
        help="Atmosphere keys for stage compare runs",
    )
    parser.add_argument(
        "--tfort12",
        type=Path,
        default=None,
        help="Path to Fortran reference tfort.12 (default: <root>/synthe/Lines_v5_PL_subrange/tfort.12)",
    )
    parser.add_argument(
        "--tfort19",
        type=Path,
        default=None,
        help="Path to Fortran reference tfort.19 (default: <root>/synthe/Lines_v5_PL_subrange/tfort.19)",
    )
    parser.add_argument(
        "--python-timeout",
        type=str,
        default=None,
        help="Python stage timeout in seconds; use 'none' or '0' for no timeout",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = (
        args.outdir.resolve() if args.outdir else (root / "stage_diag" / "parity_gate")
    )
    outdir.mkdir(parents=True, exist_ok=True)
    tfort12_path = (
        args.tfort12.resolve()
        if args.tfort12
        else (root / "synthe" / "Lines_v5_PL_subrange" / "tfort.12")
    )
    tfort19_path = (
        args.tfort19.resolve()
        if args.tfort19
        else (root / "synthe" / "Lines_v5_PL_subrange" / "tfort.19")
    )
    python_timeout = _parse_timeout_value(args.python_timeout)

    metadata_report = line_metadata_parity.run_parity(
        gfall=root / "lines" / "gfallvac.latest",
        tfort12_path=tfort12_path,
        tfort19_path=tfort19_path,
        wlbeg=args.wlbeg,
        wlend=args.wlend,
        resolution=args.resolution,
    )
    (outdir / "line_metadata_parity.json").write_text(
        json.dumps(metadata_report, indent=2) + "\n",
        encoding="utf-8",
    )

    stage_payload: dict[str, Any] = {}
    if not args.skip_stage_compare:
        stage_results = stage_compare.run_all_atmospheres(
            kurucz_root=root,
            wl_start=args.wlbeg,
            wl_end=args.wlend,
            threshold_pct=args.threshold,
            n_workers=1,
            atmospheres=args.atmospheres,
            python_timeout=python_timeout,
        )
        stage_payload = _serialize_stage_results(stage_results)
        (outdir / "stage_compare.json").write_text(
            json.dumps(stage_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    final_report = {
        "window_nm": [args.wlbeg, args.wlend],
        "resolution": args.resolution,
        "threshold_pct": args.threshold,
        "python_timeout": python_timeout,
        "tfort12_path": str(tfort12_path),
        "tfort19_path": str(tfort19_path),
        "metadata_report_path": str(outdir / "line_metadata_parity.json"),
        "stage_report_path": (
            str(outdir / "stage_compare.json") if stage_payload else None
        ),
    }
    (outdir / "parity_gate_summary.json").write_text(
        json.dumps(final_report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
