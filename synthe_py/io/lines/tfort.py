"""Parsers for the legacy SYNTHE tape-12/14 binary line lists."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

_ANGSTROM_PER_CM = 1e8
_CM_INV_PER_EV = 8065.54429


_ELEMENT_SYMBOLS: List[str] = [
    "",  # placeholder for 0 index
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
]


@dataclass
class Tape14Record:
    wavelength_vac: float
    wavelength_air: float
    excitation_energy_cm: float
    log_gf: float
    gf: float
    gamma_rad: float
    gamma_stark: float
    gamma_vdw: float
    element_symbol: str
    ion_stage: int
    n_lower: Optional[int]
    n_upper: Optional[int]


def _iter_tape14(path: Path) -> Iterator[Tape14Record]:
    record_len = 14 * 8 + 28 * 4
    with path.open("rb") as handle:
        while True:
            header = handle.read(4)
            if not header:
                break
            (payload_len,) = struct.unpack("<i", header)
            payload = handle.read(payload_len)
            trailer = handle.read(4)
            if len(payload) != payload_len or len(trailer) != 4:
                raise ValueError("Truncated tape-14 record")
            if payload_len != record_len:
                raise ValueError(
                    f"Unexpected tape-14 record length {payload_len} (expected {record_len})"
                )

            doubles = struct.unpack("<14d", payload[: 14 * 8])

            offset = 14 * 8
            nelion = struct.unpack_from("<i", payload, offset)[0]
            offset += 4
            gamma_rad = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            gamma_stark = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            gamma_vdw = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            offset += 4  # REF (unused)
            nblo = int(round(struct.unpack_from("<f", payload, offset)[0]))
            offset += 4
            nbup = int(round(struct.unpack_from("<f", payload, offset)[0]))
            offset += 4
            offset += 8  # ISO1/X1
            offset += 8  # ISO2/X2
            log_gf = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            offset += 8  # XJ/XJP
            code = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            excitation = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            gf = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            gamma_s_log = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            gamma_r_log = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            gamma_w_log = struct.unpack_from("<f", payload, offset)[0]

            # Use the exponentiated values if raw rates are zero
            if gamma_rad <= 0.0 and gamma_r_log != 0.0:
                gamma_rad = 10.0 ** gamma_r_log
            if gamma_stark <= 0.0 and gamma_s_log != 0.0:
                gamma_stark = 10.0 ** gamma_s_log
            if gamma_vdw <= 0.0 and gamma_w_log != 0.0:
                gamma_vdw = 10.0 ** gamma_w_log

            nelem = int(code + 1e-6)
            frac = code - nelem
            ion_stage = int(round(frac * 100.0)) + 1 if frac > 1e-6 else 1
            symbol = ""
            if 1 <= nelem < len(_ELEMENT_SYMBOLS):
                symbol = _ELEMENT_SYMBOLS[nelem]
            else:
                symbol = f"Z{nelem}"

            yield Tape14Record(
                wavelength_vac=float(doubles[11]),
                wavelength_air=float(doubles[0]),
                excitation_energy_cm=float(excitation),
                log_gf=float(log_gf),
                gf=float(gf) if gf > 0.0 else float(10.0 ** log_gf),
                gamma_rad=float(gamma_rad),
                gamma_stark=float(gamma_stark),
                gamma_vdw=float(gamma_vdw),
                element_symbol=symbol,
                ion_stage=ion_stage,
                n_lower=nblo if nblo > 0 else None,
                n_upper=nbup if nbup > 0 else None,
            )


def parse_tfort14(path: Path) -> Iterable[Tape14Record]:
    return list(_iter_tape14(path))


def find_companion_tape14(source: Path) -> Optional[Path]:
    if source.suffix == ".14":
        return source
    if source.suffix == ".12":
        candidate = source.with_suffix(".14")
        if candidate.exists():
            return candidate
        candidate_alt = source.parent / (source.stem + "4")
        if candidate_alt.exists():
            return candidate_alt
    return None


