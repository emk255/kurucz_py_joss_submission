"""Parser for SYNTHE fort.19 wing metadata tapes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterator, Iterable, Sequence
from enum import IntEnum

import numpy as np

_RECORD_STRUCT = struct.Struct("<dffiiiiii fffii".replace(" ", ""))


class Fort19WingType(IntEnum):
    """Semantic categorisation of fort.19 line types."""

    HYDROGEN = -1
    DEUTERIUM = -2
    HELIUM_4 = -3
    HELIUM_3 = -4
    HELIUM_3_II = -6
    NORMAL = 0
    AUTOIONIZING = 1
    CORONAL = 2
    PRD = 3
    CONTINUUM = 100
    UNKNOWN = 101

    @classmethod
    def from_code(cls, code: int) -> "Fort19WingType":
        if code in {-6, -4, -3, -2, -1, 0, 1, 2, 3}:
            return cls(code)  # type: ignore[arg-type]
        if code > 3:
            return cls.CONTINUUM
        return cls.UNKNOWN


def _iter_records(handle) -> Iterator[tuple[float, ...]]:
    """Yield unpacked fort.19 records from a binary handle."""

    while True:
        header = handle.read(4)
        if not header:
            break
        (size,) = struct.unpack("<i", header)
        payload = handle.read(size)
        trailer = handle.read(4)
        if len(payload) != size or len(trailer) != 4:
            raise ValueError("Truncated fort.19 record")
        (check,) = struct.unpack("<i", trailer)
        if check != size:
            raise ValueError("fort.19 record length mismatch")
        if size != _RECORD_STRUCT.size:
            raise ValueError(f"Unexpected fort.19 record size {size}")
        yield _RECORD_STRUCT.unpack(payload)


@dataclass(frozen=True)
class Fort19Data:
    """Structured access to fort.19 wing records."""

    wavelength_vacuum: np.ndarray
    energy_lower: np.ndarray
    oscillator_strength: np.ndarray
    n_lower: np.ndarray
    n_upper: np.ndarray
    ion_index: np.ndarray
    line_type: np.ndarray
    continuum_index: np.ndarray
    element_index: np.ndarray
    gamma_rad: np.ndarray
    gamma_stark: np.ndarray
    gamma_vdw: np.ndarray
    nbuff: np.ndarray
    limb: np.ndarray
    wing_type: np.ndarray

    def indices_for(self, wing_type: Fort19WingType) -> np.ndarray:
        """Return the indices of records matching the requested wing type."""
        return np.nonzero(self.wing_type == wing_type)[0]

    def iter_indices(self, wing_types: Iterable[Fort19WingType]) -> np.ndarray:
        """Return indices matching any of the supplied wing types."""
        mask = np.zeros_like(self.wing_type, dtype=bool)
        for wtype in wing_types:
            mask |= self.wing_type == wtype
        return np.nonzero(mask)[0]

    def subset(self, indices: Sequence[int]) -> "Fort19Data":
        """Return a new Fort19Data limited to the specified indices."""
        idx = np.asarray(indices, dtype=int)
        return Fort19Data(
            wavelength_vacuum=self.wavelength_vacuum[idx],
            energy_lower=self.energy_lower[idx],
            oscillator_strength=self.oscillator_strength[idx],
            n_lower=self.n_lower[idx],
            n_upper=self.n_upper[idx],
            ion_index=self.ion_index[idx],
            line_type=self.line_type[idx],
            continuum_index=self.continuum_index[idx],
            element_index=self.element_index[idx],
            gamma_rad=self.gamma_rad[idx],
            gamma_stark=self.gamma_stark[idx],
            gamma_vdw=self.gamma_vdw[idx],
            nbuff=self.nbuff[idx],
            limb=self.limb[idx],
            wing_type=self.wing_type[idx],
        )


def _classify_line_types(line_type: np.ndarray) -> np.ndarray:
    """Vectorised helper returning Fort19WingType per record."""

    vectorized = np.vectorize(lambda value: Fort19WingType.from_code(int(value)), otypes=[object])
    return vectorized(line_type)

def load(path: Path) -> Fort19Data:
    """Load a fort.19 file into NumPy arrays."""

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            line_type = np.asarray(data["line_type"], dtype=np.int16)
            stored_wing = data.get("wing_type")
            if stored_wing is not None:
                wing_type = np.asarray(
                    [Fort19WingType.from_code(int(code)) for code in stored_wing],
                    dtype=object,
                )
            else:
                wing_type = _classify_line_types(line_type)
            return Fort19Data(
                wavelength_vacuum=np.asarray(data["wavelength_vacuum"], dtype=np.float64),
                energy_lower=np.asarray(data["energy_lower"], dtype=np.float32),
                oscillator_strength=np.asarray(data["oscillator_strength"], dtype=np.float32),
                n_lower=np.asarray(data["n_lower"], dtype=np.int16),
                n_upper=np.asarray(data["n_upper"], dtype=np.int16),
                ion_index=np.asarray(data["ion_index"], dtype=np.int16),
                line_type=line_type,
                continuum_index=np.asarray(data["continuum_index"], dtype=np.int16),
                element_index=np.asarray(data["element_index"], dtype=np.int16),
                gamma_rad=np.asarray(data["gamma_rad"], dtype=np.float32),
                gamma_stark=np.asarray(data["gamma_stark"], dtype=np.float32),
                gamma_vdw=np.asarray(data["gamma_vdw"], dtype=np.float32),
                nbuff=np.asarray(data["nbuff"], dtype=np.int32),
                limb=np.asarray(data["limb"], dtype=np.int32),
                wing_type=wing_type,
            )

    wavelengths: list[float] = []
    energies: list[float] = []
    gfs: list[float] = []
    nblo: list[int] = []
    nbup: list[int] = []
    nelion: list[int] = []
    linetype: list[int] = []
    ncon: list[int] = []
    nelionx: list[int] = []
    gamma_r: list[float] = []
    gamma_s: list[float] = []
    gamma_w: list[float] = []
    nbuff_vals: list[int] = []
    limb_vals: list[int] = []

    with path.open("rb") as fh:
        for record in _iter_records(fh):
            (
                wl_vac,
                elo,
                gf,
                n_lower,
                n_upper,
                ion,
                line_type,
                continuum_idx,
                elem_idx,
                gamma_rad,
                gamma_stark,
                gamma_vdw,
                nbuff_val,
                limb_val,
            ) = record

            wavelengths.append(wl_vac)
            energies.append(elo)
            gfs.append(gf)
            nblo.append(n_lower)
            nbup.append(n_upper)
            nelion.append(ion)
            linetype.append(line_type)
            ncon.append(continuum_idx)
            nelionx.append(elem_idx)
            gamma_r.append(gamma_rad)
            gamma_s.append(gamma_stark)
            gamma_w.append(gamma_vdw)
            nbuff_vals.append(nbuff_val)
            limb_vals.append(limb_val)

    line_type_array = np.asarray(linetype, dtype=np.int16)
    return Fort19Data(
        wavelength_vacuum=np.asarray(wavelengths, dtype=np.float64),
        energy_lower=np.asarray(energies, dtype=np.float32),
        oscillator_strength=np.asarray(gfs, dtype=np.float32),
        n_lower=np.asarray(nblo, dtype=np.int16),
        n_upper=np.asarray(nbup, dtype=np.int16),
        ion_index=np.asarray(nelion, dtype=np.int16),
        line_type=line_type_array,
        continuum_index=np.asarray(ncon, dtype=np.int16),
        element_index=np.asarray(nelionx, dtype=np.int16),
        gamma_rad=np.asarray(gamma_r, dtype=np.float32),
        gamma_stark=np.asarray(gamma_s, dtype=np.float32),
        gamma_vdw=np.asarray(gamma_w, dtype=np.float32),
        nbuff=np.asarray(nbuff_vals, dtype=np.int32),
        limb=np.asarray(limb_vals, dtype=np.int32),
        wing_type=_classify_line_types(line_type_array),
    )


__all__ = ["Fort19Data", "Fort19WingType", "load"]
