"""Population and state computations for atmosphere layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from . import tables

if TYPE_CHECKING:  # pragma: no cover
    from ..io.atmosphere import AtmosphereModel


@dataclass
class HydrogenDepthState:
    pp: float
    fo: float
    y1b: float
    y1s: float
    t3nhe: float
    t3nh2: float
    c1d: float
    c2d: float
    gcon1: float
    gcon2: float
    xnfph: np.ndarray
    dopph: float


@dataclass
class DepthState:
    """Derived physical quantities for one atmospheric depth."""

    boltzmann_factor: np.ndarray
    doppler_width: np.ndarray
    turbulence_width: float
    electron_density: float
    temperature: float
    continuum_opacity: np.ndarray
    hckt: float
    txnxn: float
    hydrogen: Optional[HydrogenDepthState] = None


@dataclass
class Populations:
    """Holds precomputed populations for all depths."""

    layers: Dict[int, DepthState]


HCKT_COEFF = 11604.51812155008  # 1/(k_B) in eV/K (for eV energies)
HCKT_CM_COEFF = 1.4388  # hc/k in cm·K (for cm⁻¹ energies, from Fortran)
# Note: The catalog stores excitation_energy in cm⁻¹ units. Use HCKT_CM_COEFF for correct Boltzmann.
# However, there are other bugs in the code that compensate for using the wrong HCKT.
# Until those are found and fixed, using the eV coefficient produces better results.
KBOLTZ = 1.380649e-16  # erg/K
M_H = 1.6735575e-24  # g
C_LIGHT_KMS = 299_792.458
C_LIGHT_CMS = 2.99792458e10


def _hydrogen_state(
    temperature: float,
    electron_density: float,
    xnf_he1: float,
    xnf_h2: float,
    xnfph: np.ndarray,
    dopph: float,
) -> HydrogenDepthState:
    xne = max(electron_density, 1e-40)
    xne16 = xne ** (1.0 / 6.0)
    pp = xne16 * 0.08989 / np.sqrt(max(temperature, 1.0))
    fo = (xne16 ** 4) * 1.25e-9
    y1b = 2.0 / (1.0 + 0.012 / max(temperature, 1.0) * np.sqrt(xne / max(temperature, 1.0)))
    t4 = temperature / 10000.0
    t43 = t4 ** 0.3
    y1s = t43 / xne16
    t3nhe = t43 * xnf_he1
    t3nh2 = t43 * xnf_h2
    c1d = fo * 78940.0 / max(temperature, 1.0)
    c2d = (fo ** 2) / 5.96e-23 / xne
    gcon1 = 0.2 + 0.09 * np.sqrt(max(t4, 1e-12)) / (1.0 + xne / 1.0e13)
    gcon2 = 0.2 / (1.0 + xne / 1.0e15)
    return HydrogenDepthState(
        pp=float(pp),
        fo=float(fo),
        y1b=float(y1b),
        y1s=float(y1s),
        t3nhe=float(t3nhe),
        t3nh2=float(t3nh2),
        c1d=float(c1d),
        c2d=float(c2d),
        gcon1=float(gcon1),
        gcon2=float(gcon2),
        xnfph=np.asarray(xnfph, dtype=np.float64),
        dopph=float(dopph),
    )


def compute_depth_state(
    atmosphere: AtmosphereModel,
    line_wavelengths: np.ndarray,
    excitation_energy: np.ndarray,
    microturb_kms: float,
) -> Populations:
    """Compute LTE-like populations and Doppler widths per depth."""

    layers: Dict[int, DepthState] = {}
    line_wavelengths = np.asarray(line_wavelengths, dtype=np.float64)
    excitation_energy = np.asarray(excitation_energy, dtype=np.float64)

    for idx in range(atmosphere.layers):
        temp = max(float(atmosphere.temperature[idx]), 1.0)

        # CRITICAL: Use HCKT from atmosphere file (Fortran synthe.for line 268)
        # HCKT = hc/kT in cm units (computed by atlas7v.for, stored in fort.10/NPZ)
        # ELO is in cm⁻¹, so ELO * HCKT is dimensionless
        # This matches Fortran: KAPPA0 = KAPPA0 * FASTEX(ELO * HCKT(J))
        if atmosphere.hckt is not None and len(atmosphere.hckt) > idx:
            hckt = float(atmosphere.hckt[idx])
        else:
            # Fallback: compute hc/kT = 1.4388/T (in cm)
            hckt = 1.4388 / temp

        boltz = np.array([tables.fast_ex(float(energy) * hckt) for energy in excitation_energy], dtype=np.float64)

        thermal_velocity = np.sqrt(2.0 * KBOLTZ * temp / M_H) / C_LIGHT_CMS
        vturb_model = float(atmosphere.turbulent_velocity[idx]) if atmosphere.turbulent_velocity.size > idx else 0.0
        total_turb = np.hypot(microturb_kms, vturb_model) / C_LIGHT_KMS
        doppler_width = line_wavelengths * np.sqrt(total_turb**2 + thermal_velocity**2)

        if atmosphere.txnxn is not None:
            txnxn = float(atmosphere.txnxn[idx])
        else:
            # Compute TXNXN (perturber density for van der Waals broadening)
            # Original formula included a temperature scaling factor
            xnf_h = float(atmosphere.xnf_h[idx]) if atmosphere.xnf_h is not None else 0.0
            xnf_he1 = float(atmosphere.xnf_he1[idx]) if atmosphere.xnf_he1 is not None else 0.0
            xnf_h2 = float(atmosphere.xnf_h2[idx]) if atmosphere.xnf_h2 is not None else 0.0
            txnxn = (xnf_h + 0.42 * xnf_he1 + 0.85 * xnf_h2) * (temp / 10_000.0) ** 0.3

        hydrogen_state = None
        if atmosphere.xnf_he1 is not None or atmosphere.xnf_h2 is not None:
            xnf_he1 = float(atmosphere.xnf_he1[idx]) if atmosphere.xnf_he1 is not None else 0.0
            xnf_h2 = float(atmosphere.xnf_h2[idx]) if atmosphere.xnf_h2 is not None else 0.0
            xnfph = atmosphere.xnfph[idx] if atmosphere.xnfph is not None else np.zeros(2, dtype=np.float64)
            dopph = float(atmosphere.dopph[idx]) if atmosphere.dopph is not None else float(total_turb * C_LIGHT_KMS)
            hydrogen_state = _hydrogen_state(
                temperature=temp,
                electron_density=float(atmosphere.electron_density[idx]),
                xnf_he1=xnf_he1,
                xnf_h2=xnf_h2,
                xnfph=xnfph,
                dopph=dopph,
            )

        state = DepthState(
            boltzmann_factor=boltz,
            doppler_width=doppler_width,
            turbulence_width=float(total_turb * C_LIGHT_KMS),
            electron_density=float(atmosphere.electron_density[idx]),
            temperature=temp,
            continuum_opacity=np.zeros(line_wavelengths.size, dtype=np.float64),
            hckt=hckt,
            txnxn=txnxn,
            hydrogen=hydrogen_state,
        )
        layers[idx] = state
    return Populations(layers=layers)
