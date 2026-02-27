---
title: 'PySynthe: A Self-Contained Python Reimplementation of Kurucz Synthe'
tags:
  - Python
  - astronomy
  - solar spectroscopy
authors:
  - name: Elliot M. Kim
    orcid: 0009-0000-9991-0191
    affiliation: 1
  - name: Yuan-Sen Ting
    affiliation: 2
affiliations:
 - name: Cornell University
   index: 1
   ror: 05bnh6r87 
 - name: Ohio State University
   index: 2
   ror: 00rs6vg23
date: 28 February 2026
bibliography: paper.bib
---

# Summary

PySynthe is a self-contained Python reimplementation of Kurucz's SYNTHE spectrum synthesis code. Given a model atmosphere (`.atm` file) and an atomic line list, it computes synthetic stellar spectra over a specified wavelength range and resolution. The implementation is pure Python (with NumPy, SciPy, and Numba for performance) and requires no Fortran compiler or runtime. Users run synthesis via a simple pipeline: convert the atmosphere to an internal NPZ format, then compute the spectrum. Validation against Fortran reference spectra is supported through a comparison tool that reports flux and continuum agreement metrics.

# Statement of need

The original Kurucz SYNTHE code is written in Fortran and has been the standard tool for stellar spectrum synthesis for decades. A Python reimplementation offers several benefits. First, it enables direct integration with machine learning workflows—e.g., training neural network emulators or surrogate models on synthetic spectra, or using differentiable synthesis in gradient-based optimization. Second, it interoperates with the broader Python ecosystem (NumPy, SciPy, Astropy, etc.) for data handling, visualization, and analysis. Third, it provides greater flexibility for customization and extension without modifying or recompiling Fortran. Fourth, it removes the need for a Fortran compiler and runtime, simplifying installation and cross-platform use. Finally, Python's readability and tooling make the code easier to debug, profile, and adapt for research and teaching.

# State of the field                                                                                                               

# Software design

PyKurucz is organized as a Python package (`synthe_py`) with three main layers. The **I/O layer** (`synthe_py.io`) handles atmosphere loading (`.atm` and `.npz`), atomic line catalog parsing and compilation (GFALL format), and spectrum export. The **physics layer** (`synthe_py.physics`) implements opacity sources (continuum via KAPP, line opacity with Voigt profiles, hydrogen and helium broadening), population calculations (Saha, PFSAHA), and molecular equilibrium (NMOLEC). The **engine layer** (`synthe_py.engine`) orchestrates opacity computation, radiative transfer, and spectrum synthesis. A separate `tools` subpackage provides the atmosphere converter (`convert_atm_to_npz`), spectrum comparison (`compare_spectra`), and data extraction utilities. The pipeline is invoked via a shell script or the `synthe_py.cli` module. Line data is cached (parsed and compiled) to avoid repeated parsing of large line lists. Performance-critical routines use Numba JIT compilation.

PyKurucz intentionally reuses the same input data as the original Fortran SYNTHE/ATLAS code. The atomic line list (`gfallvac.latest`, Kurucz GFALL format) is read directly and compiled into a cached NumPy archive on first use. Auxiliary physics tables — including partition functions, ionization potentials, continuum opacity tables, and molecular equilibrium data (`molecules.dat`, `continua.dat`) — are pre-extracted from Fortran data files and stored as `.npz` archives under `synthe_py/data/`. This design means the Python pipeline operates on the same underlying physical data as the Fortran ground truth, isolating any differences to the numerical implementation rather than input data discrepancies.

# Research impact statement

# Mathematics

# Citations

# Figures
![Comparison of solar atmosphere model (Teff=5777K, logg=4.44, [Fe/H]=0)](compare-at12_solar.png)
*Figure 1: Comparison of the PySynthe-generated spectrum and the Fortran Synthe reference for a solar atmosphere.*

![Comparison of cool giant atmosphere (Teff=3500K, logg=0.5, [Fe/H]=0)](compare-at12_coolgiant.png)
*Figure 2: Residuals for a cool giant (low gravity, cool temperature) demonstrating PySynthe’s accuracy even for extended, molecule-rich atmospheres. Main differences trace to molecular bands in extremely cool stars.*

![Comparison of metal-poor dwarf atmosphere (Teff=6200K, logg=4.0, [Fe/H]=-2.0)](compare-at12_mp6200.png)
*Figure 3: Synthesis agreement for a metal-poor dwarf, showing that PySynthe correctly reproduces spectra for low-metallicity conditions where metal line opacity is much reduced.*

![Comparison of hot O-star atmosphere (Teff=44000K, logg=4.5, [Fe/H]=0)](compare-at12_t44000g450.png)
*Figure 4: Comparison for a hot, massive O-type star. The code robustly handles the high-energy regime including strong H and He lines and reproduces the continuum level and line strengths versus the Fortran Synthe output.*


# AI usage disclosure

A substantial part of this reimplementation was developed with AI assistance (Cursor, using a collection of language models), given the ground-truth Fortran code provided by Bob Kurucz.

# Acknowledgements

# References
