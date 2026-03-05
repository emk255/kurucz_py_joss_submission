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
---

# Summary

PySynthe is a self-contained Python reimplementation of Kurucz's SYNTHE spectrum synthesis code. Given a model atmosphere (`.atm` file) and an atomic line list, it computes synthetic stellar spectra over a specified wavelength range and resolution. The implementation is pure Python (with NumPy, SciPy, and Numba for performance) and requires no Fortran compiler or runtime. Users run synthesis via a simple pipeline: convert the atmosphere to an internal NPZ format, then compute the spectrum. Validation against Fortran reference spectra is supported through a comparison tool that reports flux and continuum agreement metrics.

# Statement of need

The original Kurucz SYNTHE code is written in Fortran and has been the standard tool for stellar spectrum synthesis for decades. Self-consistent spectrum synthesis requires two steps: ATLAS12 computes the atmospheric structure (producing a `.atm` file from Teff, logg, and abundances), and SYNTHE performs radiative transfer to obtain the emergent spectrum. A Python implementation of ATLAS12 already exists; this work provides the missing SYNTHE component, completing an end-to-end Python pipeline. A Python reimplementation offers several benefits. First, it enables direct integration with machine learning workflows—e.g., training neural network emulators or surrogate models on synthetic spectra, or using differentiable synthesis in gradient-based optimization. Second, it interoperates with the broader Python ecosystem (NumPy, SciPy, Astropy, etc.) for data handling, visualization, and analysis. Third, it provides greater flexibility for customization and extension without modifying or recompiling Fortran. Fourth, it removes the need for a Fortran compiler and runtime, simplifying installation and cross-platform use. Finally, Python's readability and tooling make the code easier to debug, profile, and adapt for research and teaching.

# State of the field        

Stellar spectrum synthesis has long been dominated by Fortran-based codes. The most widely used include Kurucz's SYNTHE [Kurucz 2005](https://ui.adsabs.harvard.edu/abs/2005MSAIS...8...14K/abstract), MOOG [Sneden 1973](https://ui.adsabs.harvard.edu/abs/1973ApJ...184..839S/abstract), Turbospectrum [Plez 2012](https://ui.adsabs.harvard.edu/abs/2012ascl.soft05004P), and SPECTRUM [Gray 1999](https://ui.adsabs.harvard.edu/abs/1999ascl.soft10002G/abstract), each of which has served as a community standard for decades. These codes are highly accurate but require Fortran toolchains, are difficult to install across operating systems, lack documentation, and do not integrate naturally with modern Python-based workflows.

Several efforts have been made to bring spectrum synthesis into the Python ecosystem. Tools such as iSpec [Blanco-Cuaresma et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...569A..111B/abstract) and pyMOOG [Adamow 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.1820A/abstract) provide Python interfaces to underlying Fortran engines, but still require a working Fortran installation and do not expose the physics layer to Python. A different class of approaches use machine learning to approximate synthetic spectra interpolated from pre-computed grids (The Cannon [Ness et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N/abstract), The Payne [Ting et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T/abstract)). While powerful for large-scale survey analysis, these are statistical approximations.

PySynthe occupies a distinct position: it is a faithful numerical reimplementation of SYNTHE, validated against the Fortran ground truth, that requires no Fortran and exposes the full synthesis pipeline as importable Python. This makes it the first tool to offer ground-truth-quality synthesis within a pure Python environment.

# Software design

PySynthe is organized as a Python package (`synthe_py`) with three main layers. The **I/O layer** (`synthe_py.io`) handles atmosphere loading (`.atm` and `.npz`), atomic line catalog parsing and compilation (GFALL format), and spectrum export. The **physics layer** (`synthe_py.physics`) implements opacity sources (continuum via KAPP, line opacity with Voigt profiles, hydrogen and helium broadening), population calculations (Saha, PFSAHA), and molecular equilibrium (NMOLEC). The **engine layer** (`synthe_py.engine`) orchestrates opacity computation, radiative transfer, and spectrum synthesis. A separate `tools` subpackage provides the atmosphere converter (`convert_atm_to_npz`), spectrum comparison (`compare_spectra`), and data extraction utilities. The pipeline is invoked via a shell script or the `synthe_py.cli` module. Line data is cached (parsed and compiled) to avoid repeated parsing of large line lists. Performance-critical routines use Numba JIT compilation.

PySynthe intentionally reuses the same input data as the original Fortran SYNTHE/ATLAS code. The atomic line list (`gfallvac.latest`, Kurucz GFALL format) is read directly and compiled into a cached NumPy archive on first use. Partition functions, ionization potentials, continuum opacity tables, and molecular equilibrium data are pre-extracted from Fortran data files and stored as `.npz` archives under `synthe_py/data/`. The molecular and continuum edge data (`molecules.dat`, `continua.dat`) are read from `lines/` at runtime during atmosphere conversion. This design means the Python pipeline operates on the same underlying physical data as the Fortran ground truth, isolating any differences to the numerical implementation rather than input data discrepancies.

# Research impact statement

PySynthe completes an end-to-end Python pipeline for stellar atmosphere modeling and spectrum synthesis. A Python reimplementation of ATLAS12 already provides the atmospheric structure component [Ting et al.](https://github.com/tingyuansen/kurucz); PySynthe provides the missing synthesis step, meaning the full chain from stellar parameters to emergent spectrum is now available without any Fortran dependency.

The primary impact of this work is enabling direct integration of physically accurate spectrum synthesis with machine learning frameworks. Because PySynthe is pure Python, it can be coupled directly with frameworks such as PyTorch to generate spectra on-the-fly during training.

Validation against 100 randomly drawn atmosphere models spans a wide range of effective temperatures, surface gravities, and metallicities, including the challenging regimes of cool giants, metal-poor dwarfs, and hot O-type stars. Sub-percent fractional flux agreement is achieved across all wavelengths for every sample, with no Fortran required at runtime. This establishes PySynthe as suitable for production use in both large-scale spectroscopic surveys and precision stellar abundance analyses.

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

A substantial part of this reimplementation was developed with AI assistance (Cursor, using a collection of language models), given the ground-truth Fortran code provided by Bob Kurucz. Beyond accelerating development, this project demonstrates a new mode of scientific software creation: AI coding agents, given sufficient context from legacy Fortran sources, can produce validated Python reimplementations that faithfully reproduce the original numerical behavior. We anticipate this workflow will generalize to other legacy astrophysical codes.

# Acknowledgements
