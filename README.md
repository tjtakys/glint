<p align="center">
  <img src="docs/logo/logo.png" style="width:420px; display:block; margin:auto;">
</p>

# GLINT

**GLINT (Gravitational Lensing Interferometric Tomography)**

GLINT is an image- and visibility-domain reconstruction framework designed for galaxy–galaxy strong lensing observed with interferometric data (e.g., ALMA).

The code performs forward modeling to recover the intrinsic kinematic structure of lensed galaxies using parametric kinematic models. In addition, GLINT supports pixelized source reconstruction through semi-linear inversion, enabling non-parametric recovery of the source-plane emission while keeping the lens model fixed.

The framework is particularly suited for spatially and spectrally resolved line observations of strongly lensed galaxies.

## Requirements

The main dependencies of GLINT include:

- modular CASA
- NumPy
- SciPy
- Matplotlib
- emcee
- Astropy
- FINUFFT

## Installation

GLINT requires **Python 3.10 or later**.

```bash
git clone https://github.com/your-username/glint.git
cd glint
pip install -e .